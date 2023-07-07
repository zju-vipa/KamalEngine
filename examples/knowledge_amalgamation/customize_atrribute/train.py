import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from kamal import vision, engine, utils, amalgamation, metrics, callbacks
from kamal.vision import sync_transforms as sT
from torch import nn, optim
from torch.optim import lr_scheduler
import torch, time
from torch.utils.tensorboard import SummaryWriter
from kamal.vision.datasets.CelebA import *
from kamal.vision.models.classification.resnet_customize import *
from torch.autograd import Variable
from kamal.amalgamation.customize_attribute import CUSTOMIZE_COMPONENT_Amalgamator,CUSTOMIZE_TARGET_Amalgamator
import argparse
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

def main():
    parser = argparse.ArgumentParser(
        description='Resnet knowledge training with multiTeacher')
    parser.add_argument('--target_attribute', default='hair', type=str,
                        help='target_attribute')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training')
    parser.add_argument('--source_saveEpoch', default=40, type=int,
                        help='initial source save epoch')
    parser.add_argument('--component_saveEpoch', default=50, type=int,
                        help='initial component save epoch')
    parser.add_argument('--target_epoch', default=50, type=int,
                        help='initial target epoch')
    parser.add_argument('--data_root', default='./data/CelebA', type=str,help='initial data root')
    parser.add_argument('--save_root', default='examples/knowledge_amalgamation/customize_atrribute/', type=str,
                        help='initial save root')
    parser.add_argument('--sourcenets_root', default='examples/knowledge_amalgamation/customize_atrribute/snapshot/sources/', type=str,
                        help='initial sourcenets root')
    parser.add_argument('--sourcedata_txtroot', default='examples/knowledge_amalgamation/customize_atrribute/data/sources/', type=str,
                        help='initial customize type')
    parser.add_argument('--componentnets_root', default='examples/knowledge_amalgamation/customize_atrribute/snapshot/components/', type=str,
                        help='initial componentnets root')
    parser.add_argument('--target_root', default='examples/knowledge_amalgamation/customize_atrribute/snapshot/target', type=str,
                        help='initial customize type')
    parser.add_argument('--log_dir', default='examples/knowledge_amalgamation/customize_atrribute/run/', type=str,
                        help='initial log dir')
    args = parser.parse_args()

    #if you don't set parameter in terminal you can set here
    args.target_attribute = 'hair'
    args.source_saveEpoch = 40
    args.component_saveEpoch = 50
    args.data_root = './data/CelebA'
    args.save_root = 'examples/knowledge_amalgamation/customize_atrribute/'
    args.sourcenets_root = os.path.join(args.save_root,'snapshot','sources/')    
    args.sourcedata_root = os.path.join(args.save_root,'data','sources/')
    args.componentnets_root = os.path.join(args.save_root,'snapshot','components/')
    args.target_root = os.path.join(args.save_root,'snapshot','target/')

    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    use_cuda = True
    target_attribute = args.target_attribute
    
    source_ids = source2target_id_dic(args.target_attribute)  #['1,2','3,4','5,6,7','8,9']   
    source_saveEpoch = args.source_saveEpoch 
    cur_targetnet_sourcenets_dic = source2target_attributes_dic[target_attribute]
    
    component_attributes = component2target_attributes_dic[target_attribute] #['Black_Hair', 'Blond_Hair', 'Brown_Hair',  'Bangs']
    num_components = len(component_attributes)
    component_saveEpoch = args.component_saveEpoch
    target_idxs = [get_attri_idx(item) for item in component_attributes]
    
    layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    
    source_channel = [64, 64, 128, 256, 512]
    component_channel = [64, 64, 128, 256, 512]
    s2c_common_channel = [32, 32, 64, 128, 256]
    
    target_channel = [90, 90, 180, 360, 720]
    c2t_common_channel = [64, 64, 128, 256, 512]

    lr_source = [1e-4, 1e-4, 1e-5, 1e-4, 1e-4]
    lr_component = [1e-4, 1e-4, 1e-5, 1e-4, 1e-4, 1e-6]
    lr_target = 1e-4
    # prepare dataloader
    train_loader = get_dataloader(args.data_root,'train', batch_size=args.batch_size, is_part=False,
                                  shuffle=False, get_name=True)
    test_loader = get_dataloader(args.data_root,'test', batch_size=args.batch_size, shuffle=False)

    
    #-------------------train component net --------------------------
    for comnet_id in range(3,len(component_attributes)):          #['Black_Hair', 'Blond_Hair', 'Brown_Hair',  'Bangs']
        cur_component_attribute = component_attributes[comnet_id]  #'Black_Hair' or 'Blond_Hair'or 'Brown_Hair'or 'Bangs'
        args.component_attribute = cur_component_attribute
        cur_source_select = source_ids[comnet_id].split(',')           #['0','1']
        
        #prepare component model--student model
        component_model = component_resnet18(channel_num=component_channel, num_classes=2)
        component_model = component_model.cuda()
        
        #prepare source model-- teacher models
        source_models, source_outlabel_id = get_sourcenet_models(comnet_id,cur_targetnet_sourcenets_dic,cur_component_attribute,source_ids,source_saveEpoch,args.sourcenets_root,source_channel)
        if use_cuda:
            for t in source_models:
                t.cuda().eval()
        
        #select model by perparing data for selected souece model
        datatxt_path, component_attribute_idxs= divide_data_sourcenet(cur_component_attribute, train_loader, source_outlabel_id, cur_source_select, source_models, args.sourcedata_root)
        source_dataloaders = get_divided_dataloader(datatxt_path,  args.batch_size, cur_source_select, 'train', shuffle=True)
  
        #prepar distiller
        distiller_sources,distiller_components = get_s2c_distill_models(len(cur_source_select),component_channel,source_channel,s2c_common_channel)
        if use_cuda:
            for s in distiller_sources:
                s.cuda()
            for c in distiller_components:
                c.cuda()

        # prepare criterions
        criterions = get_criterions(len(cur_source_select),len(layer_names))
        if use_cuda:
            for criterion in criterions:
                criterion = criterion.cuda()
         
        # prepare optimizer
        optimizer = get_optimizer(len(cur_source_select),component_model,lr_target,distiller_sources,lr_source,distiller_components,lr_component)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25], gamma=0.5)  
        special_module_idxs = range(0, len(layer_names)+1)
        
        # prepare amalgamator
        logger = utils.logger.get_logger('customise_%s' % ('attribute'))
        tb_writer = SummaryWriter(log_dir='./run/customize_%s-%s' %
                        ('attribute', time.asctime().replace(' ', '_')))
        trainer = CUSTOMIZE_COMPONENT_Amalgamator(logger, tb_writer)

        trainer.setup(args,component_net=component_model, source_nets=source_models,teacher_select=cur_source_select,\
                      target_idxs = component_attribute_idxs,target_no=source_outlabel_id, distill_teachers=distiller_sources,\
                    distill_students=distiller_components,dataloader=source_dataloaders,test_loader=test_loader,\
                    layer_names=layer_names, special_module_idxs=special_module_idxs,criterion=criterions, optimizer=optimizer, device=device )
        trainer.add_callback( 
            engine.DefaultEvents.AFTER_STEP(every=10), 
            callbacks=callbacks.MetricsLogging(keys=('tensor_total_loss','lr')))
        trainer.add_callback(
            engine.DefaultEvents.AFTER_STEP,
            callbacks=callbacks.LRSchedulerCallback(schedulers=[scheduler]))
        
        len_iter_list = []
        for it in range(len(source_dataloaders)):
            print('t{} len:{}'.format(it,len(source_dataloaders[it])))
            len_iter_list.append(len(source_dataloaders[it]))
        epoch_iter = min(len_iter_list)
        trainer.run(start_iter=0, max_iter=len(train_loader)*args.component_saveEpoch,epoch_length = len(train_loader)) #epoch_iter*component_saveEpoch

        
    #----------------train target net---------------------
    #prepare target model
    target_model = target_resnet18(num_classes=2, target_attributes=component_attributes, channel_num=target_channel)
    target_model = target_model.cuda()
    
    #prepare component model 
    component_models = get_component_models(component_attributes,source_ids,component_saveEpoch,component_channel,args.componentnets_root)
    if use_cuda:
        for t in component_models:
            t.cuda().eval()
    
    #target net don't slelect,so dataloader is train_loader

    #prepar distiller
    distiller_targets = get_c2t_distill_models(num_components,target_channel,c2t_common_channel)
    if use_cuda:
        for t in distiller_targets:
            t.cuda()
    #prepare criterions
    criterions =  get_criterions(num_components,len(layer_names))
    if use_cuda:
        for criterion in criterions:
            criterion = criterion.cuda()
   
    
    # prepare Optimizer 
    optimizers = get_target_optimizer(num_components,target_model,lr_target,distiller_targets,lr_component)

    #prepare scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25], gamma=0.5)  

    special_module_idxs = range(0, len(layer_names)+1)
    # prepare amalgamator
    logger = utils.logger.get_logger('customise_%s' % ('attribute'))
    tb_writer = SummaryWriter(log_dir='./run/customize_%s-%s' %
                    ('attribute', time.asctime().replace(' ', '_')))
    trainer = CUSTOMIZE_TARGET_Amalgamator(logger, tb_writer)

    trainer.setup(args, target_net=target_model, component_nets=component_models, target_idxs = target_idxs,component_attributes = component_attributes,\
                  distill_target=distiller_targets, dataloader=train_loader, test_loader=test_loader,  layer_names=layer_names,\
                      special_module_idxs=special_module_idxs,criterion=criterions, optimizer=optimizers, device=device )
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_STEP(every=10), 
        callbacks=callbacks.MetricsLogging(keys=('tensor_total_loss','lr')))
    trainer.add_callback(
            engine.DefaultEvents.AFTER_STEP,
            callbacks=callbacks.LRSchedulerCallback(schedulers=[scheduler]))
    
    trainer.run(start_iter=0, max_iter=len(train_loader)*args.target_epoch,epoch_length = len(train_loader) ) #len(train_loader)*args.num_epoch

if __name__ == '__main__':
    main()


    


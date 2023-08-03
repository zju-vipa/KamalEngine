import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from kamal import vision, engine, utils, amalgamation, metrics, callbacks
from kamal.vision import sync_transforms as sT
from torch import nn, optim
from torch.optim import lr_scheduler
import torch, time
from torch.utils.tensorboard import SummaryWriter
from kamal.vision.models.classification.resnet_customize import *
from torch.autograd import Variable
from kamal.amalgamation.customize_class import CUSTOMIZE_COMPONENT_Amalgamator_class,CUSTOMIZE_TARGET_Amalgamator_class
from kamal.amalgamation.customize_attribute import CUSTOMIZE_COMPONENT_Amalgamator_attribute,CUSTOMIZE_TARGET_Amalgamator_attribute
from kamal.vision.datasets.CelebA import *
from kamal.vision.datasets.customize_class_data import *

from customize_utils import *
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main_attribute(args):
    #if you don't set parameter in terminal you can set here
    args.target_attribute = 'hair'
    args.source_saveEpoch = 40
    args.component_saveEpoch = 50
    args.data_root = '/nfs/yxy/data/CelebA'
    args.save_root = 'examples/amalgamation/customize_ka/attribute/'
    args.sourcenets_root = os.path.join(args.save_root,'snapshot','sources/')    
    args.sourcedata_root = os.path.join(args.save_root,'data','sources/')
    args.componentnets_root = os.path.join(args.save_root,'snapshot','components/')
    args.target_root = os.path.join(args.save_root,'snapshot','target/')

    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    use_cuda = True
    target_attribute = args.target_attribute
    
    source_ids = source2target_id_dic[args.target_attribute] #['1,2','3,4','5,6,7','8,9']   
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
    train_loader = get_dataloader_attribute(args.data_root,'train', batch_size=args.batch_size, is_part=False,
                                  shuffle=False, get_name=True)
    test_loader = get_dataloader_attribute(args.data_root,'test', batch_size=args.batch_size, shuffle=False)

    
    #-------------------train component net --------------------------
    for comnet_id in range(3,len(component_attributes)):          #['Black_Hair', 'Blond_Hair', 'Brown_Hair',  'Bangs']
        cur_component_attribute = component_attributes[comnet_id]  #'Black_Hair' or 'Blond_Hair'or 'Brown_Hair'or 'Bangs'
        args.component_attribute = cur_component_attribute
        cur_source_select = source_ids[comnet_id].split(',')           #['0','1']
        
        #prepare component model--student model
        component_model = component_resnet18(channel_num=component_channel, num_classes=2)
        component_model = component_model.cuda()
        
        #prepare source model-- teacher models
        source_models, source_outlabel_id = get_sourcenet_models_attribute(comnet_id,cur_targetnet_sourcenets_dic,cur_component_attribute,source_ids,source_saveEpoch,args.sourcenets_root,source_channel)
        if use_cuda:
            for t in source_models:
                t.cuda().eval()
        
        #select model by perparing data for selected souece model
        datatxt_path, component_attribute_idxs= divide_data_sourcenet_attribute(cur_component_attribute, train_loader, source_outlabel_id, cur_source_select, source_models, args.sourcedata_root)
        source_dataloaders = get_divided_dataloader_attribute(datatxt_path,  args.batch_size, cur_source_select, 'train', shuffle=True)
  
        #prepar distiller
        distiller_sources,distiller_components = get_s2c_distill_models_attribute(len(cur_source_select),component_channel,source_channel,s2c_common_channel)
        if use_cuda:
            for s in distiller_sources:
                s.cuda()
            for c in distiller_components:
                c.cuda()

        # prepare criterions
        criterions = get_criterions_attribute(len(cur_source_select),len(layer_names))
        if use_cuda:
            for criterion in criterions:
                criterion = criterion.cuda()
         
        # prepare optimizer
        optimizer = get_optimizer_attribute(len(cur_source_select),component_model,lr_target,distiller_sources,lr_source,distiller_components,lr_component)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25], gamma=0.5)  
        special_module_idxs = range(0, len(layer_names)+1)
        
        # prepare amalgamator
        logger = utils.logger.get_logger('customise_%s' % ('attribute'))
        tb_writer = SummaryWriter(log_dir='./run/customize_%s-%s' %
                        ('attribute', time.asctime().replace(' ', '_')))
        trainer = CUSTOMIZE_COMPONENT_Amalgamator_attribute(logger, tb_writer)

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
    component_models = get_component_models_attribute(component_attributes,source_ids,component_saveEpoch,component_channel,args.componentnets_root)
    if use_cuda:
        for t in component_models:
            t.cuda().eval()
    
    #target net don't slelect,so dataloader is train_loader

    #prepar distiller
    distiller_targets = get_c2t_distill_models_attribute(num_components,target_channel,c2t_common_channel)
    if use_cuda:
        for t in distiller_targets:
            t.cuda()
    #prepare criterions
    criterions =  get_criterions_attribute(num_components,len(layer_names))
    if use_cuda:
        for criterion in criterions:
            criterion = criterion.cuda()
   
    
    # prepare Optimizer 
    optimizers = get_target_optimizer_attribute(num_components,target_model,lr_target,distiller_targets,lr_component)

    #prepare scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25], gamma=0.5)  

    special_module_idxs = range(0, len(layer_names)+1)
    # prepare amalgamator
    logger = utils.logger.get_logger('customise_%s' % ('attribute'))
    tb_writer = SummaryWriter(log_dir='./run/customize_%s-%s' %
                    ('attribute', time.asctime().replace(' ', '_')))
    trainer = CUSTOMIZE_TARGET_Amalgamator_attribute(logger, tb_writer)

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

def main_class(args):
    
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    use_cuda = True
    # if you don't set parameter in terminal, you can set here
    args.main_class = 'airplane'     #'airplane' or 'car' or 'dog' or 'cub'
    args.aux_class = 'car'           #'airplane' or 'car' or 'dog' or 'cub'
    args.main_parts = '1,2'          #'1' or '2'
    args.aux_parts = '1,2,3,4'        #'1' or '2' or '3' or '4'
    args.num_mainclass = dataset_to_cls_num[args.main_class]          #100 or 196 or 120 or 200
    args.num_auxclass = dataset_to_cls_num[args.aux_class]           #100/4 or 196/4 or 120/4 or 200/4
    args.batch_size = 64

    args.data_root = '/nfs/yxy/data/'
    args.save_root = './class/'
    
    data_dir = args.data_root+'{}/'.format(args.main_class)

    main_parts_list = args.main_parts.split(',') #['1','2']
    aux_parts_list = args.aux_parts.split(',')   #['1','2','3','4']

    
    args.component_saveEpoch = [1,1]
       
    args.source_saveEpoch = [1,1,1,1]
    layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
    
    source_channel = [64, 64, 128, 256, 512]
    component_channel = [64, 64, 128, 256, 512]
    s2c_common_channel = [32, 32, 64, 128, 256]
    
    target_channel = [72, 72, 144, 288, 576]
    c2t_common_channel = [64, 64, 128, 256, 512]

    lr_source = [1e-3, 1e-3, 1e-3, 1e-3, 8e-4]
    lr_component = [1e-3, 1e-3, 1e-3, 1e-3, 8e-4]
    lr_target = 5e-4
    
    # prepare model
    args.sourcenets_root = os.path.join(args.save_root,'snapshot','sources/')    
    args.sourcedata_root = os.path.join(args.save_root,'data','sources/')
    args.componentnets_root = os.path.join(args.save_root,'snapshot','components/')
    args.target_root = os.path.join(args.save_root,'snapshot','target/')

    num_source_tocomp =  2
    
        
    # prepare amalgamator
    logger = utils.logger.get_logger('customise_%s' % ('attribute'))
    tb_writer = SummaryWriter(log_dir='./class/run/customize_%s-%s' %
                    ('class', time.asctime().replace(' ', '_')))
    trainer1 = CUSTOMIZE_COMPONENT_Amalgamator_class(logger, tb_writer)
    print('count:',torch.cuda.device_count())
    print('id 0:',torch.cuda.get_device_name(0))
    # print('id 1:',torch.cuda.get_device_name(1))
    # print('id 2:',torch.cuda.get_device_name(2))
    print('cur id :',torch.cuda.current_device())
    # #-------------------train component net --------------------------
    for comnet_id in range(len(main_parts_list)):          #['1', '2']
        cur_com_part_num = int(main_parts_list[comnet_id]) # 1 or 2
        cur_aux_parts_num = [int(aux_parts_list[2*comnet_id]),int(aux_parts_list[2*comnet_id+1])] # ['1', '2'] or ['3', '4']
        cur_source_saveEpoch = [args.source_saveEpoch[2*comnet_id],args.source_saveEpoch[2*comnet_id+1]]    # [40,40] 
        cur_num_sources = len(cur_aux_parts_num)
        #prepare data
        train_loader = get_dataloader_path_class(data_dir, 'train', batch_size=args.batch_size,
                                  shuffle=True, part_num=cur_com_part_num,is_part=True)
        test_loader = get_dataloader_class(data_dir, 'test', batch_size=args.batch_size,
                                  shuffle=True, part_num=cur_com_part_num,is_part=True)
       
        #prepare component model--student model
        component_model = resnet18(pretrained=True, channel_num=component_channel,
                        num_classes=args.num_mainclass // 2)
        component_model = component_model.cuda()

        #prepare source model-- teacher models
        pre_cls_num = args.num_mainclass // 2
        source_models = get_sourcenet_models_class(args.main_class,cur_com_part_num,pre_cls_num,args.aux_class,cur_aux_parts_num,cur_source_saveEpoch,args.sourcenets_root,source_channel)
        if use_cuda:
            for t in source_models:
                t.cuda().eval()
        #select model by perparing data for selected souece model
        path = divide_data_sourcenet_class(args.main_class,cur_com_part_num,args.aux_class,cur_aux_parts_num,train_loader, source_models, args.sourcedata_root, use_cuda=True)
        source_dataloaders = get_divided_dataloader_class(data_dir,path,cur_com_part_num,cur_aux_parts_num, args.batch_size,'train', shuffle=True)
      
        #prepar distiller
        distiller_sources,distiller_components = get_s2c_distill_models_class(num_source_tocomp ,component_channel,source_channel,s2c_common_channel)
        if use_cuda:
            for s in distiller_sources:
                s.cuda()
            for c in distiller_components:
                c.cuda()

        # prepare criterions
        criterion = nn.MSELoss(reduction='mean')
        criterion = criterion.cuda()

        # prepare optimizer
        optimizer = get_optimizer_class(num_source_tocomp ,component_model,lr_target,distiller_sources,lr_source,distiller_components,lr_source)
        if args.main_class == 'dog':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[50, 100, 150], gamma=0.5)
        elif args.main_class == 'airplane':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[20, 50, 100], gamma=0.5)

        special_module_idxs = range(0, len(layer_names)+1)
        trainer1.add_callback( 
                        engine.DefaultEvents.AFTER_STEP(every=10), 
                        callbacks=callbacks.MetricsLogging(keys=('tensor_total_loss','lr')))
        trainer1.add_callback(
                        engine.DefaultEvents.AFTER_STEP,
                        callbacks=callbacks.LRSchedulerCallback(schedulers=[scheduler]))

        trainer1.setup(args, component_net=component_model,component_part=cur_com_part_num, source_nets=source_models,\
                    aux_parts=cur_aux_parts_num,distill_students=distiller_components,distill_teachers=distiller_sources,\
                    dataloader=source_dataloaders,test_loader=test_loader,layer_names=layer_names, \
                    special_module_idxs=special_module_idxs,criterion=criterion,optimizer=optimizer, device=device )
        
        
        len_iter_list = []
        for it in range(len(source_dataloaders)):
            print('t{} len:{}'.format(it,len(source_dataloaders[it])))
            len_iter_list.append(len(source_dataloaders[it]))
        epoch_iter = min(len_iter_list)
        trainer1.run(start_iter=0, max_iter=epoch_iter*args.component_saveEpoch[comnet_id],epoch_length = epoch_iter) #epoch_iter*component_saveEpoch[comnet_id]
        
        trainer1.remove_callback(
                        engine.DefaultEvents.AFTER_STEP,
                        callback=callbacks.LRSchedulerCallback(schedulers=[scheduler]))
        trainer1.remove_callback(
                        engine.DefaultEvents.AFTER_STEP(every=10), 
                        callback=callbacks.MetricsLogging(keys=('tensor_total_loss','lr')))

        
    #----------------train target net---------------------

    #prepare target model
    component_attributes = main_parts_list #['1','2']
    target_model = resnet18_multitask(pretrained=False, channel_nums=target_channel,
                                 target_attributes=component_attributes, num_classes=args.num_mainclass //2)
    target_model = target_model.cuda()
    #prepare component model 
    component_models = get_component_models_class(args.main_class,main_parts_list,args.aux_class,aux_parts_list,args.component_saveEpoch,args.componentnets_root,component_channel)
    if use_cuda:
        for t in component_models:
            t.cuda().eval()
    
    #target net don't slelect,so dataloader is train_loader
    train_loader = get_dataloader_class(data_dir, 'train', batch_size=args.batch_size,
                                  is_part=False, shuffle=True)

    test_loaders = [get_dataloader_class(data_dir, 'test', batch_size=args.batch_size,
                                   is_part=True, part_num=1, shuffle=False),
                    get_dataloader_class(data_dir, 'test', batch_size=args.batch_size,
                                   is_part=True, part_num=2, shuffle=False)]
    
    #prepar distiller
    num_components = len(component_attributes)
    distiller_targets = get_c2t_distill_models_class(num_components,target_channel,c2t_common_channel)
    if use_cuda:
        for t in distiller_targets:
            t.cuda()
    #prepare criterions
    criterion = nn.MSELoss(reduction='mean')
    criterion = criterion.cuda()
   
    # prepare Optimizer 
    optimizers = get_target_optimizer_class(num_components,target_model,lr_target,distiller_targets,lr_component)

    if args.main_class == 'dog':
            scheduler = lr_scheduler.MultiStepLR(optimizers,
                                                milestones=[50, 100, 150], gamma=0.5)
    elif args.main_class == 'airplane':
        scheduler = lr_scheduler.MultiStepLR(optimizers,
                                                milestones=[20, 50, 100], gamma=0.5)

    special_module_idxs = range(0, len(layer_names)+1)
    # prepare amalgamator
    logger = utils.logger.get_logger('customise_%s' % ('class'))
    tb_writer = SummaryWriter(log_dir='./run/customize_%s-%s' %
                    ('attribute', time.asctime().replace(' ', '_')))
    trainer2 = CUSTOMIZE_TARGET_Amalgamator_class(logger, tb_writer)

    trainer2.setup(args, target_net=target_model, component_nets=component_models, component_attributes = component_attributes,\
                  distill_target=distiller_targets, dataloader=train_loader, test_loaders=test_loaders,  layer_names=layer_names,\
                      special_module_idxs=special_module_idxs,criterion=criterion, optimizer=optimizers, device=device )
    trainer2.add_callback( 
        engine.DefaultEvents.AFTER_STEP(every=10), 
        callbacks=callbacks.MetricsLogging(keys=('tensor_total_loss','lr')))
    trainer2.add_callback(
        engine.DefaultEvents.AFTER_STEP,
        callbacks=callbacks.LRSchedulerCallback(schedulers=[scheduler]))
    
    trainer2.run(start_iter=0, max_iter=len(train_loader)*args.target_epoch,epoch_length = len(train_loader) ) #len(train_loader)*args.target_epoch



def main():
    parser = argparse.ArgumentParser(
        description='Resnet knowledge training with multiTeacher')
    parser.add_argument('--type', default='class', type=str,
                        help="choose train process, 'attribute' or 'class'")
    parser.add_argument('--target_attribute', default='hair', type=str,
                        help='target_attribute for attribute training process')
    parser.add_argument('--target_class', default='airplane', type=str,
                        help='target_class for class training process')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training')
    parser.add_argument('--source_saveEpoch', default=40, type=int,
                        help='initial source save epoch')
    parser.add_argument('--component_saveEpoch', default=50, type=int,
                        help='initial component save epoch')
    parser.add_argument('--target_epoch', default=50, type=int,
                        help='initial target epoch')
    parser.add_argument('--data_root', default='./data/', type=str,
                        help='initial data root')
    parser.add_argument('--save_root', default='./attribute/', type=str,
                        help='./class/ for class training process')
    parser.add_argument('--sourcenets_root', default='./attribute/snapshot/sources/', type=str,
                        help='./class/snapshot/sources/ for class training process')
    parser.add_argument('--sourcedata_txtroot', default='./attribute/data/sources/', type=str,
                        help='./class/data/sources/ for class training process')
    parser.add_argument('--componentnets_root', default='./attribute/snapshot/components/', type=str,
                        help='./class/snapshot/components/ for class training process')
    parser.add_argument('--target_root', default='./attribute/snapshot/target', type=str,
                        help='./class/snapshot/target/for class training process')
    parser.add_argument('--log_dir', default='./attribute/run/', type=str,
                        help='./class/run/ for class training process')
    args = parser.parse_args()
    if args.type == 'attribute':
        main_attribute(args)
    elif args.type == 'class':
        main_class(args)


if __name__ == '__main__':
    main()
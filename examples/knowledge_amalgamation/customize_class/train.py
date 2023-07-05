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
from kamal.amalgamation.customize_class import CUSTOMIZE_COMPONENT_Amalgamator,CUSTOMIZE_TARGET_Amalgamator

from kamal.vision.datasets.customize_class_data import *
from utils import *
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    parser = argparse.ArgumentParser(
        description='Resnet knowledge training with multiTeacher')
    parser.add_argument('--target_class', default='airplane', type=str,
                        help='target_class')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_epoch', default=300, type=str,
                        help='epoch number')
    parser.add_argument('--data_root', default='./examples/knowledge_amalgamation/customize_class/', type=str,
                        help='initial data root')
    parser.add_argument('--save_root', default='./examples/knowledge_amalgamation/customize_class/', type=str,
                        help='initial save root')
    parser.add_argument('--sourcenets_root', default='./snapshot/sources/', type=str,
                        help='initial sourcenets root')
    parser.add_argument('--sourcedata_root', default='./data/sources/', type=str,
                        help='initial source data root')
    parser.add_argument('--componentnets_root', default='./snapshot/components/', type=str,
                        help='initial componentnets root')
    parser.add_argument('--target_root', default='./examples/knowledge_amalgamation/customize_atrribute/snapshot/', type=str,
                        help='initial target root')
    parser.add_argument('--log_dir', default='./run/', type=str,
                        help='initial log dir')
    args = parser.parse_args()


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
    args.save_root = '/home/yxy/kacode_pr1/KamalEngine/examples/knowledge_amalgamation/customize_class/'
    
    data_dir = args.data_root+'{}/'.format(args.main_class)

    main_parts_list = args.main_parts.split(',') #['1','2']
    aux_parts_list = args.aux_parts.split(',')   #['1','2','3','4']

    
    component_saveEpoch = [400,400]
       
    source_saveEpoch = [300,300,300,300]
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
    tb_writer = SummaryWriter(log_dir='./run/customize_%s-%s' %
                    ('attribute', time.asctime().replace(' ', '_')))
    trainer1 = CUSTOMIZE_COMPONENT_Amalgamator(logger, tb_writer)
    print('count:',torch.cuda.device_count())
    print('id 0:',torch.cuda.get_device_name(0))
    # print('id 1:',torch.cuda.get_device_name(1))
    # print('id 2:',torch.cuda.get_device_name(2))
    print('cur id :',torch.cuda.current_device())
    # #-------------------train component net --------------------------
    for comnet_id in range(len(main_parts_list)):          #['1', '2']
        cur_com_part_num = int(main_parts_list[comnet_id]) # 1 or 2
        cur_aux_parts_num = [int(aux_parts_list[2*comnet_id]),int(aux_parts_list[2*comnet_id+1])] # ['1', '2'] or ['3', '4']
        cur_source_saveEpoch = [source_saveEpoch[2*comnet_id],source_saveEpoch[2*comnet_id+1]]    # [40,40] 
        cur_num_sources = len(cur_aux_parts_num)
        #prepare data
        train_loader = get_dataloader_path(data_dir, 'train', batch_size=args.batch_size,
                                  shuffle=True, part_num=cur_com_part_num,is_part=True)
        test_loader = get_dataloader(data_dir, 'test', batch_size=args.batch_size,
                                  shuffle=True, part_num=cur_com_part_num,is_part=True)
       
        #prepare component model--student model
        component_model = resnet18(pretrained=True, channel_num=component_channel,
                        num_classes=args.num_mainclass // 2)
        component_model = component_model.cuda()

        #prepare source model-- teacher models
        pre_cls_num = args.num_mainclass // 2
        source_models = get_sourcenet_models(args.main_class,cur_com_part_num,pre_cls_num,args.aux_class,cur_aux_parts_num,cur_source_saveEpoch,args.sourcenets_root,source_channel)
        if use_cuda:
            for t in source_models:
                t.cuda().eval()
        #select model by perparing data for selected souece model
        path = divide_data_sourcenet(args.main_class,cur_com_part_num,args.aux_class,cur_aux_parts_num,train_loader, source_models, args.sourcedata_root, use_cuda=True)
        source_dataloaders = get_divided_dataloader(data_dir,path,cur_com_part_num,cur_aux_parts_num, args.batch_size,'train', shuffle=True)
      
        #prepar distiller
        distiller_sources,distiller_components = get_s2c_distill_models(num_source_tocomp ,component_channel,source_channel,s2c_common_channel)
        if use_cuda:
            for s in distiller_sources:
                s.cuda()
            for c in distiller_components:
                c.cuda()

        # prepare criterions
        criterion = nn.MSELoss(reduction='mean')
        criterion = criterion.cuda()

        # prepare optimizer
        optimizer = get_optimizer(num_source_tocomp ,component_model,lr_target,distiller_sources,lr_source,distiller_components,lr_source)
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
        trainer1.run(start_iter=0, max_iter=epoch_iter*component_saveEpoch[comnet_id],epoch_length = epoch_iter) #epoch_iter*component_saveEpoch[comnet_id]
        
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
    component_models = get_component_models(args.main_class,main_parts_list,args.aux_class,aux_parts_list,component_saveEpoch,args.componentnets_root,component_channel)
    if use_cuda:
        for t in component_models:
            t.cuda().eval()
    
    #target net don't slelect,so dataloader is train_loader
    train_loader = get_dataloader(data_dir, 'train', batch_size=args.batch_size,
                                  is_part=False, shuffle=True)

    test_loaders = [get_dataloader(data_dir, 'test', batch_size=args.batch_size,
                                   is_part=True, part_num=1, shuffle=False),
                    get_dataloader(data_dir, 'test', batch_size=args.batch_size,
                                   is_part=True, part_num=2, shuffle=False)]
    
    #prepar distiller
    num_components = len(component_attributes)
    distiller_targets = get_c2t_distill_models(num_components,target_channel,c2t_common_channel)
    if use_cuda:
        for t in distiller_targets:
            t.cuda()
    #prepare criterions
    criterion = nn.MSELoss(reduction='mean')
    criterion = criterion.cuda()
   
    # prepare Optimizer 
    optimizers = get_target_optimizer(num_components,target_model,lr_target,distiller_targets,lr_component)

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
    trainer2 = CUSTOMIZE_TARGET_Amalgamator(logger, tb_writer)

    trainer2.setup(args, target_net=target_model, component_nets=component_models, component_attributes = component_attributes,\
                  distill_target=distiller_targets, dataloader=train_loader, test_loaders=test_loaders,  layer_names=layer_names,\
                      special_module_idxs=special_module_idxs,criterion=criterion, optimizer=optimizers, device=device )
    trainer2.add_callback( 
        engine.DefaultEvents.AFTER_STEP(every=10), 
        callbacks=callbacks.MetricsLogging(keys=('tensor_total_loss','lr')))
    trainer2.add_callback(
        engine.DefaultEvents.AFTER_STEP,
        callbacks=callbacks.LRSchedulerCallback(schedulers=[scheduler]))
    
    trainer2.run(start_iter=0, max_iter=len(train_loader)*args.num_epoch,epoch_length = len(train_loader) ) #len(train_loader)*args.num_epoch

if __name__ == '__main__':
    main()
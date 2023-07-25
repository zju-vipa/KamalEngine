import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

import argparse
import torch
import time
import pickle
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms as T
from visdom import Visdom
from kamal import engine, utils, slim, vision, metrics, callbacks
from kamal.utils.logger import Logger
from kamal.vision.models.classification.vgg_block import vgg_stock, vgg_bw, cfgs, split_block



def parse(dataset,flag,num_per_class):
    args = argparse.Namespace()
    if dataset == 'CIFAR10':
        args.dataset = 'CIFAR10'
        args.num_class = 10
        args.ckpt = './ckpt/teacher/vgg16-blockwise-cifar10.pth'
        args.dataset_mean = [0.4914, 0.4822, 0.4465]
        args.dataset_std = [0.2023, 0.1994, 0.2010]
               
        if flag == 'block':
            args.lrs_s = [0.00025, 0.00025, 0.00025] 
            args.lrs_adapt_t2s = [0.0025, 0.0025]
            args.lrs_adapt_s2t = [0.0025, 0.0025]
            args.num_epoch = [100, 60, 60]
        else:
            args.lrs_s = [0.0001, 0.0001] 
            args.lrs_adapt_t2s = [0.0025, 0.0025]
            args.lrs_adapt_s2t = [0.0025, 0.0025]
            args.num_epoch = [100, 100]
    else:
        args.dataset = 'CIFAR100'
        args.num_class = 100
        args.ckpt = './ckpt/teacher/vgg16-blockwise-cifar100.pth'
        args.dataset_mean = [0.5071, 0.4867, 0.4408]
        args.dataset_std = [0.2675, 0.2565, 0.2761]
        if flag == 'block':
            args.lrs_s = [0.00025, 0.001, 0.001] 
            args.lrs_adapt_t2s = [0.001, 0.001]
            args.lrs_adapt_s2t = [0.001, 0.001]
            args.num_epoch = [1000, 200, 200]
        else:
            args.lrs_s = [0.00005, 0.00005] 
            args.lrs_adapt_t2s = [0.001, 0.001]
            args.lrs_adapt_s2t = [0.001, 0.001]
            args.num_epoch = [150, 150]
    
    args.data_path = './data/'
    args.num_per_class = num_per_class
    factor = args.num_per_class / 10
    args.batch_size = int(64*factor)
    args.norm_loss = True

    args.lrs_s = [item*factor for item in args.lrs_s]
    args.lrs_adapt_t2s = [item*factor for item in args.lrs_adapt_t2s]
    args.lrs_adapt_s2t = [item*factor for item in args.lrs_adapt_s2t]
    return args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=240)
    parser.add_argument('--num_workers', type=int,
                        default=8, help='num of workers to use')
    parser.add_argument('--distill', type=str, default='graft_kd', choices=[
                        'kd', 'hint', 'attention', 'sp', 'cc', 'vid', 'svd', 'pkt', 'nst', 'rkd','graft_kd'])
    # dataset
    parser.add_argument('--data_root', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'],
                        help='dataset name (default: CIFAR10)')
    parser.add_argument('--num_class', type=int, default=10, choices=[10,100],
                        help='num class (default: 10)')
    parser.add_argument('--build_dataset', type=bool, default=False)
    args = parser.parse_args()
    if args.build_dataset == True:
        vision.datasets.preprocess.build_dataset()
    os.makedirs('./log/', exist_ok=True)
    logger_acc = Logger('log/accuracy-for-various-num_sample.txt')
 
    # distiller setup
    if args.distill == 'graft_kd':
        logger = utils.logger.get_logger('distill_%s' % (args.distill))
        tb_writer = SummaryWriter(log_dir='./run/distill_%s-%s' %
                        (args.distill, time.asctime().replace(' ', '_')))
        distiller_block = slim.GRAFT_BLOCK_Distiller(logger, tb_writer)
        distiller_block.add_callback( 
            engine.DefaultEvents.AFTER_STEP(every=10), 
            callbacks=callbacks.MetricsLogging(keys=('total_loss', 'loss_kld', 'loss_ce', 'loss_additional', 'lr')))
        
        distiller_net = slim.GRAFT_NET_Distiller(logger, tb_writer)
        distiller_net.add_callback( 
                    engine.DefaultEvents.AFTER_STEP(every=10), 
                    callbacks=callbacks.MetricsLogging(keys=('total_loss', 'loss_kld', 'loss_ce', 'loss_additional', 'lr')))
    
    nums = list(range(1,11))
    nums += [20, 50]
    for i in nums:
        logger_acc.write('Num-of-Samples', i)
      
        filename = args.dataset.lower()+'-random-{}-per-class.pkl'.format(i)
        file_path = os.path.join(args.data_root, filename)
        with open(file_path, 'rb') as f:
            train_entry = pickle.load(f)
        
        flag = 'block'
        arg_parse = parse(args.dataset,flag,i)
        os.makedirs('log', exist_ok=True)
        
        #prepare data
        train_transform = T.Compose([T.RandomCrop(32, padding=4),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(mean=arg_parse.dataset_mean,std=arg_parse.dataset_std)])
        test_transform = T.Compose([T.ToTensor(),T.Normalize(mean=arg_parse.dataset_mean,std=arg_parse.dataset_std)])
        train_loader = torch.utils.data.DataLoader(
        vision.datasets.graftkd_cifarfew.CIFARFew(args.data_root,train_entry,transform=train_transform),
        batch_size=arg_parse.batch_size, num_workers=4, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torchvision.datasets.__dict__[arg_parse.dataset](root=arg_parse.data_path, train=False,
                    transform=test_transform,download=True),
        batch_size=256, num_workers=4, shuffle=False)
        # prepare block model
        teacher = vision.models.classification.vgg_stock(cfgs['vgg16'], arg_parse.dataset, arg_parse.num_class)
        params_t = torch.load(arg_parse.ckpt)
        teacher.cuda().eval()
        teacher.load_state_dict(params_t)

        params_s = {}
        for key in params_t.keys():
            key_split = key.split('.')
            if key_split[0] == 'features' and \
                    key_split[1] in ['0', '1', '2']:
                params_s[key] = params_t[key]
        student = vision.models.classification.vgg_bw(cfgs['vgg16-graft'], True, arg_parse.dataset, arg_parse.num_class)
        student.cuda().train()
        student.load_state_dict(params_s, strict=False)
       
        block_graft_ids = [3, 4, 5]
        blocks_s_len = [1, 1, 1]
        blocks_s = [student.features[i] for i in block_graft_ids[:-1]]
        blocks_s += [nn.Sequential(nn.Flatten().cuda(), student.classifier)]

        #prepare adaption
        cfg_blocks_t = vision.models.classification.split_block(cfgs['vgg16'])
        cfg_blocks_s = vision.models.classification.split_block(cfgs['vgg16-graft'])
        num_block = len(block_graft_ids)
        adaptions_t2s = [nn.Conv2d(cfg_blocks_t[block_graft_ids[i]][-2],cfg_blocks_s[block_graft_ids[i]][-2],kernel_size=1).cuda()
                            for i in range(0, num_block - 1)]
        for m in adaptions_t2s:
            utils._utils.init_conv(m)
        adaptions_s2t = [nn.Conv2d(cfg_blocks_s[block_graft_ids[i]][-2],cfg_blocks_t[block_graft_ids[i]][-2],kernel_size=1).cuda()
                            for i in range(0, num_block - 1)]
        for m in adaptions_s2t:
            utils._utils.init_conv(m)
        adaptions_all = [adaptions_t2s, adaptions_s2t]

    #block train
        os.makedirs('./ckpt/student', exist_ok=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # prepare optimizier
        optimizers_s = [optim.Adam(blocks_s[i].parameters(), lr=arg_parse.lrs_s[i])
                    for i in range(0, num_block)]
        optimizers_adapt_t2s = [optim.Adam(adaptions_t2s[i].parameters(),
                                    lr=arg_parse.lrs_adapt_t2s[i])
                            for i in range(0, num_block - 1)]
        optimizers_adapt_s2t = [optim.Adam(adaptions_s2t[i].parameters(),
                                lr=arg_parse.lrs_adapt_s2t[i])
                        for i in range(0, num_block - 1)]
        optimizers_all = [optimizers_s, optimizers_adapt_t2s, optimizers_adapt_s2t]

        distiller_block.setup(arg_parse,blocks_s, teacher=teacher, dataloader=train_loader, test_loader=test_loader, s_blocks_graft_ids=block_graft_ids, s_blocks_len=blocks_s_len,adaptions = adaptions_all,
                    optimizers=optimizers_all, device=device)
        distiller_block.run(arg_parse.num_epoch,start_iter=0)
    #net train
        flag = 'net'
        arg_parse = parse(args.dataset,flag,i)
        train_transform = T.Compose([T.RandomCrop(32, padding=4),T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(mean=arg_parse.dataset_mean,std=arg_parse.dataset_std)])
        test_transform = T.Compose([T.ToTensor(),T.Normalize(mean=arg_parse.dataset_mean,std=arg_parse.dataset_std)])
        #prepare data
        train_loader = torch.utils.data.DataLoader(
        vision.datasets.graft_kd_cifarfew.CIFARFew(args.data_root,train_entry,transform=train_transform),
        batch_size=arg_parse.batch_size, num_workers=4, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torchvision.datasets.__dict__[args.dataset](root=arg_parse.data_path, train=False,
                    transform=test_transform,download=True),
        batch_size=256, num_workers=4, shuffle=False)

        distiller_net.setup(arg_parse,s_blocks=blocks_s, teacher=teacher, dataloader=train_loader, test_loader=test_loader,s_blocks_graft_ids=block_graft_ids, s_blocks_len=blocks_s_len,  adaptions = adaptions_all,
                    device=device)
        distiller_net.run(arg_parse.num_epoch,start_iter=0)



if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main()
 

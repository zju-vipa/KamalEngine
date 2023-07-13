# coding: utf-8
import numpy as np
import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


import torch
from torch import nn

from torch import optim
from torch.optim import lr_scheduler
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

from kamal.vision import sync_transforms as sT
from torch import nn, optim
from kamal.vision.models.classification.resnet_customize import *
from utils import *
import argparse




from kamal.vision.datasets.customize_class_data import *
from utils import make_dir_recursive, train, test, cal_accuracy

num_class_dic = {'airplane':100, 'car':196, 'dog':120, 'cub':200}

def train_model(args, dataset_dirs, save_dir, batch_size, lr):
    print('count:',torch.cuda.device_count())
    print('id 0:',torch.cuda.get_device_name(0))
    # print('id 1:',torch.cuda.get_device_name(1))
    # print('id 2:',torch.cuda.get_device_name(2))
    print('cur id :',torch.cuda.current_device())
    make_dir_recursive(save_dir)

    #prepare data
    train_loader = get_dataloader_multi_set(dataset_dirs, 'train',
                                            batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader_multi_set(dataset_dirs, 'test',
                                           batch_size=batch_size, shuffle=False)

    # prpare model 
    model = resnet18(pretrained=True, num_classes=args.num_mainclass + args.num_auxclass)

    # ---------------------- Loss ----------------------
    criterion = nn.CrossEntropyLoss()

    if args.use_cuda:
        criterion = criterion.cuda()
        model = model.cuda()
    # ---------------------- Optimizer ----------------------
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[10, 20, 30], gamma=0.5)

    # ---------------------- Training ----------------------
    # log_file = open(save_dir+'/recode.log', mode = 'w',encoding='utf-8')
    
   
    for epoch in range(args.num_epoch):
        acc_msg = " " * 4
        acc_msg += "Epoch Accuracy: {}"
        # ------------ Test ------------
        if epoch == 0:
            mean_acc = test(test_loader, model, args.use_cuda)
            print(acc_msg.format(mean_acc))

        scheduler.step()
        # ------------ Train ------------
        print("Training... Epoch = {}".format(epoch + 1))
        print("    Learning Rate: {}".format(scheduler.get_lr()[0]))
        loss_value = train(train_loader, model, [criterion], [optimizer], args.use_cuda)

        loss_msg = " " * 4
        loss_msg += "Total Loss: {}, "
        print(loss_msg.format(loss_value))

        # ------------ Test ------------
        mean_acc = test(test_loader, model, args.use_cuda)
        print(acc_msg.format(mean_acc))

        # --------- Save the model --------
        save_name = os.path.join(save_dir, 'resnet-18-{:0>3}.pkl'.format(epoch + 1))
        torch.save(model.state_dict(), save_name)



def main_train():
    parser = argparse.ArgumentParser(
        description='Resnet knowledge training with multiTeacher')
    parser.add_argument('--main_class', default='airplane', type=str,
                        help='init main_class')
    parser.add_argument('--aux_class', default='car', type=str,
                        help='init aux_class')
    parser.add_argument('--main_part', default='1', type=str,
                        help='init main_part')
    parser.add_argument('--aux_part', default='2', type=str,
                        help='init aux_part')
    parser.add_argument('--num_mainclass', default=50, type=int,
                        help='init num_mainclass')
    parser.add_argument('--num_auxclass', default=49, type=int,
                        help='init num_auxclass')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_epoch', default=50, type=str,
                        help='Number of epoch')
    parser.add_argument('--use_cuda', default=True, type=bool,
                        help='initial cuda use')
    parser.add_argument('--data_root', default='/nfs/yxy/data/', type=str,
                        help='initial data root')
    parser.add_argument('--save_root', default='./snapshot/teachers/', type=str,
                        help='initial save root')
    parser.add_argument('--log_dir', default='/home/yxy/kacode2/Customize-Visual/run/', type=str,
                        help='initial log dir')
    args = parser.parse_args()

    # if you don't set parameter in terminal, you can set here
    args.main_class = 'airplane'     #'airplane' or 'car' or 'dog' or 'cub'
    args.aux_class = 'car'           #'airplane' or 'car' or 'dog' or 'cub'
    args.main_part = '2'             #'1' or '2'
    args.aux_part = '4'              #'1' or '2' or '3' or '4'
    args.num_mainclass = 50          #100/2 or 196/2 or 120/2 or 200/2
    args.num_auxclass = 49           #100/4 or 196/4 or 120/4 or 200/4
    args.batch_size = 64
    
    args.num_epoch = 300
    args.data_root = '/nfs/yxy/data/'
    args.save_root = './snapshot/sources/'

    dataset_dirs = [
        args.data_root+'{}/image_part{}'.format(args.main_class,args.main_part),
        args.data_root+'{}/image_part{}'.format(args.aux_class,args.aux_part),
    ]
    save_dir =args.save_root+ '{}_part{}-{}_part{}'.format(args.main_class, args.main_part, args.aux_class,args.aux_part)

    main_part_no = 1
    aux_part_no = 1
    
    # ------------------------------------------------
    lr = 0.001
    train_model(args, dataset_dirs, save_dir, args.batch_size, lr)


def test_model(test_loader, model, pre_cls, use_cuda):
    if use_cuda:
        model.cuda('1')
    model.eval()

    accuracies = []

    for i, (data, labels) in enumerate(test_loader):
        target_labels = labels

        if use_cuda:
            data = data.cuda()
        scores = model(data)

        cur_batch_size = data.size(0)

        tmp_acc = cal_accuracy(scores.detach().cpu().numpy()[:, :pre_cls],
                               target_labels.cpu().numpy().astype(np.int))

        accuracies.append(tmp_acc*cur_batch_size)

    norm_coeff = 1.0 / len(test_loader.dataset)

    mean_acc = np.array(accuracies).sum()*norm_coeff # accuracies per epoch

    return mean_acc


def main_test():

    main_part_no = 1
    aux_part_no = 1

    dataset = '/nfs/yxy/data/airplane_part{}-car_part{}'.format(main_part_no, aux_part_no)
    dataset_name = 'airplane'
    num_classes = 99
    pre_cls = 50

    # ---------------------- Model ----------------------
    model = resnet18(pretrained=False, num_classes=num_classes)
    model_path = './snapshot/teachers/{}/resnet-18-040.pkl'.format(dataset)
    model.load_state_dict(torch.load(model_path))

    # ---------------------- Dataset ----------------------

    batch_size = 64
    root_path = '/nfs/yxy/data/{}/'.format(dataset_name)
    test_loader = get_dataloader(root_path , 'test', batch_size=batch_size,
                                 is_part=True, part_num=main_part_no, shuffle=False)

    acc = test_model(test_loader, model, pre_cls=pre_cls, use_cuda=True)

    print('Accuracy: ', acc)


if __name__ == '__main__':
    main_train()
    # main_test()
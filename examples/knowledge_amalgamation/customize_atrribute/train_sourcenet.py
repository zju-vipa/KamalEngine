# coding: utf-8

import numpy as np
import os,sys
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.optim import lr_scheduler
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
from kamal import vision, engine, utils, amalgamation, metrics, callbacks
from kamal.vision import sync_transforms as sT
from torch import nn, optim
import torch, time
from torch.utils.tensorboard import SummaryWriter
from kamal.vision.datasets.CelebA import *
from kamal.vision.models.classification.resnet_customize import *
from torch.autograd import Variable
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

target_attributes_dir = {'Black_Hair':{'1':['Black_Hair','Arched_Eyebrows','Attractive','Big_Lips'],
                                       '2':['Black_Hair','Bags_Under_Eyes','Blurry']},
                        'Blond_Hair': {'3':['Blond_Hair','Pale_Skin','Receding_Hairline'],
                                       '4':['Blond_Hair','High_Cheekbones']},
                        'Brown_Hair': {'5':['Brown_Hair','Double_Chin'],
                                       '6':['Brown_Hair','Oval_Face','Bushy_Eyebrows'],
                                       '7':['Brown_Hair','Narrow_Eyes','Chubby']},
                        'Bangs':      {'8':['Bangs','Atrrractive','Receding_Hairline','Big_Nose'],
                                       '9':['Bangs','Blurry','Pale_Skin']}}
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def train(train_loader, target_idxs,
          model, criterions, optimizers, use_cuda,
          iteration):
    if use_cuda: model.cuda()
    model.train()

    losses = []
    for i, (data, labels) in enumerate(train_loader):
        # ------ Get data and label ------
        target_labels = labels[:, target_idxs]
        if use_cuda:
            data = data.cuda()
            target_labels = target_labels.cuda()
        data = Variable(data)
        target_labels = Variable(target_labels)
        cur_batch_size = data.size(0)
        # ------ Forward ------
        scores = model(data)
        # ------ Calculate loss for each task ------
        tensor_losses = []
        for j in range(len(target_idxs)):
            tensor_losses.append(
                criterions[0](scores[j], target_labels[:, j])
            )
        # ------ Calculate total loss ------
        total_loss = None
        for l in tensor_losses:
            if total_loss is None:
                total_loss = l.clone()
            else:
                total_loss += l
        part_loss_value = [l.item() for l in tensor_losses]
        # ------ Optimization ------
        optimizers[0].zero_grad()
        total_loss.backward()
        optimizers[0].step()

        part_loss_value = [item*cur_batch_size for item in part_loss_value]
        losses.append(part_loss_value)
        iteration += 1
    norm_coeff = 1.0 / len(train_loader.dataset)

    # mean loss in a epoch for each task
    mean_loss_values = norm_coeff*np.array(losses).sum(axis=0)
    total_loss_value = mean_loss_values.sum()

    return total_loss_value, mean_loss_values.tolist(), iteration


def cal_accuracy(scores, labels):
    assert (scores.shape[0] == labels.shape[0])
    preds = np.argmax(scores, axis=1)
    accuracy = float(sum(preds == labels)) / len(labels)
    
    return accuracy


def test(test_loader, target_idxs, model, use_cuda):
    if use_cuda: model.cuda()
    model.eval()
    accuracies = []

    for i, (data, labels) in enumerate(test_loader):
        target_labels = labels[:, target_idxs]
        if use_cuda:
            data = data.cuda()
        data = Variable(data)
        scores = model(data)
        cur_batch_size = data.size(0)
        task_accs = []
        for j in range(len(target_idxs)):
            tmp_acc = cal_accuracy(scores[j].data.cpu().numpy(),
                                   target_labels[:, j].cpu().numpy().astype(np.int64))
            task_accs.append(tmp_acc*cur_batch_size) # cur accuracy multiple cur_batch_size
        accuracies.append(task_accs)
    norm_coeff = 1.0 / len(test_loader.dataset)
    mean_accs = np.array(accuracies).sum(axis=0)*norm_coeff # accuracies on each task
    total_acc = mean_accs.mean() # mean accuracy on all tasks

    return total_acc, mean_accs.tolist()


def get_attri_idx(target_attribute):
    target_idx = attr_names.index(target_attribute)
    return target_idx

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        print('path: {} not exists'.format(dir_path))
        os.mkdir(dir_path)
        print('make directroy: ', dir_path)
    else:
        print('path: {} exists'.format(dir_path))

def parse_args():

    parser = argparse.ArgumentParser(
        description='Resnet 18 part training with single attribute')
    parser.add_argument('--target_attribute', default='Black_Hair', type=str,
                        help='target_attribute')
    parser.add_argument('--source_id', default='1', type=str,
                        help='source_id')
    parser.add_argument('--part_num', default=1, type=int,
                        help='part_num')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_epoch', default=50, type=int,
                        help='Batch size for training')
    parser.add_argument('--use_cuda', default=True, type=str2bool,
                        help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        help='initial learning rate')
    parser.add_argument('--data_root', default='/nfs/yxy/data/', type=str,
                        help='initial data root')
    parser.add_argument('--save_root', default='./snapshot/teachers/', type=str,
                        help='initial save root')
    return parser

def main():
    parser = parse_args()
    args = parser.parse_args()

    #if you don't set parameters in terminal, you can set here
    args.target_attribute = 'Bangs'
    args.source_id = '1'
    args.lr = 1e-4
    args.num_epoch = 50

    source_attributes = target_attributes_dir[args.target_attribute][args.source_id]
    source_channel = [64, 64, 128, 256, 512]
    
    random_seed = 3256
    torch.manual_seed(random_seed)

    if args.use_cuda:
        use_cuda = True
        torch.cuda.manual_seed(random_seed)
    else:
        use_cuda = False

    # ---------------------- Dataset ----------------------
    save_dir = args.save_root + args.target_attribute
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    target_idxs = [get_attri_idx(item) for item in source_attributes]
    save_idxs = '_'
    for i in range(len(target_idxs)):
        save_idxs += str(target_idxs[i]) + '_'
    print('save_idxs: {}'.format(save_idxs))

    train_loader = get_dataloader('train', batch_size=args.batch_size, is_part=False, shuffle=True)
    test_loader = get_dataloader('test', batch_size=args.batch_size, shuffle=False)

    # ---------------------- Model ----------------------
    print('source_attributes: {}'.format(source_attributes))
    model = source_resnet18(channel_nums =source_channel, pretrained=True, num_classes=2, target_attributes=source_attributes)

    # ---------------------- Loss ----------------------
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        criterion = criterion.cuda()
        model = model.cuda()
    # ---------------------- Optimizer ----------------------
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[5, 10, 15, 20, 25], gamma=0.5)

    # ---------------------- Training ---------------------
    iteration = 0
    for epoch in range(args.num_epoch):
        # ------------ Test ------------
        if epoch == 0:
            mean_acc, task_accs = test(test_loader, target_idxs, model, use_cuda)

            acc_msg = " " * 4
            acc_msg += "Mean Accuracy: {}, "
            for item in source_attributes:
                acc_msg += item + " Accuracy: {}, "
            acc_msg = acc_msg[:-2]
            print(acc_msg.format(mean_acc, *task_accs))

        model.train()

        scheduler.step()
        # ------------ Train ------------
        print("Training... Epoch = {}".format(epoch + 1))
        print("    Learning Rate: {}".format(scheduler.get_lr()[0]))
        total_loss, part_losses, iteration = train(train_loader, target_idxs,
                                                   model, [criterion], [optimizer], use_cuda,
                                                   iteration)

        loss_msg = " "*4
        loss_msg += "Total Loss: {}, "
        for item in source_attributes:
            loss_msg += item + " Loss: {}, "
        loss_msg = loss_msg[:-2]
        print(loss_msg.format(total_loss, *part_losses))

        # ------------ Test ------------
        mean_acc, task_accs = test(test_loader, target_idxs, model, use_cuda)

        acc_msg = " " * 4
        acc_msg += "Mean Accuracy: {}, "
        for item in source_attributes:
            acc_msg += item + " Accuracy: {}, "
        acc_msg = acc_msg[:-2]
        print(acc_msg.format(mean_acc, *task_accs))

        # --------- Save the model --------
        torch.save(model.state_dict(),
                   save_dir + '/' + 'resnet-teachers' + save_idxs + '-{:0>3}.pkl'.format(epoch + 1))


if __name__ == '__main__':
    main()


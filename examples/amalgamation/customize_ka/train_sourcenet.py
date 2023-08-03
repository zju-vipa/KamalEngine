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
from kamal.vision.datasets.CelebA import *
from kamal.vision.datasets.customize_class_data import *
from examples.amalgamation.customize_ka.customize_utils import make_dir_recursive

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

num_class_dic = {'airplane':100, 'car':196, 'dog':120, 'cub':200}

target_attributes_dir = {'Black_Hair':{'1':['Black_Hair','Arched_Eyebrows','Attractive','Big_Lips'],
                                       '2':['Black_Hair','Bags_Under_Eyes','Blurry']},
                        'Blond_Hair': {'3':['Blond_Hair','Pale_Skin','Receding_Hairline'],
                                       '4':['Blond_Hair','High_Cheekbones']},
                        'Brown_Hair': {'5':['Brown_Hair','Double_Chin'],
                                       '6':['Brown_Hair','Oval_Face','Bushy_Eyebrows'],
                                       '7':['Brown_Hair','Narrow_Eyes','Chubby']},
                        'Bangs':      {'8':['Bangs','Attractive','Big_Nose','Receding_Hairline'],
                                       '9':['Bangs','Blurry','Pale_Skin']}}

def get_attri_idx(tar_attribute):
    target_idx = attr_names.index(tar_attribute)
    return target_idx

def attribute_train(train_loader, target_idxs,
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

def attribute_cal_accuracy(scores, labels):
    assert (scores.shape[0] == labels.shape[0])
    preds = np.argmax(scores, axis=1)
    accuracy = float(sum(preds == labels)) / len(labels)
    
    return accuracy

def attribute_test(test_loader, target_idxs, model, use_cuda):
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
            tmp_acc = attribute_cal_accuracy(scores[j].data.cpu().numpy(),
                                   target_labels[:, j].cpu().numpy().astype(np.int64))
            task_accs.append(tmp_acc*cur_batch_size) # cur accuracy multiple cur_batch_size
        accuracies.append(task_accs)
    norm_coeff = 1.0 / len(test_loader.dataset)
    mean_accs = np.array(accuracies).sum(axis=0)*norm_coeff # accuracies on each task
    total_acc = mean_accs.mean() # mean accuracy on all tasks

    return total_acc, mean_accs.tolist()

def train_sourcenet_attribute():
    #if you don't set parameters in terminal, you can set here
    target_attribute = 'Bangs'
    source_id = '8'
    data_root = '/nfs/yxy/data/CelebA'
    save_root = './attribute/snapshot/teachers/'
    batch_size= 64
    use_cuda= True
    lr = 1e-4
    num_epoch = 50

    source_attributes = target_attributes_dir[target_attribute][source_id]
    source_channel = [64, 64, 128, 256, 512]
    
    random_seed = 3256
    torch.manual_seed(random_seed)

    if use_cuda:
        use_cuda = True
        torch.cuda.manual_seed(random_seed)
    else:
        use_cuda = False

    # ---------------------- Dataset ----------------------
    save_dir = save_root + target_attribute
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    target_idxs = [get_attri_idx(item) for item in source_attributes]
    save_idxs = '_'
    for i in range(len(target_idxs)):
        save_idxs += str(target_idxs[i]) + '_'
    print('save_idxs: {}'.format(save_idxs))

    train_loader = get_dataloader_attribute(data_root,'train', batch_size=batch_size, is_part=False, shuffle=True)
    test_loader = get_dataloader_attribute(data_root,'test', batch_size=batch_size, shuffle=False)

    # ---------------------- Model ----------------------
    print('source_attributes: {}'.format(source_attributes))
    model = source_resnet18(channel_nums =source_channel, pretrained=True, num_classes=2, target_attributes=source_attributes)

    # ---------------------- Loss ----------------------
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        criterion = criterion.cuda()
        model = model.cuda()
    # ---------------------- Optimizer ----------------------
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[5, 10, 15, 20, 25], gamma=0.5)

    # ---------------------- Training ---------------------
    iteration = 0
    for epoch in range(num_epoch):
        # ------------ Test ------------
        if epoch == 0:
            mean_acc, task_accs = attribute_test(test_loader, target_idxs, model, use_cuda)

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
        total_loss, part_losses, iteration = attribute_train(train_loader, target_idxs,
                                                   model, [criterion], [optimizer], use_cuda,
                                                   iteration)

        loss_msg = " "*4
        loss_msg += "Total Loss: {}, "
        for item in source_attributes:
            loss_msg += item + " Loss: {}, "
        loss_msg = loss_msg[:-2]
        print(loss_msg.format(total_loss, *part_losses))

        # ------------ Test ------------
        mean_acc, task_accs = attribute_test(test_loader, target_idxs, model, use_cuda)

        acc_msg = " " * 4
        acc_msg += "Mean Accuracy: {}, "
        for item in source_attributes:
            acc_msg += item + " Accuracy: {}, "
        acc_msg = acc_msg[:-2]
        print(acc_msg.format(mean_acc, *task_accs))

        # --------- Save the model --------
        torch.save(model.state_dict(),
                   save_dir + '/' + 'resnet-teachers' + save_idxs + '-{:0>3}.pkl'.format(epoch + 1))
    return 0

def class_train(train_loader, model, criterions, optimizers, use_cuda):
    if use_cuda:
        model.cuda()
    model.train()

    losses = []
    for i, (data, labels) in enumerate(train_loader):
        # ------ Get data and label ------
        target_labels = labels

        if use_cuda:
            data = data.cuda()
            target_labels = target_labels.cuda()

        # data = Variable(data)
        # target_labels = Variable(target_labels)

        cur_batch_size = data.size(0)
        # ------ Forward ------
        scores = model(data)

        # ------ Calculate loss ------
        loss = criterions[0](scores, target_labels)

        # ------ Optimization ------
        optimizers[0].zero_grad()
        loss.backward()
        optimizers[0].step()


        loss_value = cur_batch_size * loss.item()
        losses.append(loss_value)

    norm_coeff = 1.0 / len(train_loader.dataset)

    # mean loss in a epoch for each task
    mean_loss_value = norm_coeff*np.array(losses).sum(axis=0)

    return mean_loss_value

def class_cal_accuracy(scores, labels):
    # assert (scores.shape[0] == labels.shape[0])

    preds = np.argmax(scores, axis=1)

    accuracy = float(sum(preds == labels)) / len(labels)
    return accuracy

def class_test(test_loader, model, use_cuda):
    if use_cuda:
        model.cuda()
    model.eval()

    accuracies = []

    for i, (data, labels) in enumerate(test_loader):
        target_labels = labels

        if use_cuda:
            data = data.cuda()
        scores = model(data)

        cur_batch_size = data.size(0)

        tmp_acc = class_cal_accuracy(scores.detach().cpu().numpy(),
                               target_labels.cpu().numpy().astype(np.int64))

        accuracies.append(tmp_acc*cur_batch_size)

    norm_coeff = 1.0 / len(test_loader.dataset)

    mean_acc = np.array(accuracies).sum()*norm_coeff # accuracies per epoch

    return mean_acc

def train_sourcenet_class():
    # if you don't set parameter in terminal, you can set here
    main_class = 'airplane'     #'airplane' or 'car' or 'dog' or 'cub'
    aux_class = 'car'           #'airplane' or 'car' or 'dog' or 'cub'
    main_part = '1'             #'1' or '2'
    aux_part = '2'              #'1' or '2' or '3' or '4'
    num_mainclass = 50          #100/2 or 196/2 or 120/2 or 200/2
    num_auxclass = 49           #100/4 or 196/4 or 120/4 or 200/4
    batch_size = 64
    
    num_epoch = 300
    data_root = '/nfs/yxy/data/'
    save_root = './class/snapshot/sources/'
    use_cuda = True

    dataset_dirs = [
        data_root+'{}/image_part{}'.format(main_class,main_part),
        data_root+'{}/image_part{}'.format(aux_class,aux_part),
    ]
    save_dir =save_root+ '{}_part{}-{}_part{}'.format(main_class, main_part, aux_class,aux_part)

    main_part_no = 1
    aux_part_no = 1

    lr = 0.001

    make_dir_recursive(save_dir)
    #prepare data
    train_loader = get_dataloader_multi_set_class(dataset_dirs, 'train',
                                            batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader_multi_set_class(dataset_dirs, 'test',
                                           batch_size=batch_size, shuffle=False)

    # prpare model 
    model = resnet18(pretrained=True, num_classes=num_mainclass + num_auxclass)

    # ---------------------- Loss ----------------------
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        criterion = criterion.cuda()
        model = model.cuda()
    # ---------------------- Optimizer ----------------------
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[10, 20, 30], gamma=0.5)

    # ---------------------- Training ----------------------
    # log_file = open(save_dir+'/recode.log', mode = 'w',encoding='utf-8')
   
    for epoch in range(num_epoch):
        acc_msg = " " * 4
        acc_msg += "Epoch Accuracy: {}"
        # # ------------ Test ------------
        # if epoch == 0:
        #     mean_acc = class_test(test_loader, model, use_cuda)
        #     print(acc_msg.format(mean_acc))

        scheduler.step()
        # ------------ Train ------------
        print("Training... Epoch = {}".format(epoch + 1))
        print("    Learning Rate: {}".format(scheduler.get_lr()[0]))
        loss_value = class_train(train_loader, model, [criterion], [optimizer], use_cuda)

        loss_msg = " " * 4
        loss_msg += "Total Loss: {}, "
        print(loss_msg.format(loss_value))

        # ------------ Test ------------
        mean_acc = class_test(test_loader, model, use_cuda)
        print(acc_msg.format(mean_acc))

        # --------- Save the model --------
        save_name = os.path.join(save_dir, 'resnet-18-{:0>3}.pkl'.format(epoch + 1))
        torch.save(model.state_dict(), save_name)

    return 0

def main():
    parser = argparse.ArgumentParser(
        description='Resnet knowledge training with multiTeacher')
    parser.add_argument('--type', default='class', type=str,
                        help="choose train process, 'attribute' or 'class'")
    args = parser.parse_args()
    if args.type == 'attribute':
        train_sourcenet_attribute()
    elif args.type == 'class':
        train_sourcenet_class()


if __name__ == '__main__':
    main()
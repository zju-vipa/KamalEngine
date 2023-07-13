import os,sys
from kamal import vision, engine, utils, amalgamation, metrics, callbacks
from kamal.vision import sync_transforms as sT
from torch import nn, optim
import torch, time
from torch.utils.tensorboard import SummaryWriter
from kamal.vision.models.classification.resnet_customize import *
from torch.autograd import Variable
import numpy as np
dataset_to_cls_num = {
    'dog': 120,
    'airplane': 100,
    'cub': 200,
    'car': 196
}
def cal_entropy(logits_batch):
    exp_logits = torch.exp(logits_batch)

    probs_batch = exp_logits / exp_logits.sum(dim=1, keepdim=True)

    entropy_batch = -(probs_batch * torch.log(probs_batch)).sum(dim=1, keepdim=True)

    return entropy_batch

#divide data by entropy of teachers
def divide_data_sourcenet(component_dataset,component_part,aux_dataset,aux_parts,train_loader, teachers, root, use_cuda=True):
    
    path = os.path.join(root,'{}_part{}-{}_part{}_{}'.format(component_dataset,component_part,aux_dataset,aux_parts[0],aux_parts[1]))
    if not os.path.exists(path):
        os.makedirs(path)

    fnames = []
    selected_teachers = []
    for data, target, name in train_loader:
        data = data.cuda()
        entropy_batch_teacher = []
        for t in teachers:
            t.cuda().eval()
            logits = t(data)
            entropy_batch_teacher.append(
                cal_entropy(logits)
            )
        entropy_batch_teacher = torch.cat(entropy_batch_teacher, dim=1)

        selected_t = torch.argmin(entropy_batch_teacher, dim=1).cpu().numpy()

        fnames += list(name)
        selected_teachers += selected_t.tolist()

    f1 = open(path+'/data-t{}.txt'.format(aux_parts[0]), 'w')
    f2 = open(path+'/data-t{}.txt'.format(aux_parts[1]), 'w')

    for name, t in zip(fnames, selected_teachers):
        tmp = name.split('/')
        name = tmp[-2] + '/' + tmp[-1]
        if t == 0:
            f1.write(name + '\n')
        elif t == 1:
            f2.write(name + '\n')
        else:
            print('Error:', t)

    return path


def get_sourcenet_models(component_dataset,component_part,pre_cls_num,aux_dataset,aux_parts,cur_source_saveEpoch,sourcenets_root,source_channel):
    dataset_names = [
        '{}_part{}-{}_part{}'.format(component_dataset, component_part,aux_dataset, aux_parts[0]),
        '{}_part{}-{}_part{}'.format(component_dataset, component_part,aux_dataset, aux_parts[1])
    ]

    teacher_weights = [
        sourcenets_root+'{}/resnet-18-{:0>3}.pkl'.format(dataset_names[dn],cur_source_saveEpoch[dn])
        for dn in range(len(dataset_names))
    ]
    num_teacher = len(teacher_weights)
    sourcenets = list()
    for i in range(num_teacher):
        params = torch.load(teacher_weights[i])
        params['fc_layer.weight'] = params['fc_layer.weight'][:pre_cls_num]
        params['fc_layer.bias'] = params['fc_layer.bias'][:pre_cls_num]
        teacher = resnet18(pretrained=False,
                           num_classes=pre_cls_num,channel_num = source_channel)
        teacher.load_state_dict(params)

        sourcenets.append(teacher.cuda())
    return sourcenets

def get_component_models(component_dataset,component_parts,aux_dataset,aux_parts,component_saveEpoch,component_root,component_channel):
    components = []
    for i in range(len(component_parts)):
        component_path = component_root + 'component-'+'{}_part{}'.format(component_dataset,component_parts[i]) + '/' + \
                            '{}_part{}-{}_part{}_{}'.format(component_dataset,component_parts[i],aux_dataset,aux_parts[2*i],aux_parts[2*i+1])+ '/' + \
                            '{}_part{}-{}_part{}_{}'.format(component_dataset,component_parts[i],aux_dataset,aux_parts[2*i],aux_parts[2*i+1])+'-{:0>3}.pkl'.format(component_saveEpoch[i])
        component = resnet18(channel_num=component_channel,pretrained=False, num_classes=dataset_to_cls_num[component_dataset]  // 2)
        component.load_state_dict(torch.load(component_path))
        components.append(component)
    return components

def get_s2c_distill_models(num_sources,component_input_channel,source_input_channel,common_channel):
    # ---- Transfer Bridge ----
    distill_teachers = [] 
    for i in range(num_sources):
        distill_teachers.append(distill_models(input_channel=source_input_channel, output_channel=common_channel))
  
    distill_students = []
    for i in range(num_sources):
        distill_students.append(distill_models(input_channel=component_input_channel, output_channel=common_channel,s=True))
    return distill_teachers,distill_students

def get_c2t_distill_models(num_components,target_input_channel,common_channel):
    # ---- Transfer Bridge ----
    distill_encoders = []
    for i in range(num_components):
        distill_encoders.append(distill_models(input_channel=target_input_channel,
                                               output_channel=common_channel, s=False,is_m=True, m_no=i))
    return distill_encoders
        

def get_optimizer(num_teachers,model,lr_target,distill_teachers,lr_teacher,distill_students,lr_student):
    opt1 = []
    opt1.append({'params': model.parameters(), 'lr': lr_target})
    for i in range(num_teachers):
        #teacher
        opt1.append({'params': distill_teachers[i].conv0.parameters(), 'lr': lr_teacher[0]})
        opt1.append({'params': distill_teachers[i].conv1.parameters(), 'lr': lr_teacher[1]})
        opt1.append({'params': distill_teachers[i].conv2.parameters(), 'lr': lr_teacher[2]})
        opt1.append({'params': distill_teachers[i].conv3.parameters(), 'lr': lr_teacher[3]})
        opt1.append({'params': distill_teachers[i].conv4.parameters(), 'lr': lr_teacher[4]})

        opt1.append({'params': distill_students[i].conv0.parameters(), 'lr': lr_student[0]})
        opt1.append({'params': distill_students[i].conv1.parameters(), 'lr': lr_student[1]})
        opt1.append({'params': distill_students[i].conv2.parameters(), 'lr': lr_student[2]})
        opt1.append({'params': distill_students[i].conv3.parameters(), 'lr': lr_student[3]})
        opt1.append({'params': distill_students[i].conv4.parameters(), 'lr': lr_student[4]})

    optimizer = optim.Adam(opt1)
    return optimizer

def get_target_optimizer(num_components,model,lr_target,distill_encoders,lr_component):
    opt1 = []
    opt1.append({'params': model.parameters(), 'lr': lr_target})
    for i in range(num_components):
        opt1.append({'params': distill_encoders[i].conv0.parameters(), 'lr': lr_component[0]})
        opt1.append({'params': distill_encoders[i].conv1.parameters(), 'lr': lr_component[1]})
        opt1.append({'params': distill_encoders[i].conv2.parameters(), 'lr': lr_component[2]})
        opt1.append({'params': distill_encoders[i].conv3.parameters(), 'lr': lr_component[3]})
        opt1.append({'params': distill_encoders[i].conv4.parameters(), 'lr': lr_component[4]})

    optimizer = optim.Adam(opt1)
    return optimizer
def make_dir(dir_path):
    if not os.path.exists(dir_path):
        print('path: {} not exists'.format(dir_path))
        os.mkdir(dir_path)
        print('make directroy: ', dir_path)
    else:
        print('path: {} exists'.format(dir_path))


def make_dir_recursive(path):
    ps = path.split('/')
    accumate_path = ''
    for p in ps:
        accumate_path += (p+'/')
        make_dir(accumate_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


# ---------------- Train and Test ----------------

def train(train_loader, model, criterions, optimizers, use_cuda):
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


def cal_accuracy(scores, labels):
    # assert (scores.shape[0] == labels.shape[0])

    preds = np.argmax(scores, axis=1)

    accuracy = float(sum(preds == labels)) / len(labels)
    return accuracy


def test(test_loader, model, use_cuda):
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

        tmp_acc = cal_accuracy(scores.detach().cpu().numpy(),
                               target_labels.cpu().numpy().astype(np.int64))

        accuracies.append(tmp_acc*cur_batch_size)

    norm_coeff = 1.0 / len(test_loader.dataset)

    mean_acc = np.array(accuracies).sum()*norm_coeff # accuracies per epoch

    return mean_acc


def test_multi_task(test_loaders, model, use_cuda):
    if use_cuda:
        model.cuda()
    model.eval()

    accuracies = []

    for i, items in enumerate(zip(*test_loaders)):
        acc_pair = []
        for j, item in enumerate(items):
            data = item[0]
            label = item[1]
            if use_cuda:
                data = data.cuda()
            scores = model(data)

            cur_batch_size = data.size(0)

            tmp_acc = cal_accuracy(scores[j].detach().cpu().numpy(),
                                   label.cpu().numpy().astype(np.int64))
            acc_pair.append(tmp_acc*cur_batch_size)

        accuracies.append(acc_pair)

    norm_coeff = [1.0/len(item.dataset) for item in test_loaders]

    mean_acc = np.array(accuracies).sum(axis=0)*np.array(norm_coeff) # accuracies per epoch

    return mean_acc.tolist()


layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
# layer_names = ['conv1', 'layer1', 'layer2', 'layer3']


def cal_wn_loss(model):
    model.cuda()
    weight_params = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_params.append(param)

    wn_loss_total = torch.tensor(0.).cuda()
    wn_loss_part = []
    for i in range(len(weight_params)):
        weight_norm = torch.pow(weight_params[i], 2).sum(dim=1)
        # wn_loss += torch.pow(weight_norm - torch.tensor(1.), 2).sum()
        loss = torch.pow(weight_norm - torch.tensor(1.), 2).mean()
        wn_loss_total += loss
        wn_loss_part.append(loss)

    return wn_loss_total, wn_loss_part




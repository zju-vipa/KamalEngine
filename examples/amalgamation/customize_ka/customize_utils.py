import os,sys
from kamal import vision, engine, utils, amalgamation, metrics, callbacks
from kamal.vision import sync_transforms as sT
from torch import nn, optim
import torch, time
from torch.utils.tensorboard import SummaryWriter
from kamal.vision.models.classification.resnet_customize import *
from kamal.vision.datasets.CelebA import *
from torch.autograd import Variable
import numpy as np

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

component2target_attributes_dic={'hair':['Black_Hair','Blond_Hair','Brown_Hair','Bangs'],
                                 'eye' :['Bags_Under_Eyes','Bushy_Eyebrows','Narrow_Eyes'],
                                 'face':['Oval_Face','Young','Heavy_Makeup']}
source2target_id_dic={'hair':['1,2','3,4','5,6,7','8,9'],
                      'eye' :['1,2','3,4,5','6,7'],
                      'face':['1,2','3,4','5,6']}

source2target_attributes_dic = {'hair':{'Black_Hair':{'1':['Black_Hair','Arched_Eyebrows','Attractive','Big_Lips'],
                                                      '2':['Black_Hair','Bags_Under_Eyes','Blurry']},
                                        'Blond_Hair':{'3':['Blond_Hair','Pale_Skin','Receding_Hairline'],
                                                      '4':['Blond_Hair','High_Cheekbones']},
                                        'Brown_Hair':{'5':['Brown_Hair','Double_Chin'],
                                                      '6':['Brown_Hair','Oval_Face','Bushy_Eyebrows'],
                                                      '7':['Brown_Hair','Narrow_Eyes','Chubby']},
                                        'Bangs':     {'8':['Bangs','Attractive','Receding_Hairline','Big_Nose'],
                                                      '9':['Bangs','Blurry','Pale_Skin']}},
                                'eye' :{'Bags_Under_Eyes':{'1':['Bags_Under_Eyes','Rosy_Cheeks','Oval_Face','Smiling'],
                                                           '2':['Bags_Under_Eyes','Heavy_Makeup']},
                                        'Bushy_Eyebrows': {'3':['Bushy_Eyebrows','Big_Lips','Pale_Skin'],
                                                           '4':['Bushy_Eyebrows','Young'],
                                                           '5':['Bushy_Eyebrows','Heavy_Makeup','Wearing_Necklace']},
                                        'Narrow_Eyes':    {'6':['Narrow_Eyes','Wearing_Lipstick'],
                                                           '7':['Narrow_Eyes','Mouth_Slightly_Open','Wearing_Necktie']}},
                                'face':{'Oval_Face':      {'1':['Oval_Face','Male'],
                                                           '2':['Oval_Face','Big_Lips','Pointy_Nose']},
                                        'Young':          {'3':['Young','Wearing_Lipstick'],
                                                           '4':['Young','Gray_Hair','Receding_Hariline','Goatee']},
                                        'Heavy_Makeup':   {'5':['Heavy_Makeup','Male'],
                                                           '6':['Heavy_Makeup','Side_Burns','Double_chin','Attractive']} }                     
                         }

source2component_attributes_dic = {'Black_Hair': {'1':['Black_Hair','Arched_Eyebrows','Attractive','Big_Lips'],
                                                  '2':['Black_Hair','Bags_Under_Eyes','Blurry']},
                                    'Blond_Hair':{'3':['Blond_Hair','Pale_Skin','Receding_Hairline'],
                                                  '4':['Blond_Hair','High_Cheekbones']},
                                    'Brown_Hair':{'5':['Brown_Hair','Double_Chin'],
                                                  '6':['Brown_Hair','Oval_Face','Bushy_Eyebrows'],
                                                  '7':['Brown_Hair','Narrow_Eyes','Chubby']},
                                    'Bangs':     {'8':['Bangs','Attractive','Receding_Hairline','Big_Nose'],
                                                  '9':['Bangs','Blurry','Pale_Skin']},
                                    'Bags_Under_Eyes':{'1':['Bags_Under_Eyes','Rosy_Cheeks','Oval_Face','Smiling'],
                                                       '2':['Bags_Under_Eyes','Heavy_Makeup']},
                                    'Bushy_Eyebrows': {'3':['Bushy_Eyebrows','Big_Lips','Pale_Skin'],
                                                       '4':['Bushy_Eyebrows','Young'],
                                                       '5':['Bushy_Eyebrows','Heavy_Makeup','Wearing_Necklace']},
                                    'Narrow_Eyes':    {'6':['Narrow_Eyes','Wearing_Lipstick'],
                                                       '7':['Narrow_Eyes','Mouth_Slightly_Open','Wearing_Necktie']},
                                    'Oval_Face':    {'1':['Oval_Face','Male'],
                                                     '2':['Oval_Face','Big_Lips','Pointy_Nose']},
                                    'Young':        {'3':['Young','Wearing_Lipstick'],
                                                     '4':['Young','Gray_Hair','Receding_Hariline','Goatee']},
                                    'Heavy_Makeup': {'5':['Heavy_Makeup','Male'],
                                                     '6':['Heavy_Makeup','Side_Burns','Double_chin','Attractive']}                      
                         }

def get_attri_idx(target_attribute):
    target_idx = attr_names.index(target_attribute)
    return target_idx

def get_sourcenet_models_attribute(comp_i,dic,attribute,ids,epochs,sourcenets_root,source_channel):
    #get attributes list of 'source 1,2' 
    source_ids = ids[comp_i].split(',')   #['0','1']
    source_epochs = epochs        #[40,40]
    num_source = len(source_ids)
    sourcenets = []
    target_sort_id =[]
    for i in range(num_source): 
        sourcenets_attributes = dic[attribute][source_ids[i]]                  #get source_attributes
        atrris2labels =[]                                                      #get source_attributes to labels
        for n, item in enumerate(sourcenets_attributes):
            atrris2labels.append(get_attri_idx(item))
            if item in attribute:
                target_sort_id.append(n)
        teacher_save_idx = '_'
        for j in range(len(atrris2labels)):
            teacher_save_idx += str(atrris2labels[j]) + '_'
        sourcenet_path = sourcenets_root + attribute +  '/resnet-teachers' + teacher_save_idx + '-{:0>3}.pkl'.format(source_epochs)
        sourcenet = source_resnet18(channel_nums =source_channel ,pretrained=False, num_classes=2, target_attributes=sourcenets_attributes)
        sourcenet.load_state_dict(torch.load(sourcenet_path))
        sourcenets.append(sourcenet)
    return sourcenets,target_sort_id

def get_component_models_attribute(attributes,component_ids,epochs,component_channel,component_root):
        components = []
        for i in range(len(attributes)):
            source_select = component_ids[i].split(',')
            select_str = get_select_str_attribute(source_select)
            component_path = component_root + 'component-' + attributes[i] + '/' + \
                            attributes[i] + '-select-' + select_str + '-large-p1' + '/' + \
                            attributes[i] + '-select-' + select_str + '-large-p1' + '-{:0>3}.pkl'.format(epochs)
            component = component_resnet18(channel_num=component_channel, num_classes=2)
            component.load_state_dict(torch.load(component_path))
            components.append(component)
        
        return components

def get_s2c_distill_models_attribute(num_sources,component_input_channel,source_input_channel,common_channel):
    distill_teachers = []
    for i in range(num_sources):
        distill_teachers.append(distill_models(input_channel=source_input_channel, output_channel=common_channel))
  
    distill_students = []
    for i in range(num_sources):
        distill_students.append(distill_models(input_channel=component_input_channel, output_channel=common_channel,s=True))
    return distill_teachers,distill_students
def get_c2t_distill_models_attribute(num_components,target_input_channel,common_channel):
    distill_encoders = []
    for i in range(num_components):
        distill_encoders.append(distill_models(input_channel=target_input_channel,
                                               output_channel=common_channel, is_m=True, m_no=i))
    return distill_encoders

def get_criterions_attribute(num_teachers,num_layer):
    criterions = []
    for i in range(num_teachers):
        for j in range(num_layer):
            criterions.append(nn.MSELoss(reduction='mean'))
    criterions.append(nn.MSELoss(reduction='mean'))
    return criterions
def get_optimizer_attribute(num_teachers,model,lr_target,distill_teachers,lr_teacher,distill_students,lr_student):
    opt1 = []
    opt1.append({'params': model.parameters(), 'lr': lr_target})
    for i in range(num_teachers):
        #teacher
        opt1.append({'params': distill_teachers[i].conv0.parameters(), 'lr': lr_teacher[0]})
        opt1.append({'params': distill_teachers[i].conv1.parameters(), 'lr': lr_teacher[1]})
        opt1.append({'params': distill_teachers[i].conv2.parameters(), 'lr': lr_teacher[2]})
        opt1.append({'params': distill_teachers[i].conv3.parameters(), 'lr': lr_teacher[3]})
        opt1.append({'params': distill_teachers[i].conv4.parameters(), 'lr': lr_teacher[4]})
        #student
        # opt1.append({'params': distill_students[i].parameters(), 'lr': lr_student[5]})

        opt1.append({'params': distill_students[i].conv0.parameters(), 'lr': lr_student[0]})
        opt1.append({'params': distill_students[i].conv1.parameters(), 'lr': lr_student[1]})
        opt1.append({'params': distill_students[i].conv2.parameters(), 'lr': lr_student[2]})
        opt1.append({'params': distill_students[i].conv3.parameters(), 'lr': lr_student[3]})
        opt1.append({'params': distill_students[i].conv4.parameters(), 'lr': lr_student[4]})
        # lr_student[5] = 1e-7
        opt1.append({'params': distill_students[i].m, 'lr': lr_student[5]})

    optimizer = optim.Adam(opt1)
    return optimizer
def get_target_optimizer_attribute(num_components,model,lr_target,distill_encoders,lr_component):
    opt1 = []
    opt1.append({'params': model.parameters(), 'lr': lr_target})
    for i in range(num_components):
        opt1.append({'params': distill_encoders[i].conv0.parameters(), 'lr': lr_component[0]})
        opt1.append({'params': distill_encoders[i].conv1.parameters(), 'lr': lr_component[1]})
        opt1.append({'params': distill_encoders[i].conv2.parameters(), 'lr': lr_component[2]})
        opt1.append({'params': distill_encoders[i].conv3.parameters(), 'lr': lr_component[3]})
        opt1.append({'params': distill_encoders[i].conv4.parameters(), 'lr': lr_component[4]})
        # lr_student[5] = 1e-7
        opt1.append({'params': distill_encoders[i].m, 'lr': lr_component[5]})

    optimizer = optim.Adam(opt1)
    return optimizer
def cal_entropy_attribute(target_all):
    entropy_all = []
    for i in range(len(target_all)):
        sm = torch.exp(target_all[i])/((torch.exp(target_all[i])).sum())
        entropy = -(sm * torch.log(sm)).sum()
        entropy = entropy.data.cpu().numpy()
        entropy_all.append(entropy)
    en_min = np.argmin(entropy_all)
    return en_min

#divide data by entropy of teachers
def divide_data_sourcenet_attribute(component_attribute, data_loader, target_no, teacher_select, teachers, root, use_cuda=True):
    target_idxs = [get_attri_idx(item) for item in [component_attribute]]
    teacher_select_str = ''
    for m in teacher_select:
        teacher_select_str = teacher_select_str+m
    path = os.path.join(root,component_attribute,teacher_select_str)
    if not os.path.exists(path):
        os.makedirs(path)

    for i, (data, labels, img_name) in enumerate(data_loader):
        if use_cuda:
            data = data.cuda()
        # print(type(data))
        data = Variable(data)
        target_labels = labels[:, target_idxs]

        batch_size = data.size(0)

        # get featuremaps from teacher and student using hook
        targets = []
        num_teacher = len(teachers)
        for t in range(num_teacher):
            with torch.no_grad():
                teachers[t].cuda().eval()
                targets_full = teachers[t](data)
                target = targets_full[target_no[t]]
            targets.append(target)
        
        select_t = []
        for b in range(batch_size):
            target_all = []
            for t in range(num_teacher):
                target_all.append(targets[t][b])
            select_t.append(cal_entropy_attribute(target_all))
        
        for name, select_no, label in zip(img_name, select_t, target_labels):
        # for name, select_no in zip(img_name, select_t):
            teacher_txt = os.path.join(path,'data-t{}.txt'.format(teacher_select[select_no]))
            f = open(teacher_txt, 'a')
            label = label.cpu().numpy().astype(np.int64)
            f.write(os.path.basename(name) + ' ' + str(label[0]) + '\n')
    
    return path,target_idxs

def get_select_str_attribute(select_list):
    select_str = ''
    for s in select_list:
        select_str = select_str+s
    return select_str




###################################################################################################
dataset_to_cls_num = {
    'dog': 120,
    'airplane': 100,
    'cub': 200,
    'car': 196
}
def cal_entropy_class(logits_batch):
    exp_logits = torch.exp(logits_batch)

    probs_batch = exp_logits / exp_logits.sum(dim=1, keepdim=True)

    entropy_batch = -(probs_batch * torch.log(probs_batch)).sum(dim=1, keepdim=True)

    return entropy_batch

#divide data by entropy of teachers
def divide_data_sourcenet_class(component_dataset,component_part,aux_dataset,aux_parts,train_loader, teachers, root, use_cuda=True):
    
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
                cal_entropy_class(logits)
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


def get_sourcenet_models_class(component_dataset,component_part,pre_cls_num,aux_dataset,aux_parts,cur_source_saveEpoch,sourcenets_root,source_channel):
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

def get_component_models_class(component_dataset,component_parts,aux_dataset,aux_parts,component_saveEpoch,component_root,component_channel):
    components = []
    for i in range(len(component_parts)):
        component_path = component_root + 'component-'+'{}_part{}'.format(component_dataset,component_parts[i]) + '/' + \
                            '{}_part{}-{}_part{}_{}'.format(component_dataset,component_parts[i],aux_dataset,aux_parts[2*i],aux_parts[2*i+1])+ '/' + \
                            '{}_part{}-{}_part{}_{}'.format(component_dataset,component_parts[i],aux_dataset,aux_parts[2*i],aux_parts[2*i+1])+'-{:0>3}.pkl'.format(component_saveEpoch[i])
        component = resnet18(channel_num=component_channel,pretrained=False, num_classes=dataset_to_cls_num[component_dataset]  // 2)
        component.load_state_dict(torch.load(component_path))
        components.append(component)
    return components

def get_s2c_distill_models_class(num_sources,component_input_channel,source_input_channel,common_channel):
    # ---- Transfer Bridge ----
    distill_teachers = [] 
    for i in range(num_sources):
        distill_teachers.append(distill_models(input_channel=source_input_channel, output_channel=common_channel))
  
    distill_students = []
    for i in range(num_sources):
        distill_students.append(distill_models(input_channel=component_input_channel, output_channel=common_channel,s=True))
    return distill_teachers,distill_students

def get_c2t_distill_models_class(num_components,target_input_channel,common_channel):
    # ---- Transfer Bridge ----
    distill_encoders = []
    for i in range(num_components):
        distill_encoders.append(distill_models(input_channel=target_input_channel,
                                               output_channel=common_channel, s=False,is_m=True, m_no=i))
    return distill_encoders
        

def get_optimizer_class(num_teachers,model,lr_target,distill_teachers,lr_teacher,distill_students,lr_student):
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

def get_target_optimizer_class(num_components,model,lr_target,distill_encoders,lr_component):
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

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")



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





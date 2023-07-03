import torch
import torch.nn as nn
import torch.nn.functional as F
from kamal.core.engine.events import DefaultEvents, Event
from kamal.core.engine.engine import Engine
from kamal.core.engine.hooks import FeatureHook
from kamal.core import tasks
from torch.autograd import Variable
from kamal.utils import set_mode
import typing
import time
import numpy as np
import os
from kamal.utils import move_to_device, set_mode

class CUSTOMIZE_COMPONENT_Amalgamator(Engine):

    def __init__(self, logger=None, tb_writer=None ):
        super(CUSTOMIZE_COMPONENT_Amalgamator, self).__init__( logger, tb_writer )

    def setup(self,args, component_net, source_nets, teacher_select,target_idxs,target_no,distill_teachers, distill_students, dataloader, test_loader,  layer_names,special_module_idxs,
                                    criterion, optimizer, device=None ):
        self.args = args
        self.student = component_net
        self.teachers = nn.ModuleList(source_nets)
        self.teacher_select = teacher_select
        self.target_idxs = target_idxs
        self.target_no = target_no
        self.dataloader = dataloader
        self.test_loader = test_loader
        self.distill_teachers = distill_teachers
        self.distill_students = distill_students
        self.special_module_idxs = special_module_idxs
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.teacher_select_str = self.get_select_str(self.teacher_select)

        self.num_teacher = len(self.teacher_select)
        self.num_module = len(layer_names)
        self.total_loss_epoch = [0.0] * (self.num_teacher + 1)
        self.part_loss_epoch = []
        self.wn_loss_epoch = []
        for t in range(self.num_teacher):
            self.part_loss_epoch.append([0.0] * (self.num_module + 1 + 1))
            self.wn_loss_epoch.append([0.0] * (self.num_module + 1))
        
        self.save_dirname = args.component_attribute + '-select-' + self.teacher_select_str + '-large-p1'
        self.componentnets_root = os.path.join(args.componentnets_root,'component-' + args.component_attribute,self.save_dirname)
        if not os.path.exists(self.componentnets_root):
            os.makedirs(self.componentnets_root)

    def run(self, max_iter, start_iter=0, epoch_length=None):
        self.state.iter = self._state.start_iter = start_iter
        self.state.max_iter = max_iter
        self.state.epoch_length = epoch_length if epoch_length else len(self.dataloader[0])
        print('epoch len:',self.state.epoch_length)
        self.state.dataloader = zip(*self.dataloader)
        self.state.dataloader_iter = iter(zip(*self.dataloader))
        self.state.step_fn = self.step_fn

        self.trigger_events(DefaultEvents.BEFORE_RUN)
        for self.state.iter in range( start_iter, max_iter ):
            if self.state.epoch_length!=None and \
                 self.state.iter%self.state.epoch_length==0: # Epoch Start
                    self.trigger_events(DefaultEvents.BEFORE_EPOCH)
                    self.loss_clear()
            self.trigger_events(DefaultEvents.BEFORE_STEP)
            self.state.batch = self._get_batch()
            step_output = self.step_fn(self.state.batch)
            if isinstance(step_output, dict):
                self.state.metrics.update(step_output)
            self.trigger_events(DefaultEvents.AFTER_STEP)        
            if self.state.epoch_length!=None and \
                 (self.state.iter+1)%self.state.epoch_length==0: # Epoch End
                    epoch = (self.state.iter+1)//self.state.epoch_length
                    self.get_loss()
                    print('Loss: {} \npart loss: {}\nwn loss: {}'.format(self.total_loss_epoch, self.part_loss_epoch,self.wn_loss_epoch))
                    self.total_accuracy, self.mean_acc = self.test(self.test_loader,self.target_idxs,self.student,self.device)
                    self.print_acc(self.total_accuracy)

                    torch.save(self.student.state_dict(), os.path.join(self.componentnets_root,self.save_dirname + '-{:0>3}.pkl'.format(epoch)))
                    for t in range(len(self.teacher_select)):
                        torch.save(self.distill_teachers[t].state_dict(), self.componentnets_root + '/' +
                                'distill_sources-{}'.format(t) + '-{:0>3}.pkl'.format(epoch))
                        torch.save(self.distill_students[t].state_dict(), self.componentnets_root + '/' +
                                'distill_components-{}'.format(t) + '-{:0>3}.pkl'.format(epoch))
                    self.trigger_events(DefaultEvents.AFTER_EPOCH)
        self.trigger_events(DefaultEvents.AFTER_RUN)
    def step_fn(self,batch):
        start_time = time.perf_counter()
        tensor_total_loss = torch.tensor(0.).cuda()
        total_loss_batch = [0.0] * (self.num_teacher + 1)
        part_loss_batch = []
        wn_loss_batch = []
        for t in range(self.num_teacher):
            part_loss_batch.append([0.0] * (self.num_module + 1 + 1))
            wn_loss_batch.append([0.0] * (self.num_module + 1))
        
        # batch = move_to_device(batch, self.device)
        for t in range(self.num_teacher):
            data=  move_to_device(batch[t][0], self.device)
            batch_size = len(data)
            data = Variable(data)
            
            # distiller teacher
            with torch.no_grad():
                global outputs_teacher
                outputs_teacher = []
                self.teachers[t]
                handles_t = self.add_resnet_hook_t(self.teachers[t])
                targets_full = self.teachers[t](data)
                target = targets_full[self.target_no[t]]
            
            distill_t = self.distill_teachers[t](outputs_teacher)
            distill_t.append(target)
            for handle in handles_t:
                handle.remove()
        
            # distiller student
            global outputs_student
            outputs_student = []
            handles_s = self.add_resnet_hook_s(self.student)
            scores = self.student(data)
            outputs_student.append(scores)
            
            for handle in handles_s:
                handle.remove()
            distill_s = self.distill_students[t](outputs_student)

            for special_module_idx in self.special_module_idxs:
                loss = self.criterion[special_module_idx](distill_s[special_module_idx],
                                                      distill_t[special_module_idx])

                part_loss_batch[t][special_module_idx] = loss.item()
                total_loss_batch[t] += loss.item()
                
                self.part_loss_epoch[t][special_module_idx] += loss.item() * batch_size
                self.total_loss_epoch[t] += loss.item() * batch_size

                tensor_total_loss += loss

            # calculate weight norm loss of distill_teachers
            wn_loss_total, wn_loss_part = self.cal_wn_loss(self.distill_teachers[t])
            # wn_loss.cuda()
            part_loss_batch[t][6] = wn_loss_total.item()
            total_loss_batch[t] += wn_loss_total.item()
            self.part_loss_epoch[t][6] += wn_loss_total.item() * batch_size
            self.total_loss_epoch[t] += wn_loss_total.item() * batch_size
            # wn_loss_batch[t] = [wn_loss_part[wn].item() for wn in range(num_module)]
            for wn in range(self.num_module):
                wn_loss_batch[t][wn] = wn_loss_part[wn].item()
                self.wn_loss_epoch[t][wn] += wn_loss_part[wn].item() * batch_size
            wn_loss_batch[t][-1] = wn_loss_total.item()
            self.wn_loss_epoch[t][-1] += wn_loss_total.item() * batch_size
            # print('wn loss{}: {}'.format(t, wn_loss_batch))

            tensor_total_loss += wn_loss_total
            total_loss_batch[-1] += total_loss_batch[t]

        m_data = []
        for num in range(len(self.distill_students)):
            m_data.append(self.distill_students[num].m.item())
        
        loss_dict = { 
                      'part_loss_batch': part_loss_batch,
                      'wn_loss_batch': wn_loss_batch,
                      'total_loss_batch':total_loss_batch,}
        self.optimizer.zero_grad()
        tensor_total_loss.backward(retain_graph=True)
        self.optimizer.step()

        step_time = time.perf_counter() - start_time
        metrics = { loss_name: loss_value for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'tensor_total_loss':tensor_total_loss.item(),
            'step_time': step_time,
            'lr': float( self.optimizer.param_groups[0]['lr']) 
        })
        return metrics
    def cal_wn_loss(self,model):
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
    def test(self,test_loader, target_idxs, model, device):
        model = move_to_device(model,device)
        model.eval()
        accuracies = []
        for i, (data, labels) in enumerate(test_loader):
            target_labels = labels[:, target_idxs]
            data = move_to_device(data,device)
            data = Variable(data)
            scores = model(data)
            cur_batch_size = data.size(0)
            task_accs = []
            for j in range(len(target_idxs)):
                tmp_acc = self.cal_accuracy(scores.data.cpu().numpy(),
                                    target_labels[:, j].cpu().numpy().astype(np.int64))
                task_accs.append(tmp_acc*cur_batch_size) # cur accuracy multiple cur_batch_size
            accuracies.append(task_accs)
        norm_coeff = 1.0 / len(test_loader.dataset)
        mean_accs = np.array(accuracies).sum(axis=0)*norm_coeff # accuracies on each task

        total_acc = mean_accs.mean() # mean accuracy on all tasks

        return total_acc, mean_accs.tolist()   
    def obtain_features_t(self,module, input, output):
        outputs_teacher.append(output)
    
    def obtain_features_s(self,module, input, output):
        outputs_student.append(output)
        
    def add_resnet_hook_t(self,model):
        handles = []

        handles.append(model.conv1.register_forward_hook(self.obtain_features_t))
        handles.append(model.layer1.register_forward_hook(self.obtain_features_t))
        handles.append(model.layer2.register_forward_hook(self.obtain_features_t))
        handles.append(model.layer3.register_forward_hook(self.obtain_features_t))
        handles.append(model.layer4.register_forward_hook(self.obtain_features_t))

        return handles
    def add_resnet_hook_s(self,model):
        handles = []

        handles.append(model.conv1.register_forward_hook(self.obtain_features_s))
        handles.append(model.layer1.register_forward_hook(self.obtain_features_s))
        handles.append(model.layer2.register_forward_hook(self.obtain_features_s))
        handles.append(model.layer3.register_forward_hook(self.obtain_features_s))
        handles.append(model.layer4.register_forward_hook(self.obtain_features_s))

        return handles
    
    def get_select_str(self,select_list):
        select_str = ''
        for s in select_list:
            select_str = select_str+s
        return select_str
    def cal_accuracy(self,scores, labels):
        assert (scores.shape[0] == labels.shape[0])

        preds = np.argmax(scores, axis=1)

        accuracy = float(sum(preds == labels)) / len(labels)
        return accuracy
    
    def get_loss(self):
        data_num = 0
        for t in range(len(self.dataloader)):
            data_num += len(self.dataloader[t].dataset)
        # norm_coeff = 1.0 / len(data_loader.dataset)
        norm_coeff = 1.0 / data_num
        # norm_coeff = 1.0 / len(train_loader.dataset)

        for t in range(self.num_teacher):
            for i in range(self.num_module + 1 + 1):
                self.part_loss_epoch[t][i] *= norm_coeff
            for i in range(self.num_module + 1):
                self.wn_loss_epoch[t][i] *= norm_coeff
            self.total_loss_epoch[t] *= norm_coeff
            self.total_loss_epoch[-1] += self.total_loss_epoch[t]

        return self.total_loss_epoch, self.part_loss_epoch
    def print_acc(self,mean_acc):
        print('Accuracy: {}'.format(mean_acc))


    def loss_clear(self):
        self.total_loss_epoch = [0.0] * (self.num_teacher + 1)
        self.part_loss_epoch = [] 
        for t in range(self.num_teacher):
            self.part_loss_epoch.append([0.0] * (self.num_module + 1 + 1))



class CUSTOMIZE_TARGET_Amalgamator(Engine):

    def __init__(self, logger=None, tb_writer=None ):
        super(CUSTOMIZE_TARGET_Amalgamator, self).__init__( logger, tb_writer )

    def setup(self,args, target_net, component_nets, target_idxs,component_attributes,distill_target, dataloader, test_loader,  layer_names, special_module_idxs,
                                    criterion, optimizer, device=None ):
        self.args = args
        self.student = target_net
        self.teachers = nn.ModuleList(component_nets)
        self.component_attributes = component_attributes
        self.target_idxs = target_idxs
        self.dataloader = dataloader
        self.test_loader = test_loader
        self.distill_target = distill_target
        self.special_module_idxs = special_module_idxs
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.num_teacher = len(self.component_attributes)
        self.num_module =  len(layer_names)
        self.total_loss_epoch = [0.0] * (self.num_teacher + 1)
        self.part_loss_epoch = [] 
        self.wn_loss_epoch = []
        for t in range(self.num_teacher):
            self.part_loss_epoch.append([0.0] * (self.num_module + 1 + 1))
            self.wn_loss_epoch.append([0.0] * (self.num_module + 1))
        
        self.save_dirname = args.target_attribute + '_s2s_large_selection_90'
        self.target_root = os.path.join(args.target_root,self.save_dirname)
        if not os.path.exists(self.target_root):
            os.makedirs(self.target_root)

    def run(self, max_iter, start_iter=0, epoch_length=None):
        self.state.iter = self._state.start_iter = start_iter
        self.state.max_iter = max_iter
        self.state.epoch_length = epoch_length if epoch_length else len(self.dataloader)
        self.state.dataloader = self.dataloader
        self.state.dataloader_iter = iter(self.dataloader)
        self.state.step_fn = self.step_fn

        self.trigger_events(DefaultEvents.BEFORE_RUN)
        for self.state.iter in range( start_iter, max_iter ):
            if self.state.epoch_length!=None and \
                 self.state.iter%self.state.epoch_length==0: # Epoch Start
                    self.trigger_events(DefaultEvents.BEFORE_EPOCH)
                    self.loss_clear()
            self.trigger_events(DefaultEvents.BEFORE_STEP)
            self.state.batch = self._get_batch()
            step_output = self.step_fn(self.state.batch)
            if isinstance(step_output, dict):
                self.state.metrics.update(step_output)
            self.trigger_events(DefaultEvents.AFTER_STEP)        
            if self.state.epoch_length!=None and \
                 (self.state.iter+1)%self.state.epoch_length==0: # Epoch End
                    epoch = (self.state.iter+1)//self.state.epoch_length
                    self.get_loss()
                    print('Loss: {}\npart loss: {}'.format(self.total_loss_epoch, self.part_loss_epoch))
                    self.total_accuracy, self.mean_acc = self.test(self.test_loader,self.target_idxs,self.student,self.device)
                    self.print_acc(self.total_accuracy, self.mean_acc)
                        
                    torch.save(self.student.state_dict(), os.path.join(self.target_root, self.save_dirname+ '-{:0>3}.pkl'.format(epoch + 1)))
                    for t in range(self.num_teacher):
                        torch.save(self.distill_target[t].state_dict(), os.path.join(self.target_root, 'distill_targets-{}' + '-{:0>3}.pkl'.format(t, epoch + 1)))

                    self.trigger_events(DefaultEvents.AFTER_EPOCH)
        self.trigger_events(DefaultEvents.AFTER_RUN)


    def step_fn(self,batch):
        start_time = time.perf_counter()
        tensor_total_loss = torch.tensor(0.).cuda()
        total_loss_batch = [0.0] * (self.num_teacher + 1)
        part_loss_batch = []
        for t in range(self.num_teacher):
            part_loss_batch.append([0.0] * (self.num_module + 1 + 1))
        
        data = move_to_device(batch[0],self.device)
        for t in range(self.num_teacher):
            batch_size = len(data)
            data = Variable(data)
            
            # distiller teacher
            with torch.no_grad():
                global outputs_teacher
                outputs_teacher = []
                self.teachers[t].eval()
                handles_t = self.add_resnet_hook_t(self.teachers[t])
                targets_full = self.teachers[t](data)
                outputs_teacher.append(targets_full)
                for handle in handles_t:
                    handle.remove()
        
            # distiller student
            global outputs_student
            outputs_student = []
            handles_s = self.add_resnet_hook_s(self.student)
            scores = self.student(data)
            outputs_student.append(scores)
            for handle in handles_s:
                handle.remove()

            distill_tar = self.distill_target[t](outputs_student)

            for special_module_idx in self.special_module_idxs:
                loss = self.criterion[special_module_idx](distill_tar[special_module_idx],
                                                      outputs_teacher[special_module_idx])

                part_loss_batch[t][special_module_idx] = loss.item()
                total_loss_batch[t] += loss.item()
                
                self.part_loss_epoch[t][special_module_idx] += loss.item() * batch_size
                self.total_loss_epoch[t] += loss.item() * batch_size

                if tensor_total_loss is None:
                        tensor_total_loss = loss.clone()
                else:
                        tensor_total_loss += loss

            total_loss_batch[-1] += total_loss_batch[t]

        m_data = []
        for num in range(len(self.distill_target)):
            m_data.append(self.distill_target[num].m.item())
        
        loss_dict = { 
                      'part_loss_batch': part_loss_batch,
                      'total_loss_batch':total_loss_batch,}
        self.optimizer.zero_grad()
        tensor_total_loss.backward(retain_graph=True)
        self.optimizer.step()

        step_time = time.perf_counter() - start_time
        metrics = { loss_name: loss_value for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'tensor_total_loss':tensor_total_loss.item(),
            'step_time': step_time,
            'lr': float( self.optimizer.param_groups[0]['lr'] ) 
        })
        return metrics
    
    def cal_wn_loss(self,model):
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
    
    def test(self,test_loader, target_idxs, model, device):
        model = move_to_device(model,device)
        model.eval()
        accuracies = []
        for i, (data, labels) in enumerate(test_loader):
            target_labels = labels[:, target_idxs]
            data = move_to_device(data,device)
            data = Variable(data)
            scores = model(data)
            cur_batch_size = data.size(0)
            task_accs = []
            for j in range(len(target_idxs)):
                tmp_acc = self.cal_accuracy(scores[j].data.cpu().numpy(),
                                    target_labels[:, j].cpu().numpy().astype(np.int64))
                task_accs.append(tmp_acc*cur_batch_size) # cur accuracy multiple cur_batch_size
            accuracies.append(task_accs)
        norm_coeff = 1.0 / len(test_loader.dataset)
        mean_accs = np.array(accuracies).sum(axis=0)*norm_coeff # accuracies on each task
        total_acc = mean_accs.mean() # mean accuracy on all tasks
        return total_acc, mean_accs.tolist()

    
    def obtain_features_t(self,module, input, output):
        outputs_teacher.append(output)
    
    def obtain_features_s(self,module, input, output):
        outputs_student.append(output)

    def add_resnet_hook_t(self,model):
        handles = []

        handles.append(model.conv1.register_forward_hook(self.obtain_features_t))
        handles.append(model.layer1.register_forward_hook(self.obtain_features_t))
        handles.append(model.layer2.register_forward_hook(self.obtain_features_t))
        handles.append(model.layer3.register_forward_hook(self.obtain_features_t))
        handles.append(model.layer4.register_forward_hook(self.obtain_features_t))

        return handles
    def add_resnet_hook_s(self,model):
        handles = []

        handles.append(model.conv1.register_forward_hook(self.obtain_features_s))
        handles.append(model.layer1.register_forward_hook(self.obtain_features_s))
        handles.append(model.layer2.register_forward_hook(self.obtain_features_s))
        handles.append(model.layer3.register_forward_hook(self.obtain_features_s))
        handles.append(model.layer4.register_forward_hook(self.obtain_features_s))

        return handles
    def cal_accuracy(self,scores, labels):
        assert (scores.shape[0] == labels.shape[0])

        preds = np.argmax(scores, axis=1)

        accuracy = float(sum(preds == labels)) / len(labels)
        return accuracy
    def get_select_str(self,select_list):
        select_str = ''
        for s in select_list:
            select_str = select_str+s
        return select_str
    
    def get_loss(self):
        data_num = len(self.dataloader.dataset)
        # norm_coeff = 1.0 / len(data_loader.dataset)
        norm_coeff = 1.0 / data_num
        # norm_coeff = 1.0 / len(train_loader.dataset)

        for t in range(self.num_teacher):
            for i in range(self.num_module + 1):
                self.part_loss_epoch[t][i] *= norm_coeff
            self.total_loss_epoch[t] *= norm_coeff
            self.total_loss_epoch[-1] += self.total_loss_epoch[t]

        return self.total_loss_epoch, self.part_loss_epoch
    def print_acc(self,mean_acc, task_accs):
        acc_msg = " " * 4
        acc_msg += "Mean Accuracy: {}, "
        for item in self.component_attributes:
            acc_msg += item + " Accuracy: {}, "
        acc_msg = acc_msg[:-2]
        print(acc_msg.format(mean_acc, *task_accs))


    def loss_clear(self):
        self.total_loss_epoch = [0.0] * (self.num_teacher + 1)
        self.part_loss_epoch = [] 
        for t in range(self.num_teacher):
            self.part_loss_epoch.append([0.0] * (self.num_module + 1 + 1))

            
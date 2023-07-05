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
from  kamal.utils import logger

class CUSTOMIZE_COMPONENT_Amalgamator(Engine):

    def __init__(self, logger=None, tb_writer=None ):
        super(CUSTOMIZE_COMPONENT_Amalgamator, self).__init__( logger, tb_writer )

    def setup(self,args, component_net,component_part, source_nets, aux_parts, distill_students,distill_teachers,\
              dataloader, test_loader,layer_names,special_module_idxs, criterion, optimizer, device=None ):
        self.args = args
        self.student = component_net
        self.teachers = nn.ModuleList(source_nets)

        self.distill_students = distill_students
        self.distill_teachers = distill_teachers
        
        self.dataloader = dataloader
        self.test_loader = test_loader
        self.special_module_idxs = special_module_idxs
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.num_teacher = len(self.teachers)
        self.num_module = len(layer_names)

        self.total_loss_epoch = [0.0] * (self.num_teacher + 1)
        self.part_loss_epoch = []
        self.wn_loss_epoch = []
        for t in range(self.num_teacher):
            self.part_loss_epoch.append([0.0] * (self.num_module + 1 + 1))
            self.wn_loss_epoch.append([0.0] * (self.num_module + 1))
        
        
        self.save_dirname = '{}_part{}-{}_part{}_{}'.format(args.main_class,component_part,args.aux_class,aux_parts[0],aux_parts[1])
        self.componentnets_root = os.path.join(args.componentnets_root,'component-'+'{}_part{}'.format(args.main_class,component_part), self.save_dirname)
        if not os.path.exists(self.componentnets_root):
            os.makedirs(self.componentnets_root)
        self.logger_com = logger.Logger(self.componentnets_root+'/{}_{}.txt'.\
                format(self.args.main_class, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    def run(self, max_iter, start_iter=0, epoch_length=None):
        self.state.iter = self._state.start_iter = start_iter
        self.state.max_iter = max_iter
        self.state.epoch_length = epoch_length if epoch_length else len(self.dataloader[0])
        print('epoch len:',self.state.epoch_length)
        self.state.dataloader = self.dataloader
        self.state.dataloader_iter = [iter(dataloader) for dataloader in self.state.dataloader]
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
                    self.mean_acc = self.test(self.test_loader,self.student,self.device)
                    print('Accuracy: {}'.format(self.mean_acc))
                    if self.logger_com:
                            self.logger_com.write('Epoch: {}'.format(epoch), 'Accuracy: {}'.format(self.mean_acc))
                    
                    torch.save(self.student.state_dict(), os.path.join(self.componentnets_root,self.save_dirname + '-{:0>3}.pkl'.format(epoch)))
                    for t in range(self.num_teacher):
                        torch.save(self.distill_teachers[t].state_dict(), self.componentnets_root + '/' +
                                'distill_sources-{}'.format(t) + '-{:0>3}.pkl'.format(epoch))
                        torch.save(self.distill_students[t].state_dict(), self.componentnets_root + '/' +
                                'distill_components-{}'.format(t) + '-{:0>3}.pkl'.format(epoch))
                    self.trigger_events(DefaultEvents.AFTER_EPOCH)
        self.trigger_events(DefaultEvents.AFTER_RUN)
        if self.logger_com:
            self.logger_com.close()

    def _get_batch(self):
        try:
            # batch = next( self.state.dataloader_iter )
            batch =[next(it) for it in self.state.dataloader_iter]
            a = 0
        except StopIteration:
            self.state.dataloader_iter = [iter(dataloader) for dataloader in self.state.dataloader]# reset iterator
            print('reset')
            batch = [next(it) for it in self.state.dataloader_iter]
            a = 0

        if not isinstance(batch, (list, tuple)):
            batch = [ batch, ] # no targets
        return batch
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
        for t in range(2):
            data=  move_to_device(batch[t][0], self.device)
            batch_size = len(data)
            data = Variable(data)
            
           # get featuremaps from teacher and student using hook
            with torch.no_grad():
                global outputs_teacher
                outputs_teacher = []
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
            
            distill_t = self.distill_teachers[t](outputs_teacher)
            distill_s = self.distill_students[t](outputs_student)

            for special_module_idx in self.special_module_idxs:
                if special_module_idx < 5:
                    loss = self.criterion(distill_s[special_module_idx],
                                     distill_t[special_module_idx])

                    part_loss_batch[t][special_module_idx] = loss.item()
                    total_loss_batch[t] += loss.item()
                    self.part_loss_epoch[t][special_module_idx] += loss.item() * batch_size
                    self.total_loss_epoch[t] += loss.item() * batch_size

                    tensor_total_loss += loss
                elif special_module_idx == 5:
                    loss = self.criterion(outputs_student[special_module_idx],
                                     outputs_teacher[-1])

                    part_loss_batch[t][special_module_idx] = loss.item()
                    total_loss_batch[t] += loss.item()
                    self.part_loss_epoch[t][special_module_idx] += loss.item() * batch_size
                    self.total_loss_epoch[t] += loss.item() * batch_size

                    tensor_total_loss += loss

                # ---------------- Weight Regularization ----------------
                wn_loss_total, wn_loss_part = self.cal_wn_loss(self.distill_teachers[t])
                part_loss_batch[t][-1] = wn_loss_total.item()
                total_loss_batch[t] += wn_loss_total.item()
                self.part_loss_epoch[t][-1] += wn_loss_total.item() * batch_size
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
   
    
    def test(self,test_loader, model, device):
        model = move_to_device(model,device)
        model.eval()
        accuracies = []
        for i, (data, labels) in enumerate(test_loader):
            target_labels = labels
            data = move_to_device(data,device)
            data = Variable(data)
            scores = model(data)
            cur_batch_size = data.size(0)
           
            tmp_acc = self.cal_accuracy(scores.detach().cpu().numpy(),
                                target_labels.cpu().numpy().astype(np.int64))
            accuracies.append(tmp_acc*cur_batch_size)
        norm_coeff = 1.0 / len(test_loader.dataset)
        mean_acc = np.array(accuracies).sum()*norm_coeff # accuracies per epoch

        return mean_acc 
      
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
        self.wn_loss_eopch = []
        for t in range(self.num_teacher):
            self.part_loss_epoch.append([0.0] * (self.num_module + 1 + 1))
            self.wn_loss_eopch.append([0.0] * (self.num_module + 1))

    def get_select_str(self,select_list):
        select_str = ''
        for s in select_list:
            select_str = select_str+s
        return select_str

class CUSTOMIZE_TARGET_Amalgamator(Engine):

    def __init__(self, logger=None, tb_writer=None ):
        super(CUSTOMIZE_TARGET_Amalgamator, self).__init__( logger, tb_writer )

    def setup(self,args, target_net, component_nets, component_attributes,distill_target, dataloader, test_loaders,  layer_names, special_module_idxs,
                                    criterion, optimizer, device=None ):
        self.args = args
        self.student = target_net
        self.teachers = nn.ModuleList(component_nets)
        self.component_attributes = component_attributes
        self.dataloader = dataloader
        self.test_loaders = test_loaders
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
        
        self.save_dirname = args.main_class + '_s2s_large_selection_90'
        self.target_root = os.path.join(args.target_root,self.save_dirname)
        if not os.path.exists(self.target_root):
            os.makedirs(self.target_root)
        self.logger_tar = logger.Logger(self.target_root+'/{}_{}.txt'.\
                format(self.args.main_class, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

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
                    self.total_accuracy, self.mean_accs = self.test_multi_task(self.test_loaders,self.student,self.device)
                    self.print_acc(epoch,self.total_accuracy, self.mean_accs)
                    if self.logger_tar:
                        self.logger_tar.write('Epoch: {}'.format(epoch),' Mean Accuracy:{}, Task Acc {}, {}'.format(self.total_accuracy,self.mean_accs[0], self.mean_accs[1]))
                        
                    torch.save(self.student.state_dict(), os.path.join(self.target_root, self.save_dirname+ '-{:0>3}.pkl'.format(epoch + 1)))
                    for t in range(self.num_teacher):
                        torch.save(self.distill_target[t].state_dict(), os.path.join(self.target_root, 'distill_targets-{}' + '-{:0>3}.pkl'.format(t, epoch + 1)))

                    self.trigger_events(DefaultEvents.AFTER_EPOCH)
        self.trigger_events(DefaultEvents.AFTER_RUN)
        if self.logger_tar:
                self.logger_tar.close()

    def step_fn(self,batch):
        start_time = time.perf_counter()
        tensor_total_loss = torch.tensor(0.).cuda()
        total_loss_batch = [0.0] * (self.num_teacher + 1)
        part_loss_batch = []
        wn_loss_eopch = []
        for t in range(self.num_teacher):
            part_loss_batch.append([0.0] * (self.num_module + 1 + 1))
            wn_loss_eopch.append([0.0] * (self.num_module + 1))
        data = move_to_device(batch[0],self.device)
        teachers_features = []
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
                teachers_features.append(outputs_teacher)
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
                if special_module_idx < 5:
                    loss = self.criterion(distill_tar[special_module_idx],
                                     teachers_features[t][special_module_idx])

                    part_loss_batch[t][special_module_idx] = loss.item()
                    total_loss_batch[t] += loss.item()
                    self.part_loss_epoch[t][special_module_idx] += loss.item() * batch_size
                    self.total_loss_epoch[t] += loss.item() * batch_size

                    tensor_total_loss += loss
                elif special_module_idx == 5:
                    loss = self.criterion(distill_tar[special_module_idx],
                                     teachers_features[t][-1])

                    part_loss_batch[t][special_module_idx] = loss.item()
                    total_loss_batch[t] += loss.item()
                    self.part_loss_epoch[t][special_module_idx] += loss.item() * batch_size
                    self.total_loss_epoch[t] += loss.item() * batch_size

                    tensor_total_loss += loss

            total_loss_batch[-1] += total_loss_batch[t]
        
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
    
    def test_multi_task(self,test_loaders, model, device):
        
        model = move_to_device(model,device)
        model.eval()

        accuracies = []
        for i, items in enumerate(zip(*test_loaders)):
            acc_pair = []
            for j, item in enumerate(items):
                data = item[0]
                label = item[1]
               
                data = move_to_device(data,device)
                scores = model(data)

                cur_batch_size = data.size(0)

                tmp_acc = self.cal_accuracy(scores[j].detach().cpu().numpy(),
                                    label.cpu().numpy().astype(np.int64))
                acc_pair.append(tmp_acc*cur_batch_size)

            accuracies.append(acc_pair)

        norm_coeff = [1.0/len(item.dataset) for item in test_loaders]

        mean_accs = np.array(accuracies).sum(axis=0)*np.array(norm_coeff) # accuracies per epoch
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
    
    def print_acc(self,epoch,mean_acc, task_accs):
        acc_msg = " " * 4
        acc_msg += "Mean Accuracy: {}, "
        for item in self.component_attributes:
            acc_msg += 'part'+item + " Accuracy: {}, "
        acc_msg = acc_msg[:-2]
        print(acc_msg.format(mean_acc, *task_accs))
        

    def loss_clear(self):
        self.total_loss_epoch = [0.0] * (self.num_teacher + 1)
        self.part_loss_epoch = [] 
        for t in range(self.num_teacher):
            self.part_loss_epoch.append([0.0] * (self.num_module + 1 + 1))

            
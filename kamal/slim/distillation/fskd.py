from .kd import KDDistiller
from kamal.core.tasks.loss import KDLoss
from kamal.utils import set_mode, move_to_device
from kamal.core.engine.events import DefaultEvents, Event
import torch.nn as nn
import torch._ops
import torch.nn.functional as F
import time
from collections import OrderedDict
from  kamal.utils import logger
import torch.optim as optim

class FSKD_BLOCK_Distiller(KDDistiller):

    def __init__(self, logger=None, tb_writer=None ):
        super(FSKD_BLOCK_Distiller, self).__init__( logger, tb_writer )

    def setup(self,args, s_blocks, teacher, dataloader,test_loader,s_blocks_graft_ids, s_blocks_len,adaptions, optimizers, device=None ):
        self.blocks_s = s_blocks
        self.teacher = teacher
        self.dataloader = dataloader
        self.test_loader = test_loader
        self.args = args
        self.adaptions_t2s, self.adaptions_s2t = adaptions
        self.optimizers_s, self.optimizers_adapt_t2s, self.optimizers_adapt_s2t = optimizers
        self.device = device
        
        self.blocks_graft_ids = s_blocks_graft_ids
        self.blocks_len = s_blocks_len
        self.params_s_best = OrderedDict()
        self.logger_block = logger.Logger('/home/yxy/kacode/KamalEngine/examples/kd/log/graft_block_{}_{}_num_per_class_{}.txt'.\
                format(self.args.dataset, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()), 
                       self.args.num_per_class))


    def run(self, epochs_list, start_iter=0, epoch_length=None):
        for block in self.blocks_s:
            block.train()
        for block_id in range(len(self.blocks_s)):
            self.block_idx = block_id
            epochs_i = epochs_list[block_id]
            self.optimizer_s = self.optimizers_s[block_id]
            self.block_s = self.blocks_s[block_id]
            self.block_graft_id = self.blocks_graft_ids[block_id]
            self.block_len = self.blocks_len[block_id]
            if block_id > 0 and block_id < len(self.blocks_s) - 1:
                adaption_t2s = self.adaptions_t2s[block_id-1]
                adaption_s2t = self.adaptions_s2t[block_id]
                self.optimizer_adapt_t2s = self.optimizers_adapt_t2s[block_id-1]
                self.optimizer_adapt_s2t = self.optimizers_adapt_s2t[block_id]
                self.block_s = nn.Sequential(adaption_t2s, self.block_s, adaption_s2t)
            elif block_id == 0:
                adaption_s2t = self.adaptions_s2t[block_id]
                self.optimizer_adapt_s2t = self.optimizers_adapt_s2t[block_id]
                self.block_s = nn.Sequential(self.block_s, adaption_s2t)
            elif block_id == len(self.blocks_s) - 1:
                adaption_t2s = self.adaptions_t2s[block_id-1]
                self.optimizer_adapt_t2s = self.optimizers_adapt_t2s[block_id-1]
                self.block_s = nn.Sequential(adaption_t2s, self.block_s)
            
            self.state.iter = self._state.start_iter = start_iter
            self.state.max_iter = epochs_i*len(self.dataloader)
            self.state.epoch_length = epoch_length if epoch_length else len(self.dataloader)
            self.state.dataloader = self.dataloader
            self.state.dataloader_iter = iter(self.dataloader)
            self.state.step_fn = self.step_fn
            self.best_accuarcy = 0.0
           
            self.trigger_events(DefaultEvents.BEFORE_RUN)
            for self.state.iter in range( start_iter, self.state.max_iter ):
                if self.state.epoch_length!=None and \
                    self.state.iter%self.state.epoch_length==0: # Epoch Start
                        self.trigger_events(DefaultEvents.BEFORE_EPOCH)
                
                self.trigger_events(DefaultEvents.BEFORE_STEP)
                self.state.batch = self._get_batch()
                step_output = self.step_fn(self.state.batch)
                if isinstance(step_output, dict):
                    self.state.metrics.update(step_output)
                self.trigger_events(DefaultEvents.AFTER_STEP) 

                if self.state.epoch_length!=None and \
                    (self.state.iter+1)%self.state.epoch_length==0: # Epoch End
                        self.accuracy = self.test()
                        if self.best_accuarcy < self.accuracy:
                            self.best_accuarcy = self.accuracy
                        block_warp = self.warp_block(self.blocks_s, block_id, self.adaptions_t2s, self.adaptions_s2t)
                        self.params_s_best['block-{}'.format(block_id)] \
                            = block_warp.cpu().state_dict().copy()  # deep copy !!!
                        block_warp.cuda()
                        if self.logger_block:
                            self.logger_block.write('Accuracy-B{}'.format(block_id), self.accuracy)
                            self.trigger_events(DefaultEvents.AFTER_EPOCH)
        self.trigger_events(DefaultEvents.AFTER_RUN)
        
        for block_id in range(len(self.blocks_s)):
            block = self.warp_block(self.blocks_s, block_id, self.adaptions_t2s, self.adaptions_s2t)
            block.load_state_dict(self.params_s_best['block-{}'.format(block_id)])
            block.cuda()
            self.teacher.set_scion(block, self.blocks_graft_ids[block_id], 1)
            accuracy = self.test()
            if self.logger_block:
                self.logger_block.write('Test-Best-Accuracy-B{}'.format(block_id), accuracy)
        if self.logger_block:
            self.logger_block.close()
        with open('/home/yxy/kdcode/NetGraft/ckpt/student/vgg16-student-graft-block-{}-{}perclass.pth'.\
                format(self.args.dataset, self.args.num_per_class), 'bw') as f:
            torch.save(self.params_s_best, f)


    def step_fn(self,batch):
        start_time = time.perf_counter()
        batch = move_to_device(batch, self.device)
        [inputs]= batch
        self.teacher.reset_scion()
        logits_t = self.teacher(inputs).detach()
        self.teacher.set_scion(self.block_s, self.block_graft_id, self.block_len)
        logits_s = self.teacher(inputs)

        if self.args.norm_loss:
            loss = F.mse_loss(F.normalize(logits_s), F.normalize(logits_t), reduction='sum')
        else:
            loss = F.mse_loss(logits_s, logits_t, reduction='mean')
        loss_dict ={
             "loss_mse": loss 
        }
        loss = sum( loss_dict.values() )
        if self.block_idx > 0 and self.block_idx < len(self.blocks_s) - 1:
            self.optimizer_s.zero_grad()
            self.optimizer_adapt_s2t.zero_grad()
            self.optimizer_adapt_t2s.zero_grad()
            loss.backward()

            self.optimizer_s.step()
            self.optimizer_adapt_s2t.step()
            self.optimizer_adapt_t2s.step()
        elif self.block_idx == 0:
            self.optimizer_s.zero_grad()
            self.optimizer_adapt_s2t.zero_grad()
            loss.backward()
            self.optimizer_s.step()
            self.optimizer_adapt_s2t.step()
        elif self.block_idx == len(self.blocks_s) - 1:
            self.optimizer_s.zero_grad()
            self.optimizer_adapt_t2s.zero_grad()
            loss.backward()
            self.optimizer_s.step()
            self.optimizer_adapt_t2s.step()
        step_time = time.perf_counter() - start_time
        metrics = { loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'total_loss': loss.item(),
            'step_time': step_time,
        })
        return metrics
    
    def test(self):
        self.teacher.eval()
        correct = 0
        for i, (data, target) in enumerate(self.test_loader):
            data, target = data.cuda(), target.cuda()
            output = self.teacher(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(self.test_loader.dataset)
        return acc
    
    def warp_block(self,blocks_s, block_id, adaptions_t2s, adaptions_s2t):
        block_s = blocks_s[block_id]

        if block_id > 0 and block_id < len(blocks_s) - 1:
            adaption_t2s = adaptions_t2s[block_id-1]
            adaption_s2t = adaptions_s2t[block_id]

            block_s = nn.Sequential(adaption_t2s, block_s, adaption_s2t)
        elif block_id == 0:
            adaption_s2t = adaptions_s2t[block_id]

            block_s = nn.Sequential(block_s, adaption_s2t)
        elif block_id == len(blocks_s) - 1:
            adaption_t2s = adaptions_t2s[block_id-1]

            block_s = nn.Sequential(adaption_t2s, block_s)
        return block_s

class FSKD_NET_Distiller(KDDistiller):

    def __init__(self, logger=None, tb_writer=None ):
        super(FSKD_NET_Distiller, self).__init__( logger, tb_writer )

    def setup(self,args, s_blocks, teacher, dataloader,test_loader,s_blocks_graft_ids, s_blocks_len,adaptions, device=None ):
        self.adaptions_t2s, self.adaptions_s2t = adaptions
        self.num_block = len(s_blocks_graft_ids)
        self.blocks_s = []
        # optimizer = optim.Adam(block.parameters(), lr=0.0001)
        for block_id in range(self.num_block):
            self.blocks_s.append(
                self.warp_block(s_blocks, block_id, self.adaptions_t2s, self.adaptions_s2t).cuda()
            )
        params = torch.load('/home/yxy/kdcode/NetGraft/ckpt/student/vgg16-student-graft-block-{}-{}perclass.pth'.\
                        format(args.dataset, args.num_per_class))
        for block_id in range(self.num_block):
            self.blocks_s[block_id].load_state_dict(
                params['block-{}'.format(block_id)]
            )
        
        self.teacher = teacher
        self.dataloader = dataloader
        self.test_loader = test_loader
        self.args = args
        self.blocks_graft_ids = s_blocks_graft_ids
        self.blocks_len = s_blocks_len
        self.device = device
        self.params_s_best = OrderedDict()
        self.logger_net = logger.Logger('/home/yxy/kacode/KamalEngine/examples/kd/log/graft_net_{}_{}_{}perclass.txt'.\
                    format(args.dataset, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
                           args.num_per_class))

    def run(self, epochs_list, start_iter=0, epoch_length=None):
        
        for block_id in range(self.num_block-1):
            self.block = nn.Sequential(*self.blocks_s[:(block_id + 2)])
            self.optimizer = optim.Adam(self.block.parameters(), lr=0.0001)
            self.scion_len = sum(self.blocks_len[:(block_id + 2)])
            accuracy_best_block = 0.0
            # params_best_save = None
            
            self.block_idx = block_id
            epochs_i = epochs_list[block_id]
                
            self.state.iter = self._state.start_iter = start_iter
            self.state.max_iter = epochs_i*len(self.dataloader)
            self.state.epoch_length = epoch_length if epoch_length else len(self.dataloader)
            self.state.dataloader = self.dataloader
            self.state.dataloader_iter = iter(self.dataloader)
            self.state.step_fn = self.step_fn
            self.best_accuarcy = 0.0

            self.trigger_events(DefaultEvents.BEFORE_RUN)
            for self.state.iter in range( start_iter, self.state.max_iter ):
                if self.state.epoch_length!=None and \
                    self.state.iter%self.state.epoch_length==0: # Epoch Start
                        self.trigger_events(DefaultEvents.BEFORE_EPOCH)
                self.trigger_events(DefaultEvents.BEFORE_STEP)
                self.state.batch = self._get_batch()
                step_output = self.step_fn(self.state.batch)
                if isinstance(step_output, dict):
                    self.state.metrics.update(step_output)
                self.trigger_events(DefaultEvents.AFTER_STEP)        
                if self.state.epoch_length!=None and \
                    (self.state.iter+1)%self.state.epoch_length==0: # Epoch End
                        self.accuracy = self.test()
                        if self.best_accuarcy < self.accuracy:
                            self.best_accuarcy = self.accuracy
                            params_tmp = self.block.cpu().state_dict()
                            self.params_best_save = params_tmp.copy()
                            self.block.cuda()
                        if self.logger_net:
                            self.logger_net.write('Accuracy-length-{}'.format(self.scion_len), self.accuracy)
                        self.trigger_events(DefaultEvents.AFTER_EPOCH)
                
            if  block_id == (self.num_block - 2):
                self.block.load_state_dict(self.params_best_save)
            

        self.trigger_events(DefaultEvents.AFTER_RUN) 
        if self.logger_net:
            self.logger_net.write('Student Best Accuracy', accuracy_best_block)

        with open('/home/yxy/kacode/KamalEngine/examples/kd/ckpt/student/vgg16-student-graft-net-{}-{}perclass.pth'\
                            .format(self.args.dataset, self.args.num_per_class), 'wb') as f:
            torch.save(self.block.state_dict(), f)
        if self.logger_net:
            self.logger_net.close()
    

    def step_fn(self, batch):
        start_time = time.perf_counter()
        batch = move_to_device(batch, self.device)
        [inputs] = batch
        self.teacher.eval()
        self.block.train()
        self.teacher.reset_scion()
        logits_t = self.teacher(inputs).detach()
        self.teacher.set_scion(self.block, self.blocks_graft_ids[0], self.scion_len)
        logits_s = self.teacher(inputs)

        if self.args.norm_loss:
            loss = F.mse_loss(F.normalize(logits_s), F.normalize(logits_t), reduction='sum')
        else:
            loss = F.mse_loss(logits_s, logits_t, reduction='mean')
        loss_dict ={
             "loss_mse": loss 
        }
        loss = sum( loss_dict.values() )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        step_time = time.perf_counter() - start_time
        # if self.logger_net:
        #     self.logger_net.write('Loss-length-{}'.format(self.scion_len), loss.item())

        metrics = { loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'total_loss': loss.item(),
            'step_time': step_time,
        })
        return metrics
    
    def test(self):
        self.teacher.eval()
        correct = 0
        for i, (data, target) in enumerate(self.test_loader):
            data, target = data.cuda(), target.cuda()
            output = self.teacher(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(self.test_loader.dataset)
        return acc
    
    def warp_block(self,blocks_s, block_id, adaptions_t2s, adaptions_s2t):
        block_s = blocks_s[block_id]

        if block_id > 0 and block_id < len(blocks_s) - 1:
            adaption_t2s = adaptions_t2s[block_id-1]
            adaption_s2t = adaptions_s2t[block_id]

            block_s = nn.Sequential(adaption_t2s, block_s, adaption_s2t)
        elif block_id == 0:
            adaption_s2t = adaptions_s2t[block_id]

            block_s = nn.Sequential(block_s, adaption_s2t)
        elif block_id == len(blocks_s) - 1:
            adaption_t2s = adaptions_t2s[block_id-1]

            block_s = nn.Sequential(adaption_t2s, block_s)
        return block_s



import torch
import torch.nn as nn
from kamal.core.engine.engine import Engine, Event, DefaultEvents, State
from kamal.core import tasks
from kamal.utils import set_mode, move_to_device, get_logger, split_batch
from kamal.utils.center_train_utils import get_translator_output
from typing import Callable, Mapping, Any, Sequence
import time
import weakref
import torch.nn.functional as F

class LocalTrainer(Engine):
    def __init__( self, 
                  logger=None,
                  tb_writer=None):
        super(LocalTrainer, self).__init__(logger=logger, tb_writer=tb_writer)

    def setup(self, 
              model, 
              dataloader,
              test_dataloader,
              optimizer, 
              scheduler,
              args,
              device,
              distribution_info,
              class_index,
              num_groups,
              count,
              model_path):
        
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device
        self.model = model
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.distribution_info = distribution_info
        self.class_index = class_index
        self.num_groups = num_groups
        self.count = count
        self.model_path = model_path
    def run( self, max_iter, start_iter=0, epoch_length=None):
        best_acc1 = 0
        self.state.iter = self._state.start_iter = start_iter
        self.state.max_iter = max_iter
        self.state.epoch_length = epoch_length if epoch_length else len(self.dataloader)
        self.state.dataloader = self.dataloader
        self.state.dataloader_iter = iter(self.dataloader)
        self.state.step_fn = self.step_fn
        self.model.to(self.device)
        with set_mode(self.model, training=True):
            self.trigger_events(DefaultEvents.BEFORE_RUN)
            # super( LocalTrainer, self ).run( self.step_fn, self.dataloader, start_iter=start_iter, max_iter=max_iter, epoch_length=epoch_length)
            for self.state.iter in range( start_iter, self.state.max_iter ):
                if self.state.epoch_length!=None and \
                    self.state.iter%self.state.epoch_length==0: # Epoch Start
                        if(self.state.iter // self.state.epoch_length>self.args.warmup):
                            self.scheduler.step()
                        self.trigger_events(DefaultEvents.BEFORE_EPOCH)
                self.trigger_events(DefaultEvents.BEFORE_STEP)
                self.state.batch = self._get_batch()
                step_output = self.step_fn(self, self.state.batch) 
                if isinstance(step_output, dict):
                    self.state.metrics.update(step_output)     
                self.trigger_events(DefaultEvents.AFTER_STEP)
                if self.state.iter % 10 == 0:
                    content = "Iter %d/%d (Epoch %d/%d, Batch %d/%d)"%(
                      self.state.iter, self.state.max_iter, 
                      self.state.iter // self.state.epoch_length, self.state.max_iter // self.state.epoch_length, 
                      self.state.iter % self.state.epoch_length, self.state.epoch_length
                    )
                    content += " %s=%.4f"%('loss', step_output['loss'])
                    content += " %s=%.4f"%('lr', step_output['lr'])
                    self.logger.info(content)
                if self.state.epoch_length!=None and \
                   (self.state.iter+1)%self.state.epoch_length==0: # Epoch End
                    self.trigger_events(DefaultEvents.AFTER_EPOCH)
                    acc1 = self.validate()
                    is_best = acc1 > best_acc1
                    best_acc1 = max(acc1, best_acc1)
                    save_dict = {
                        "label_info": self.distribution_info,
                        "weight": self.model.state_dict(),
                        "current_epoch": self.state.iter // self.state.epoch_length,
                        "best_acc": float(best_acc1),
                        "data_range": [self.class_index, self.class_index+self.num_groups[self.count]-1]
                    }
                    torch.save(save_dict,self.model_path)
                    self.trigger_events(DefaultEvents.AFTER_EPOCH)
        self.logger.info("model_%d-%d-best_acc: %.4f" % (self.class_index, self.class_index+self.num_groups[self.count]-1, best_acc1))
        self.trigger_events(DefaultEvents.AFTER_RUN)


    def step_fn(self,engine, batch):
        model = self.model
        start_time = time.perf_counter()
        batch = move_to_device(batch, self.device)
        inputs, targets = split_batch(batch)
        outputs = model(inputs)
        loss_fn=nn.CrossEntropyLoss()
        loss = loss_fn(outputs, targets) # get loss
        loss_dict = {'loss':loss}
        loss_ka = sum( loss_dict.values() )
        self.optimizer.zero_grad()
        loss_ka.backward()
        self.optimizer.step()
        step_time = time.perf_counter() - start_time
        metrics = { loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'loss': loss_ka.item(),
            'step_time': step_time,
            'lr': float( self.optimizer.param_groups[0]['lr'] )
        })
        return metrics
    def validate(self):
        with set_mode(self.model, training=False):
            top1 = AverageMeter('Acc@1', ':6.2f')
            pred=[]
            ground_truth=[]
            with torch.no_grad():
                for idx,(img,label) in enumerate(self.test_dataloader):
                    img=img.to(self.device)
                    label=label.to(self.device)
                    out = self.model(img)
                    out_logits=F.softmax(out,1)

                    pred.extend((torch.argmax(out,1)).cpu().numpy().tolist())
                    ground_truth.extend(list(label.cpu().numpy()))

                    acc1= accuracy(out, label, topk=(1,))
                    top1.update(acc1[0].item(), img.size(0))
            self.logger.info( "[Eval %s] Iter %d/%d: {'acc':%.4f}"%('model', self.state.iter , self.state.max_iter, top1.avg) )
        return top1.avg 

class FEDSATrainer(Engine):
    def __init__( self, 
                  logger=None,
                  tb_writer=None):
        super(FEDSATrainer, self).__init__(logger=logger, tb_writer=tb_writer)

    def setup(self, 
              teacher, 
              student,
              translators,
              dataloader,
              optimizer, 
              args,
              device,
              kd_loss_memter,
              loss_fn):
        
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device
        self.teacher = teacher
        self.student = student
        self.translators = translators
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.args = args
        self.kd_loss_memter = kd_loss_memter
        self.loss_fn = loss_fn

    def run( self, max_iter, start_iter=0, epoch_length=None):
        self.teacher.to(self.device)
        self.student.to(self.device)
        for i in range(len(self.translators)):
            self.translators[i].to(self.device)
            self.translators[i].train()
        with set_mode(self.student, training=True), \
             set_mode(self.teacher, training=False):
            super(FEDSATrainer, self ).run(self.step_fn, self.dataloader, start_iter=start_iter, max_iter=max_iter, epoch_length=epoch_length)

    def step_fn(self,engine, batch):
        local_loss=0
        student = self.student
        teacher = self.teacher
        translators = self.translators
        start_time = time.perf_counter()
        batch = move_to_device(batch, self.device)
        inputs, targets = split_batch(batch)
        with torch.no_grad():
            t_out,t_features = teacher(inputs,return_features=1)
            t_factors=t_features
                        #t_factors,_=get_para_output(paraphrasers,t_features)
                    
        s_out,s_features=student(inputs,return_features=1)
        s_factors=get_translator_output(translators,s_features)    
        kd_losses=[self.loss_fn(s_factors[i],t_factors[i].detach()) for i in range(len(s_factors))]
        loss=sum(kd_losses)
        loss_dict = { 
            'loss':  loss
        }
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        local_loss+=loss.item()
        self.kd_loss_memter.update(loss.item())
        step_time = time.perf_counter() - start_time
        metrics = { loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'loss': local_loss,
            'step_time': step_time,
            'lr': float( self.optimizer.param_groups[0]['lr'] )
        })
        return metrics
  

def accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].view(-1).float().sum(0) * 100. / batch_size for k in topk]  

class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self, name, fmt=':f'):
            self.name = name
            self.fmt = fmt
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

        def __str__(self):
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)

class TestTrainer(Engine):
    def __init__( self, 
                  logger=None,
                  tb_writer=None):
        super(TestTrainer, self).__init__(logger=logger, tb_writer=tb_writer)

    def setup(self, 
              model, 
              dataloader,
              test_dataloader,
              optimizer, 
              scheduler,
              args,
              device):
        
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device
        self.model = model
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def run( self, max_iter, start_iter=0, epoch_length=None):
        self.model.to(self.device)
        with set_mode(self.model, training=True):
            super( TestTrainer, self ).run( self.step_fn, self.dataloader, start_iter=start_iter, max_iter=max_iter, epoch_length=epoch_length)

    def step_fn(self,engine, batch):
        model = self.model
        start_time = time.perf_counter()
        batch = move_to_device(batch, self.device)
        inputs, targets = split_batch(batch)
        outputs = model(inputs)
        loss_fn=nn.CrossEntropyLoss()
        loss = loss_fn(outputs, targets) # get loss
        loss_dict = {'loss':loss}
        loss_ka = sum( loss_dict.values() )
        self.optimizer.zero_grad()
        loss_ka.backward()
        self.optimizer.step()
        step_time = time.perf_counter() - start_time
        metrics = { loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'loss': loss_ka.item(),
            'step_time': step_time,
            'lr': float( self.optimizer.param_groups[0]['lr'] )
        })
        return metrics
import torch.optim as optim
import torch.nn as nn
import torch
import math
import abc
import weakref
import typing
import time
import numpy as np 

from .task import TaskBase
from .callbacks import CallbackBase
from ...utils.logger import get_logger
from ...utils import comm
from .ctx import train_ctx, device_ctx
from .history import HistoryStorage



class TrainerBase(abc.ABC):
    def __init__(self, logger=None):
        self.logger = logger if logger is not None else get_logger(name='kamal', color=True)
        self._callbacks = []

    def train(self, start_iter, max_iter):
        self.iter = start_iter
        self.start_iter, self.max_iter = start_iter, max_iter

        self.history = HistoryStorage(start_iter)
        self.before_train()
        for self.iter in range( start_iter, max_iter ):
            self.before_step()
            self.step()
            self.after_step()
            self.history.step()
        self.after_train()
            
    def add_callbacks(self, callbacks: typing.Sequence[ CallbackBase ]):
        for callback in callbacks:
            callback.trainer = weakref.ref(self)
        self._callbacks.extend( callbacks )

    @abc.abstractmethod
    def step(self):
        pass

    def before_train(self):
        for callback in self._callbacks:
            callback.before_train()

    def after_train(self):
        for callback in self._callbacks:
            callback.after_train()

    def before_step(self):
        for callback in self._callbacks:
            callback.before_step()

    def after_step(self):
        for callback in self._callbacks:
            callback.after_step()
    

class SimpleTrainer(TrainerBase):
    def __init__(   self, 
                    task: TaskBase, 
                    model: nn.Module, 
                    train_loader, 
                    optimizer, 
                    device=None,
                    logger=None, ):
        super(SimpleTrainer, self).__init__(logger)
        self.task = task
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.model = model

        self._train_loader_iter = iter(train_loader)

    def train(self, start_iter, max_iter, device=None):
        self.device = device if device is not None else \
            torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        
        with train_ctx(self.model), device_ctx(self.model, self.device):
            super( SimpleTrainer, self ).train( start_iter, max_iter )
        
    def step(self):
        self.optimizer.zero_grad()
        start_time = time.perf_counter()
        # prepare data
        try:
            data = next( self._train_loader_iter )
        except StopIteration:
            # reset iterator
            self._train_loader_iter = iter(self.train_loader)
            data = next( self._train_loader_iter )
        if not isinstance( data, typing.Iterable ):
            data = [data, ]
        data = [ d.to(self.device) for d in data ]

        # get loss
        loss_dict = self.task.get_loss( self.model, *data )
        loss = sum( loss_dict.values() )
        loss.backward()
        # update weights
        self.optimizer.step()
        step_time = time.perf_counter() - start_time

        # record training info
        info = loss_dict
        info['step_time'] = float(step_time)
        self._gather_training_info( info )

    def _gather_training_info(self, info):
        info = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in info.items()
        }

        all_info = comm.gather(info)
        if comm.is_main_process():
            if "step_time" in all_info[0]:
                step_time = np.max([x.pop("step_time") for x in all_info])
                self.history.put_scalar("step_time", step_time)

            # average the rest metrics
            info = {
                k: np.mean([x[k] for x in all_info]) for k in all_info[0].keys()
            }
            total_losses_reduced = sum(loss for loss in info.values())
            self.history.put_scalar("total_loss", total_losses_reduced)

            if len(info) > 1:
                self.history.put_scalars(**info)




def eval(model, criterion, test_loader, metric, device=None, num_val_batch=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric.reset()
    model.to(device)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            if num_val_batch is not None and i>=num_val_batch:
                break # early stop

            img, target = img.to(device), target.to(device)
            out = model(img)
            pred = out.max(1)[1]
            loss = criterion( out, target )
            val_loss+=loss.detach().cpu().numpy()
            metric.update( pred, target)
    return metric.get_results(return_key_metric=True), val_loss/len(test_loader)

def train(model, criterion, optimizer, scheduler, train_loader, 
          test_loader, metric, val_criterion=None, pth_path=None, target_score=None,
          total_epochs=30, total_itrs=None, val_interval=None, verbose=False, weights_only=True, num_val_batch=None):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_score = -1
    best_val_loss = 999999

    cur_itr = 1
    cur_epoch = 1

    if total_itrs is None:
        total_itrs = len(train_loader)*total_epochs
    else:
        total_epochs =  math.ceil( total_itrs / len(train_loader) )

    if val_interval is None:
        val_interval = len(train_loader)

    if val_criterion is None:
        val_criterion = criterion

    achieve_target_score=False

    while True:
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            if cur_itr%10==0 and verbose:
                print("Epoch %d/%d, Batch %d/%d, iter %d/%d, loss=%.4f"%(cur_epoch, total_epochs, i+1, len(train_loader), cur_itr, total_itrs, loss.item()))

            if cur_itr%val_interval==0:
                model.eval()
                (metric_name, score), val_loss = eval(model=model, 
                                                       criterion=val_criterion, 
                                                       test_loader=test_loader, 
                                                       metric=metric, 
                                                       device=device, num_val_batch=num_val_batch)
                print("[TEST] Epoch %d/%d, val_loss=%.4f, %s=%.4f\n"%(cur_epoch, total_epochs, val_loss, metric_name, score))
                
                if target_score is not None and target_score<score:
                    achieve_target_score=True

                if best_score<score:
                    if pth_path is not None:
                        if weights_only:
                            torch.save( model.state_dict(), pth_path )
                        else:
                            torch.save( model, pth_path )
                    best_score=score
                    best_val_loss=val_loss
                model.train()

            if scheduler is not None:
                scheduler.step()
            
            if cur_itr>=total_itrs or achieve_target_score:
                print("val_loss=%.4f, best %s=%.4f"%(best_val_loss, metric_name, best_score))
                return best_score, best_val_loss
            cur_itr+=1
        cur_epoch+=1

def kd(student, teacher, criterion, optimizer, scheduler, train_loader, 
          test_loader, metrics, val_criterion=None, pth_path=None, total_epochs=30, total_itrs=None, val_interval=None, verbose=False):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student.train().to(device)
    teacher.eval().to(device)

    best_score = -1
    best_val_loss = 999999
    cur_itr = 1
    cur_epoch = 1

    if total_itrs is None:
        total_itrs = len(train_loader)*total_epochs
    else:
        total_epochs = total_itrs // len(train_loader)
    if val_interval is None:
        val_interval = len(train_loader)
    if val_criterion is None:
        val_criterion = criterion

    while True:
        student.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            s_out = student(img)
            with torch.no_grad():
                t_out = teacher(img)
            loss = criterion(s_out, t_out)
            loss.backward()
            optimizer.step()

            if cur_itr%10==0 and verbose:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f"%(cur_epoch, total_epochs, cur_itr, len(train_loader), loss.item()))
            
            if cur_itr%val_interval==0:
                student.eval()
                (metric_name, score), val_loss = eval(model=student,
                                                        criterion=nn.CrossEntropyLoss(), 
                                                        test_loader=test_loader, 
                                                        metrics=val_criterion, 
                                                        device=device)
                print("Epoch %d/%d, iter %d/%d val_loss=%.4f, %s=%.4f\n"%(cur_epoch, total_epochs, cur_itr, total_itrs, val_loss, metric_name, score))
                if best_score<score:
                    if pth_path is not None:
                        torch.save( student, pth_path )
                        print("Best model saved as %s"%(pth_path))
                    best_score=score
                    best_val_loss=val_loss
                student.train()
            if cur_itr == total_itrs:
                print("val_loss=%.4f, best %s=%.4f"%(best_val_loss, metric_name, best_score))
                return best_score, best_val_loss
            cur_itr+=1
            if scheduler is not None:
                scheduler.step()
        cur_epoch+=1

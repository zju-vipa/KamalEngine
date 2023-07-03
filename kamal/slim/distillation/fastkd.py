
import torch,os
import time
import torch.nn.functional as F
from kamal import utils
from kamal.utils import set_mode, move_to_device
from kamal import metrics
from kamal.core.engine.trainer import Engine
from kamal.slim.distillation.data_free import criterions


class FKDDistiller(Engine):

    def __init__(self, logger=None, tb_writer=None):
        super(FKDDistiller, self).__init__(logger=logger, tb_writer=tb_writer)

    def setup(self,
              student,
              teacher,
              scheduler,
              evaluator,
              synthesizer,
              val_loader,
              optimizer,
              criterion,
              args,
              device=None):
        self.student = student
        self.teacher = teacher
        self.evaluator = evaluator
        self.synthesizer = synthesizer
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.args = args
        self.criterion = criterion
        self.scheduler = scheduler

        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

    def run(self, max_iter, start_iter=0, epoch_length=None):
        time_cost = 0
        best_acc1 = 0
        self.iter = start_iter
        self.max_iter = max_iter
        self.epoch_length = epoch_length if epoch_length else len(self.val_loader)
        with set_mode(self.student, training=True), \
                set_mode(self.teacher, training=False):
            for self.iter in range(start_iter, max_iter):
                if self.epoch_length != None and (self.iter) % self.epoch_length == 0:
                    vis_results,cost = self.synthesizer.synthesize()
                    time_cost+=cost
                    for vis_name, vis_image in vis_results.items():
                        utils._utils.save_image_batch( vis_image, 'checkpoints/datafree-%s/%s%s.png'%(self.args.method, vis_name, self.args.log_tag) )
                for _ in range(10):
                    ka_output = self.ka_step()
                if self.iter % 10 == 0:
                    content = "Iter %d/%d (Epoch %d/%d, Batch %d/%d)"%(
                      self.iter, self.max_iter, 
                      self.iter // self.epoch_length, self.max_iter // self.epoch_length, 
                      self.iter % self.epoch_length, self.epoch_length
                    )
                    content += " %s=%.4f"%('loss', ka_output['loss'])
                    content += " %s=%.4f"%('lr',ka_output['lr'] )
                    self.logger.info(content)

                # EPOCH END
                if self.epoch_length != None and (self.iter + 1) % self.epoch_length == 0:
                    self.scheduler.step()
                    acc1 = self.validate()
                    is_best = acc1 > best_acc1
                    best_acc1 = max(acc1, best_acc1)
                    os.makedirs('checkpoints', exist_ok=True )
                    pth_path = os.path.join('checkpoints', "%s_%s_latest_%08d_%s_%s_%.4f.pth" %
                                            (self.args.method,self.args.dataset, self.iter, self.args.teacher,self.args.student, best_acc1))
                    save_checkpoint({
                        'epoch': self.iter // self.epoch_length,
                        'iter' : self.iter,
                        's_state_dict': self.student.state_dict(),
                        'best_acc1': float(best_acc1),
                        'optim': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                    }, is_best, pth_path)
        best_path = os.path.join('checkpoints', "%s_%s_best_%08d_%s_%s_%.4f.pth" %
                                            (self.args.method,self.args.dataset, self.iter, self.args.teacher,self.args.student, best_acc1))
        torch.save(self.student.state_dict(), best_path)
        self.logger.info("best_acc: %.4f" % best_acc1)
        self.logger.info("Generation Cost: %1.3f" % (time_cost/3600.) )

    def ka_step(self):
        start_time = time.perf_counter()
        if self.args.method in ['zskt', 'dfad', 'dfq', 'dafl']:
            images, cost = self.synthesizer.sample()
        else:
            images = self.synthesizer.sample()
        if self.args.gpu is not None:
            images = move_to_device(images, self.device)
        with torch.no_grad():
            t_out, t_feat = self.teacher(images, return_features=True)
        s_out = self.student(images.detach())
        loss = self.criterion(s_out, t_out.detach())        
        loss_dict = {"loss": loss}
        loss_ka = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss_ka.backward()
        self.optimizer.step()

        step_time = time.perf_counter() - start_time
        metrics = {loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items()}
        metrics.update({
            'loss_ka': loss_ka.item(),
            'step_time': step_time,
            'lr': float( self.optimizer.param_groups[0]['lr'] )})
        return metrics
    
    def validate(self):
        with set_mode(self.student, training=False), \
                set_mode(self.teacher, training=False):
            eval_results = self.evaluator(self.student, device=self.device)
            (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
            self.logger.info( "[Eval %s] Iter %d/%d: {'fast_kd_acc':%.4f}"%('model', self.iter, self.max_iter, acc1) )
        return acc1

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)
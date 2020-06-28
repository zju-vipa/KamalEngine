import torch
import time
import torch.nn.functional as F
from kamal.slim.distillation.kd import KDDistiller
from kamal.utils import set_mode
from kamal.core.criterion import kldiv

class ZSKTDistiller(KDDistiller):
    def __init__(   self, 
                    student,
                    teacher,
                    generator,
                    z_dim,
                    logger=None,
                    viz=None):
        super(ZSKTDistiller, self).__init__(logger, viz)
        self.teacher = teacher
        self.model = self.student = student
        self.generator = generator
        self.z_dim = z_dim

    def train(self, start_iter, max_iter, optim_s, optim_g, device=None):
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device
        self.optim_s, self.optim_g = optim_s, optim_g

        self.model.to(self.device)
        self.teacher.to(self.device)
        self.generator.to(self.device)
        self.train_loader = [0, ]
        
        with set_mode(self.student, training=True), \
             set_mode(self.teacher, training=False), \
             set_mode(self.generator, training=True):
            super( ZSKTDistiller, self ).train( start_iter, max_iter )
    
    def search_optimizer(self, evaluator, train_loader, hpo_space=None, mode='min', max_evals=20, max_iters=400):
        optimizer = hpo.search_optimizer(self, train_loader, evaluator=evaluator, hpo_space=hpo_space, mode=mode, max_evals=max_evals, max_iters=max_iters)
        return optimizer
    
    def step(self):
        start_time = time.perf_counter()
        
        # Adv
        z = torch.randn( self.z_dim ).to(self.device)
        fake = self.generator( z )
        self.optim_g.zero_grad()
        t_out = self.teacher( fake )
        s_out = self.student( fake )
        loss_g = -kldiv( s_out, t_out )
        loss_g.backward()
        self.optim_g.step()

        with torch.no_grad():
            fake = self.generator( z )
            t_out = self.teacher( fake.detach() )
        for _ in range(10):
            self.optim_s.zero_grad()
            s_out = self.student( fake.detach() )
            loss_s = kldiv( s_out, t_out )
            loss_s.backward()
            self.optim_s.step()
        
        loss_dict = {
            'loss_g': loss_g,
            'loss_s': loss_s,
        }

        step_time = time.perf_counter() - start_time

        # record training info
        info = loss_dict
        info['step_time'] = step_time
        info['lr_s'] = float( self.optim_s.param_groups[0]['lr'] )
        info['lr_g'] = float( self.optim_g.param_groups[0]['lr'] )
        self.history.put_scalars( **info )
    
    def reset(self):
        self.history = None
        self._train_loader_iter = iter(train_loader)
        self.iter = self.start_iter

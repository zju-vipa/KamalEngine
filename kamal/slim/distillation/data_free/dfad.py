import torch
import time
import torch.nn.functional as F
from kamal.utils import set_mode
from kamal.core.engine.trainer import Engine


class DFADistiller(Engine):
    def __init__(self,
                 logger=None,
                 tb_writer=None):
        super(DFADistiller, self).__init__(logger=logger, tb_writer=tb_writer)

    def setup(self, student, teacher, generator, dataloader, optimizer, bs, nz, device=None):
        self.model = self.student = student
        self.teacher = teacher
        self.generator = generator
        self.dataloader = dataloader
        self.optimizer_S, self.optimizer_G = optimizer
        self.batch_size = bs
        self.nz = nz

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.student.to(self.device)
        self.teacher.to(self.device)
        self.generator.to(self.device)

    def run(self, max_iter, start_iter=0, epoch_length=None):
        with set_mode(self.student, training=True), \
                set_mode(self.generator, training=True), \
                set_mode(self.teacher, training=False):
            super(DFADistiller, self).run(self.step_fn, self.dataloader, start_iter=start_iter, max_iter=max_iter, epoch_length=epoch_length)

    def step_fn(self, engine, batch):
        start_time = time.perf_counter()

        for _ in range(5):
            z = torch.randn((self.batch_size, self.nz, 1, 1)).to(self.device)
            self.optimizer_S.zero_grad()
            fake = self.generator(z).detach()
            t_logit = self.teacher(fake)
            s_logit = self.student(fake)
            loss_S = F.l1_loss(s_logit, t_logit.detach())

            loss_S.backward()
            self.optimizer_S.step()

        z = torch.randn((self.batch_size, self.nz, 1, 1)).to(self.device)
        self.optimizer_G.zero_grad()
        self.generator.train()
        fake = self.generator(z)
        t_logit = self.teacher(fake)
        s_logit = self.student(fake)

        loss_G = - F.l1_loss(s_logit, t_logit)
        loss_G.backward()
        self.optimizer_G.step()

        loss_dict = {
            'loss_g': loss_G,
            'loss_s': loss_S,
        }
        step_time = time.perf_counter() - start_time

        metrics = {loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items()}
        metrics.update({
            'step_time': step_time,
            'lr_g': float(self.optimizer_G.param_groups[0]['lr']),
            'lr_s': float(self.optimizer_S.param_groups[0]['lr'])
        })
        return metrics

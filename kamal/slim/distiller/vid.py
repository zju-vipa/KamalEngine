import numpy as np
import time
import torch.nn as nn
import torch._ops
import torch.nn.functional as F
from .kd import KDDistiller
from kamal.core.loss import KDLoss

class VIDDistiller(KDDistiller):
    def __init__(self, student, teacher, regressor_l, T=1.0, gamma=1.0, alpha=None, beta=None, stu_hooks=[], tea_hooks=[], out_flags=[], logger=None, viz=None):
        super(VIDDistiller, self).__init__(student, teacher, T=T,
                                           gamma=gamma, alpha=alpha, logger=logger, viz=viz)
        self.regressor_l = regressor_l

        self.stu_hooks = stu_hooks
        self.tea_hooks = tea_hooks
        self.out_flags = out_flags

        self._beta = beta

    def train(self, start_iter, max_iter, train_loader, optimizer, device=None):
        if device is None:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        self.regressor_l = [regressor.to(device)
                            for regressor in self.regressor_l]
        super(VIDDistiller, self).train(start_iter,max_iter, train_loader, optimizer, device)

    def step(self):
        self.optimizer.zero_grad()
        start_time = time.perf_counter()

        self.student.train()
        self.teacher.eval()
        self.regressor_l = [regressor.train()
                            for regressor in self.regressor_l]
        try:
            data, targets = self._train_loader_iter.next()
        except StopIteration:
            self._train_loader_iter = iter(self.train_loader)  # reset iterator
            data, targets = self._train_loader_iter.next()
        data, targets = data.to(self.device), targets.to(self.device)
 
        s_out = self.student(data)
        feat_s =  [f.feat_out if flag else f.feat_in for (f, flag) in zip(self.stu_hooks, self.out_flags)]
        with torch.no_grad():
            t_out = self.teacher(data)
            feat_t = [f.feat_out.detach() if flag else f.feat_in for (f, flag) in zip(self.tea_hooks, self.out_flags)]
        g_s = feat_s[1:-1]
        g_t = feat_t[1:-1]
        loss_vid = [c(f_s, f_t)
                    for f_s, f_t, c in zip(g_s, g_t, self.regressor_l)]
        loss = self._gamma * nn.CrossEntropyLoss()(s_out, targets) + self._alpha * \
            KDLoss(T=self._T, use_kldiv=True)(
                s_out, t_out)  + self._beta * sum(loss_vid)
        loss.backward()

        # update weights
        self.optimizer.step()
        step_time = time.perf_counter() - start_time

        # record training info
        info = {'loss': loss}
        info['total_loss'] = float(loss.item())
        info['step_time'] = float(step_time)
        info['lr'] = float(self.optimizer.param_groups[0]['lr'])
        self._gather_training_info(info)

class VIDRegressor(nn.Module):
    """
        Variational Information Distillation for Knowledge Transfer (CVPR 2019)
        @inproceedings{tian2019crd,
        title={Contrastive Representation Distillation},
        author={Yonglong Tian and Dilip Krishnan and Phillip Isola},
        booktitle={International Conference on Learning Representations},
        year={2020}
        }
    """
    def __init__(self,
                num_input_channels,
                num_mid_channel,
                num_target_channels,
                init_pred_var=5.0,
                eps=1e-5):
        super(VIDRegressor, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
            )
        self.eps = eps

    def forward(self, input, target):
        # pool for dimentsion match
        s_H, t_H = input.shape[2], target.shape[2]
        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_H))
        else:
            pass
        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0+torch.exp(self.log_scale))+self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5*(
            (pred_mean-target)**2/pred_var+torch.log(pred_var)
            )
        loss = torch.mean(neg_log_prob)
        return loss

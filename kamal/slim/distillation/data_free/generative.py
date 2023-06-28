import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .base import BaseSynthesis
from .hooks import DeepInversionHook
from .criterions import kldiv, get_image_prior_losses
# from utils import ImagePool, DataIter, clip_images
from kamal import utils

class GenerativeSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, img_size, iterations=1,
                 lr_g=1e-3, synthesis_batch_size=128, sample_batch_size=128, 
                 adv=0, bn=0, oh=0, act=0, balance=0, criterion=None,
                 normalizer=None, device='cpu',
                 # TODO: FP16 and distributed training 
                 autocast=None, use_fp16=False, distributed=False):
        super(GenerativeSynthesizer, self).__init__(teacher, student)
        assert len(img_size)==3, "image size should be a 3-dimension tuple"
        self.img_size = img_size 
        self.iterations = iterations
        self.nz = nz
        if criterion is None:
            criterion = kldiv
        self.criterion = criterion
        self.normalizer = normalizer
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size

        # scaling factors
        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.balance = balance
        self.act = act

        # generator
        self.generator = generator.to(device).train()
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(0.5,0.999))
        self.distributed = distributed
        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.device = device

        # hooks for deepinversion regularization
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m) )

    def synthesize(self):
        self.student.eval()
        self.generator.train()
        self.teacher.eval()
        for it in range(self.iterations):
            self.optimizer.zero_grad()
            z = torch.randn( size=(self.synthesis_batch_size, self.nz), device=self.device )
            inputs = self.generator(z)
            inputs = self.normalizer(inputs)
            t_out, t_feat = self.teacher(inputs, return_features=True)
            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy( t_out, t_out.max(1)[1] )
            loss_act = - t_feat.abs().mean()
            if self.adv>0:
                s_out = self.student(inputs)
                loss_adv = -self.criterion(s_out, t_out)
            else:
                loss_adv = loss_oh.new_zeros(1)
            p = F.softmax(t_out, dim=1).mean(0)
            loss_balance = (p * torch.log(p)).sum() # maximization
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + self.balance * loss_balance + self.act * loss_act
            loss.backward()
            self.optimizer.step()
        return { 'synthetic': self.normalizer(inputs.detach(), reverse=True) }
    
    @torch.no_grad()
    def sample(self):
        self.generator.eval()
        z = torch.randn( size=(self.sample_batch_size, self.nz), device=self.device )
        inputs = self.normalizer(self.generator(z))
        return inputs

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time

from .base import BaseSynthesis
from .hooks import DeepInversionHook
from kamal.core import tasks
from .criterions import get_image_prior_losses
# from data_free.utils import ImagePool, DataIter, clip_images
from kamal import utils

def jitter_and_flip(inputs_jit, lim=1. / 8., do_flip=True):
    lim_0, lim_1 = int(inputs_jit.shape[-2] * lim), int(inputs_jit.shape[-1] * lim)

    # apply random jitter offsets
    off1 = random.randint(-lim_0, lim_0)
    off2 = random.randint(-lim_1, lim_1)
    inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

    # Flipping
    flip = random.random() > 0.5
    if flip and do_flip:
        inputs_jit = torch.flip(inputs_jit, dims=(3,))
    return inputs_jit


class DeepInvSyntheiszer(BaseSynthesis):
    def __init__(self, teacher, student, num_classes, img_size,
                 iterations=1000, lr_g=0.1, progressive_scale=False,
                 synthesis_batch_size=128, sample_batch_size=128,
                 adv=0.0, bn=1, oh=1, tv=1e-5, l2=0.0,
                 save_dir='run/deepinversion', transform=None,
                 normalizer=None, device='cpu',
                 # TODO: FP16 and distributed training
                 autocast=None, use_fp16=False, distributed=False):
        super(DeepInvSyntheiszer, self).__init__(teacher, student)
        assert len(img_size) == 3, "image size should be a 3-dimension tuple"

        self.save_dir = save_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.normalizer = normalizer
        self.data_pool = utils._utils.ImagePool(root=self.save_dir)
        self.data_iter = None
        self.transform = transform
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size

        # Scaling factors
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.tv = tv
        self.l2 = l2
        self.num_classes = num_classes
        self.distributed = distributed

        # training configs
        self.progressive_scale = progressive_scale
        self.use_fp16 = use_fp16
        self.autocast = autocast  # for FP16
        self.device = device

        # setup hooks for BN regularization
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, 0))
        self.s_hooks = []
        for m in student.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.s_hooks.append(DeepInversionHook(m, 0))
        assert len(self.hooks) > 0, 'input model should contains at least one BN layer for DeepInversion'

    def synthesize(self, targets=None):
        start = time.time()
        # kld_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        self.student.eval()
        best_cost = 1e6
        inputs = torch.randn(size=[self.synthesis_batch_size, *self.img_size], device=self.device).requires_grad_()
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
            targets = targets.sort()[0]  # sort for better visualization
        targets = targets.to(self.device)

        optimizer = torch.optim.Adam([inputs], self.lr_g, betas=[0.5, 0.99])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=self.iterations )

        best_inputs = inputs.data
        for it in range(self.iterations):
            inputs_aug = jitter_and_flip(inputs)
            t_out = self.teacher(inputs_aug)
            if self.adv > 0:
                s_out = self.student(inputs_aug)
                loss_adv = -tasks.loss.jsdiv(s_out, t_out, T=3)
            else:
                loss_adv = loss_oh.new_zeros(1)

            loss_bn = sum([h.r_feature for h in self.hooks]) - 0.001 * sum([h.r_feature for h in self.s_hooks])
            loss_oh = F.cross_entropy(t_out, targets)

            loss_tv = get_image_prior_losses(inputs)
            loss_l2 = torch.norm(inputs, 2)
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + self.tv * loss_tv + self.l2 * loss_l2

            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            inputs.data = utils._utils.clip_images(inputs.data, self.normalizer.mean, self.normalizer.std)
        self.student.train()
        # save best inputs and reset data loader
        if self.normalizer:
            best_inputs = self.normalizer(best_inputs, True)
        end = time.time()
        self.data_pool.add(best_inputs)
        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst) if self.distributed else None
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = utils._utils.DataIter(loader)
        return {'synthetic': best_inputs}, end - start

    def sample(self):
        return self.data_iter.next()
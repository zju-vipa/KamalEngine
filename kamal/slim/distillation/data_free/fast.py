
from typing import Generator
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from .base import BaseSynthesis
from .hooks import DeepInversionHook, InstanceMeanHook
from .criterions import kldiv_m
from kamal import utils
# from data_free.criterions import jsdiv, get_image_prior_losses, kldiv
# from data_free.utils import ImagePool, DataIter, clip_images
import collections
from torchvision import transforms
from kornia import augmentation
import time


def reset_l0(model):
    for n, m in model.named_modules():
        if n == "l1.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)


def reset_bn1(model):
    for n, m in model.named_modules():
        if n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)


def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


def reset_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


class FastSynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size,
                 init_dataset=None, iterations=100, lr_g=0.1,
                 synthesis_batch_size=128, sample_batch_size=128,
                 adv=0.0, bn=1, oh=1,
                 save_dir='run/fast', transform=None, autocast=None, use_fp16=False,
                 normalizer=None, device='cpu', distributed=False, lr_z=0.01,
                 warmup=10, reset_l0=0, reset_bn=0, bn_mmt=0,
                 is_maml=1):
        super(FastSynthesizer, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.bn_mmt = bn_mmt
        self.ismaml = is_maml

        self.num_classes = num_classes
        self.distributed = distributed
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.init_dataset = init_dataset
        self.use_fp16 = use_fp16
        self.autocast = autocast  # for FP16
        self.normalizer = normalizer
        self.data_pool = utils._utils.ImagePool(root=self.save_dir)
        self.transform = transform
        self.data_iter = None
        self.generator = generator.to(device).train()
        self.device = device
        self.hooks = []

        self.ep = 0
        self.ep_start = warmup
        self.reset_l0 = reset_l0
        self.reset_bn = reset_bn
        self.prev_z = None

        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))
        self.aug = transforms.Compose([
            augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
            augmentation.RandomHorizontalFlip(),
            normalizer,
        ])

    def synthesize(self, targets=None):

        start = time.time()

        self.ep += 1
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6

        # if self.reset_all:
        #     reset_model(self.generator)

        # inputs = torch.randn( size=(self.synthesis_batch_size, *self.img_size), device=self.device ).requires_grad_()
        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_()
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        else:
            targets = targets.sort()[0]  # sort for better visualization
        targets = targets.to(self.device)

        optimizer = torch.optim.Adam([
            {'params': self.generator.parameters()},
            {'params': [z], 'lr': self.lr_z}
        ], lr=self.lr_g, betas=[0.5, 0.999])

        for it in range(self.iterations):
            inputs = self.generator(z)
            inputs_aug = self.aug(inputs)  # crop and normalize

            #############################################
            # Inversion Loss
            #############################################
            t_out = self.teacher(inputs_aug)
            if targets is None:
                targets = torch.argmax(t_out, dim=-1)
                targets = targets.to(self.device)

            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy(t_out, targets)
            if self.adv > 0 and (self.ep >= self.ep_start):
                s_out = self.student(inputs_aug)
                mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                loss_adv = -(kldiv_m(s_out, t_out, reduction='none').sum(
                    1) * mask).mean()  # decision adversarial distillation
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv

            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        self.student.train()
        self.prev_z = (z, targets)
        end = time.time()

        self.data_pool.add(best_inputs)
        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.init_dataset is not None:
            init_dst = datafree.utils.UnlabeledImageDataset(self.init_dataset, transform=self.transform)
            dst = torch.utils.data.ConcatDataset([dst, init_dst])
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = utils._utils.DataIter(loader)
        return {"synthetic": best_inputs}, end - start

    def sample(self):
        return self.data_iter.next()

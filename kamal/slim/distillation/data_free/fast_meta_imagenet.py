
from kornia.geometry.transform.affwarp import scale
from typing import Generator
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
from kamal import utils
from .base import BaseSynthesis
from kamal.core import tasks
from .hooks import DeepInversionHook, InstanceMeanHook
from .criterions import get_image_prior_losses, kldiv_m
# from data_free.utils import ImagePool, DataIter, clip_images
import collections
from torchvision import transforms
from kornia import augmentation
import time

def reptile_grad(src, tar):
    if isinstance(src, nn.Module):
        for p, tar_p in zip(src.parameters(), tar.parameters()):
            if p.grad is None:
                p.grad = Variable(torch.zeros(p.size())).cuda()
            p.grad.data.add_(p.data - tar_p.data) # , alpha=40
    else:
        if src.grad is None:
            src.grad = Variable(torch.zeros(src.size())).cuda()
        src.grad.data.add_(src.data - tar.data)

def fomaml_grad(src, tar):
    if isinstance(src, nn.Module):
        for p, tar_p in zip(src.parameters(), tar.parameters()):
            if p.grad is None:
                p.grad = Variable(torch.zeros(p.size())).cuda()
            p.grad.data.add_(tar_p.grad.data)   #, alpha=0.67
    else:
        if src.grad is None:
            src.grad = Variable(torch.zeros(src.size())).cuda()
        src.grad.data.add_(tar.data)

class FastMetaSynthesizerForImageNet(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size,
                 init_dataset=None, iterations=100, lr_g=1e-3, lr_z = 1e-3, lr_z_meta=1e-3, lr_g_meta=1e-3,
                 synthesis_batch_size=128, sample_batch_size=128, reinit=0,
                 adv=0.0, bn=1, oh=1, meta_lr_scale=1.0,
                 save_dir='run/fast', transform=None, autocast=None, use_fp16=False,
                 normalizer=None, device='cpu', distributed=False, 
                 start_kd=10, rand_label=1, reset_l0=0, reset_bn=0, bn_mmt=0,
                 lr_diff=0, is_maml=1):
        super(FastMetaSynthesizerForImageNet, self).__init__(teacher, student)
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
        self.reinit = reinit
        self.lr_diff = lr_diff
        self.ismaml = is_maml
        self.meta_lr_scale = meta_lr_scale
        self.num_classes = num_classes
        self.distributed = distributed
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.init_dataset = init_dataset
        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.normalizer = normalizer
        self.data_pool = utils._utils.ImagePool(root=self.save_dir)
        self.transform = transform
        self.data_iter = None
        self.device = device
        self.hooks = []
        self.ep = 0
        self.ep_start = start_kd
        self.rand_label = rand_label
        self.reset_l0 = reset_l0
        self.reset_bn = reset_bn
        self.lr_z_meta = lr_z_meta
        self.lr_g_meta = lr_g_meta 
        self.generator = generator.to(device).train()
        self.z = torch.randn(size=(self.synthesis_batch_size, self.nz, self.img_size[-2]//16, self.img_size[-1]//16), device=self.device).requires_grad_() 
        self.meta_optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, 
                        {'params': [self.z], 'lr': lr_z_meta}], lr=lr_g_meta, betas=[0.5, 0.999])

        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m, self.bn_mmt) )

        self.aug = transforms.Compose([ 
                #augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=32, padding_mode='reflect'),
                augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.75, 1.0]),
                #augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]]),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ])
        print(self.aug, num_classes)

    def synthesize(self, targets=None):
        start = time.time()
        self.ep+=1
        self.student.eval()
        self.teacher.eval()
        self.generator.train()
        if self.reinit>0 and self.ep%self.reinit==0:
            print('reinit!')
            #self.generator = self.generator.clone(copy_params=False).to(self.device).train()
            self.z = torch.randn(size=(self.synthesis_batch_size, self.nz, self.img_size[-2]//16, self.img_size[-1]//16), device=self.device).requires_grad_() 
            self.meta_optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, 
                            {'params': [self.z], 'lr': self.lr_z_meta}], lr=self.lr_g_meta, betas=[0.5, 0.999])
        z = self.z.detach().requires_grad_() #  torch.randn(size=(self.synthesis_batch_size, self.nz, self.img_size[-2]//16, self.img_size[-1]//16), device=self.device).requires_grad_()  # 
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        #targets = targets.sort()[0] # sort for better visualization
        targets = targets.to(self.device)
        fast_generator = self.generator.clone().train()
        optimizer = torch.optim.Adam([{'params': fast_generator.parameters()}, 
                                      {'params': [z], 'lr': self.lr_z}], lr=self.lr_g, betas=[0.5, 0.999])       
        #print(targets)  
        for it in range(self.iterations):
            inputs = fast_generator(z)
            inputs_aug = self.aug(inputs) # crop and normalize
            if it == 0:
                originalMeta = inputs
            #############################################
            # Inversion Loss
            #############################################
            t_out = self.teacher(inputs_aug)
            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy( t_out, targets )
            if self.adv>0:
                s_out = self.student(inputs_aug)
                mask = (s_out.max(1)[1]==t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean() # decision adversarial distillation
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv #+ 1e-4 * loss_prior
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if it%10==0:
            #    datafree.utils.save_image_batch( inputs, 'checkpoints/vis/%d.png'%(it) )
            #    print(loss_bn.item(), loss_oh.item())
        
        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        # perform REPTILE or FOMAML
        self.meta_optimizer.zero_grad()
        if self.ismaml:
            fomaml_grad(self.generator, fast_generator)
            fomaml_grad(self.z, z)
        else:
            reptile_grad(self.generator, fast_generator)
            reptile_grad(self.z, z)
        self.meta_optimizer.step()

        self.student.train()
        end = time.time()
        if self.data_iter:
            del self.data_iter
        self.data_pool.add( inputs )
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
            num_workers=4, pin_memory=False, sampler=train_sampler)
        self.data_iter = utils._utils.DataIter(loader)
        return {"synthetic": inputs, "meta": originalMeta}, {'loss_bn': loss_bn.item(), 'loss_oh': loss_oh.item()}, end - start
        
    def sample(self):
        return self.data_iter.next()
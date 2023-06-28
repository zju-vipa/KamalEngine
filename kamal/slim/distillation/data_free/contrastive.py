from typing import Generator
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import random


from .base import BaseSynthesis
from .hooks import DeepInversionHook, InstanceMeanHook
from .criterions import jsdiv, get_image_prior_losses, kldiv
from kamal import utils
# from utils import ImagePool, DataIter, clip_images, UnlabeledImageDataset
import collections
from torchvision import transforms
from kornia import augmentation

class MLPHead(nn.Module):
    def __init__(self, dim_in, dim_feat, dim_h=None):
        super(MLPHead, self).__init__()
        if dim_h is None:
            dim_h = dim_in

        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_h),
            nn.ReLU(inplace=True),
            nn.Linear(dim_h, dim_feat),
        )

    def forward(self, x):
        x = self.head(x)
        return F.normalize(x, dim=1, p=2)

class MultiTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str( self.transform )


class ContrastLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR.
    Adapted from https://github.com/HobbitLong/SupContrast/blob/master/losses.py"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ContrastLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, return_logits=False):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        if return_logits:
            return loss, anchor_dot_contrast
        return loss


class MemoryBank(object):
    def __init__(self, device, max_size=4096, dim_feat=512):
        self.device = device
        self.data = torch.randn( max_size, dim_feat ).to(device)
        self._ptr = 0
        self.n_updates = 0

        self.max_size = max_size
        self.dim_feat = dim_feat

    def add(self, feat):
        feat = feat.to(self.device)
        n, c = feat.shape
        assert self.dim_feat==c and self.max_size % n==0, "%d, %d"%(self.dim_feat, c, self.max_size, n)
        self.data[self._ptr:self._ptr+n] = feat.detach()
        self._ptr = (self._ptr+n) % (self.max_size)
        self.n_updates+=n

    def get_data(self, k=None, index=None):
        if k is None:
            k = self.max_size

        if self.n_updates>self.max_size:
            if index is None:
                index = random.sample(list(range(self.max_size)), k=k)
            return self.data[index], index
        else:
            #return self.data[:self._ptr]
            if index is None:
                index = random.sample(list(range(self._ptr)), k=min(k, self._ptr))
            return self.data[index], index

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class CMISynthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size, 
                 feature_layers=None, bank_size=40960, n_neg=4096, head_dim=128, init_dataset=None,
                 iterations=100, lr_g=0.1, progressive_scale=False,
                 synthesis_batch_size=128, sample_batch_size=128, 
                 adv=0.0, bn=1, oh=1, cr=0.8, cr_T=0.1,
                 save_dir='run/cmi', transform=None,
                 autocast=None, use_fp16=False, 
                 normalizer=None, device='cpu', distributed=False):
        super(CMISynthesizer, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.progressive_scale = progressive_scale
        self.nz = nz
        self.n_neg = n_neg
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.num_classes = num_classes
        self.distributed = distributed
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.bank_size = bank_size
        self.init_dataset = init_dataset

        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.data_iter = None

        self.cr = cr
        self.cr_T = cr_T
        self.cmi_hooks = []
        if feature_layers is not None:
            for layer in feature_layers:
                self.cmi_hooks.append( InstanceMeanHook(layer) )
        else:
            for m in teacher.modules():
                if isinstance(m, nn.BatchNorm2d):
                    self.cmi_hooks.append( InstanceMeanHook(m) )

        with torch.no_grad():
            teacher.eval()
            fake_inputs = torch.randn(size=(1, *img_size), device=device)
            _ = teacher(fake_inputs)
            cmi_feature = torch.cat([ h.instance_mean for h in self.cmi_hooks ], dim=1)
            print("CMI dims: %d"%(cmi_feature.shape[1]))
            del fake_inputs
        
        self.generator = generator.to(device).train()
        # local and global bank
        self.mem_bank = MemoryBank('cpu', max_size=self.bank_size, dim_feat=2*cmi_feature.shape[1]) # local + global
        
        self.head = MLPHead(cmi_feature.shape[1], head_dim).to(device).train()
        self.optimizer_head = torch.optim.Adam(self.head.parameters(), lr=self.lr_g)

        self.device = device
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m) )

        self.aug = MultiTransform([
            # global view
            transforms.Compose([ 
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ]),
            # local view
            transforms.Compose([
                augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.25, 1.0]),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ]),
        ])

        #self.contrast_loss = ContrastLoss(temperature=self.cr_T, contrast_mode='one')

    def synthesize(self, targets=None):
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        
        #inputs = torch.randn( size=(self.synthesis_batch_size, *self.img_size), device=self.device ).requires_grad_()
        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_() 
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
            targets = targets.sort()[0] # sort for better visualization
        targets = targets.to(self.device)

        reset_model(self.generator)
        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g, betas=[0.5, 0.999])
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iterations, eta_min=0.1*self.lr)
        for it in range(self.iterations):
            inputs = self.generator(z)
            global_view, local_view = self.aug(inputs) # crop and normalize

            #############################################
            # Inversion Loss
            #############################################
            t_out = self.teacher(global_view)
            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy( t_out, targets )
            if self.adv>0:
                s_out = self.student(global_view)
                mask = (s_out.max(1)[1]==t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean() # decision adversarial distillation
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss_inv = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv
            
            #############################################
            # Contrastive Loss
            #############################################
            global_feature = torch.cat([ h.instance_mean for h in self.cmi_hooks ], dim=1) 
            _ = self.teacher(local_view)
            local_feature = torch.cat([ h.instance_mean for h in self.cmi_hooks ], dim=1) 
            cached_feature, _ = self.mem_bank.get_data(self.n_neg)
            cached_local_feature, cached_global_feature = torch.chunk(cached_feature.to(self.device), chunks=2, dim=1)

            proj_feature = self.head( torch.cat([local_feature, cached_local_feature, global_feature, cached_global_feature], dim=0) )
            proj_local_feature, proj_global_feature = torch.chunk(proj_feature, chunks=2, dim=0)
            
            # https://github.com/HobbitLong/SupContrast/blob/master/losses.py
            #cr_feature = torch.cat( [proj_local_feature.unsqueeze(1), proj_global_feature.unsqueeze(1).detach()], dim=1 )
            #loss_cr = self.contrast_loss(cr_feature)
            
            # Note that the cross entropy loss will be divided by the total batch size (current batch + cached batch)
            # we split the cross entropy loss to avoid too small gradients w.r.t the generator
            #if self.mem_bank.n_updates>0:
                          # 1. gradient from current batch              +  2. gradient from cached data
            #    loss_cr = loss_cr[:, :self.synthesis_batch_size].mean() + loss_cr[:, self.synthesis_batch_size:].mean()
            #else: # 1. gradients only come from current batch      
            #    loss_cr = loss_cr.mean()

            # A naive implementation of contrastive loss
            cr_logits = torch.mm(proj_local_feature, proj_global_feature.detach().T) / self.cr_T # (N + N') x (N + N')
            cr_labels = torch.arange(start=0, end=len(cr_logits), device=self.device)
            loss_cr = F.cross_entropy( cr_logits, cr_labels, reduction='none')  #(N + N')
            if self.mem_bank.n_updates>0:
                loss_cr = loss_cr[:self.synthesis_batch_size].mean() + loss_cr[self.synthesis_batch_size:].mean()
            else:
                loss_cr = loss_cr.mean()
            
            loss = self.cr * loss_cr + loss_inv
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data
                    best_features = torch.cat([local_feature.data, global_feature.data], dim=1).data
            optimizer.zero_grad()
            self.optimizer_head.zero_grad()
            loss.backward()
            optimizer.step()
            self.optimizer_head.step()

        self.student.train()
        # save best inputs and reset data iter
        self.data_pool.add( best_inputs )
        self.mem_bank.add( best_features )

        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.init_dataset is not None:
            init_dst = UnlabeledImageDataset(self.init_dataset, transform=self.transform)
            dst = torch.utils.data.ConcatDataset([dst, init_dst])
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)
        return {"synthetic": best_inputs}
        
    def sample(self):
        return self.data_iter.next()

from kamal.core.engine.engine import Engine
from kamal.core.engine.hooks import FeatureHook
from kamal.core import tasks
from kamal.utils import set_mode, move_to_device

import torch
import torch.nn as nn
import torch.nn.functional as F

import typing, time
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResBlock(nn.Module):
    """ Residual Blocks
    """
    def __init__(self, inplanes, planes, stride=1, momentum=0.1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=momentum)
        if stride > 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes, momentum=momentum)
            )
        else:
            self.downsample = None

        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class CFL_FCBlock(nn.Module):
    """Common Feature Blocks for Fully-Connected layer
    
    This module is used to capture the common features of multiple teachers and calculate mmd with features of student.

    **Parameters:**
        - cs (int): channel number of student features
        - channel_ts (list or tuple): channel number list of teacher features
        - ch (int): channel number of hidden features
    """
    def __init__(self, cs, cts, ch, k_size=5):
        super(CFL_FCBlock, self).__init__()
        
        self.align_t = nn.ModuleList()
        for ct in cts:
            self.align_t.append(
                nn.Sequential(
                    nn.Linear(ct, ch),
                    nn.ReLU(inplace=True)
                )
            )

        self.align_s = nn.Sequential(
            nn.Linear(cs, ch),
            nn.ReLU(inplace=True),
        )

        self.extractor = nn.Sequential(
            nn.Linear(ch, ch),
            nn.ReLU(),
            nn.Linear(ch, ch),
        )

        self.dec_t = nn.ModuleList()
        for ct in cts:
            self.dec_t.append(
                nn.Sequential(
                    nn.Linear(ch, ct),
                    nn.ReLU(inplace=True),
                    nn.Linear(ct, ct)
                )
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, fs, fts):
        aligned_t = [self.align_t[i](fts[i]) for i in range(len(fts))]
        aligned_s = self.align_s(fs)

        hts = [self.extractor(f) for f in aligned_t]
        hs = self.extractor(aligned_s)

        _fts = [self.dec_t[i](hts[i]) for i in range(len(hts))]
        return (hs, hts), (_fts, fts)


class CFL_ConvBlock(nn.Module):
    """Common Feature Blocks for Convolutional layer
    
    This module is used to capture the common features of multiple teachers and calculate mmd with features of student.

    **Parameters:**
        - cs (int): channel number of student features
        - channel_ts (list or tuple): channel number list of teacher features
        - ch (int): channel number of hidden features
    """
    def __init__(self, cs, cts, ch, k_size=5):
        super(CFL_ConvBlock, self).__init__()
        
        self.align_t = nn.ModuleList()
        for ct in cts:
            self.align_t.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=ct, out_channels=ch,
                              kernel_size=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True)
                )
            )

        self.align_s = nn.Sequential(
            nn.Conv2d(in_channels=cs, out_channels=ch,
                      kernel_size=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        self.extractor = nn.Sequential(
            ResBlock(inplanes=ch, planes=ch, stride=1),
            ResBlock(inplanes=ch, planes=ch, stride=1),
        )

        self.dec_t = nn.ModuleList()
        for ct in cts:
            self.dec_t.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch, kernel_size=1, stride=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch, ct, kernel_size=1, stride=1)
                )
            )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, fs, fts):
        aligned_t = [self.align_t[i](fts[i]) for i in range(len(fts))]
        aligned_s = self.align_s(fs)

        hts = [self.extractor(f) for f in aligned_t]
        hs = self.extractor(aligned_s)

        _fts = [self.dec_t[i](hts[i]) for i in range(len(hts))]
        return (hs, hts), (_fts, fts)

class CommonFeatureAmalgamator(Engine):
    
    def setup(
        self, 
        student,
        teachers,
        layer_groups: typing.Sequence[typing.Sequence],
        layer_channels: typing.Sequence[typing.Sequence],
        dataloader:  torch.utils.data.DataLoader, 
        optimizer:    torch.optim.Optimizer, 
        weights = [1.0, 1.0, 1.0],
        on_layer_input=False,
        device = None,
    ):
        self._dataloader = dataloader
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self._device = device

        self._model = self._student = student.to(self._device)
        self._teachers = nn.ModuleList(teachers).to(self._device) 
        self._optimizer = optimizer
        self._weights = weights
        self._on_layer_input = on_layer_input

        amal_blocks = []
        for group, C in zip( layer_groups, layer_channels ):
            hooks = [ FeatureHook(layer) for layer in group ]
            if isinstance(group[0], nn.Linear):
                amal_block = CFL_FCBlock( cs=C[0], cts=C[1:], ch=C[0]//4 ).to(self._device).train()
                print("Building FC Blocks")
            else:
                amal_block = CFL_ConvBlock(cs=C[0], cts=C[1:], ch=C[0]//4).to(self._device).train()
                print("Building Conv Blocks")
            amal_blocks.append( (amal_block, hooks, C)  )
        self._amal_blocks = amal_blocks
        self._cfl_criterion = tasks.loss.CFLLoss( sigmas=[0.001, 0.01, 0.05, 0.1, 0.2, 1, 2] )

    def run(self, max_iter, start_iter=0, epoch_length=None):
        block_params = []
        for block, _, _ in self._amal_blocks:
            block_params.extend( list(block.parameters()) )
        if isinstance( self._optimizer, torch.optim.SGD ):
            self._amal_optimimizer = torch.optim.SGD( block_params, lr=self._optimizer.param_groups[0]['lr'], momentum=0.9, weight_decay=1e-4 )
        else:
            self._amal_optimimizer = torch.optim.Adam( block_params, lr=self._optimizer.param_groups[0]['lr'], weight_decay=1e-4 )
        self._amal_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self._amal_optimimizer, T_max=max_iter )
        
        with set_mode(self._student, training=True), \
             set_mode(self._teachers, training=False):
            super( CommonFeatureAmalgamator, self ).run(self.step_fn, self._dataloader, start_iter=start_iter, max_iter=max_iter, epoch_length=epoch_length)

    def step_fn(self, engine, batch):
        start_time = time.perf_counter()
        batch = move_to_device(batch, self._device)
        data = batch[0]
        s_out = self._student( data )
        with torch.no_grad():
            t_out = [ teacher( data ) for teacher in self._teachers ]
        loss_amal = 0
        loss_recons = 0
        for amal_block, hooks, C in self._amal_blocks:
            features = [ h.feat_in if self._on_layer_input else h.feat_out for h in hooks ]
            fs, fts = features[0], features[1:]
            (hs, hts), (_fts, fts) = amal_block( fs, fts )
            _loss_amal, _loss_recons = self._cfl_criterion( hs, hts, _fts, fts ) 
            loss_amal += _loss_amal
            loss_recons += _loss_recons
        loss_kd = tasks.loss.kldiv( s_out, torch.cat( t_out, dim=1 ) )
        loss_dict = { 
                'loss_kd':     self._weights[0]*loss_kd,
                'loss_amal':   self._weights[1]*loss_amal,
                'loss_recons': self._weights[2]*loss_recons
        }
        loss = sum(loss_dict.values())
        self._optimizer.zero_grad()
        self._amal_optimimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._amal_optimimizer.step()
        self._amal_scheduler.step()
        step_time = time.perf_counter() - start_time

        metrics = { loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'total_loss': loss.item(),
            'step_time': step_time,
            'lr': float( self._optimizer.param_groups[0]['lr'] )
        })
        return metrics

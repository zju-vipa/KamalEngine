
from kamal.core import engine, criterion, metrics
from kamal.utils import set_mode

import torch
import torch.nn as nn
import torch.nn.functional as F

import typing, time
import numpy as np
from torchvision.models.resnet import BasicBlock


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
        #self.bn1 = nn.BatchNorm2d(planes, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes, momentum=momentum)
        if stride > 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                #nn.BatchNorm2d(planes, momentum=momentum)
            )
        else:
            self.downsample = None

        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        #out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class CFLBlock(nn.Module):
    """Common Feature Blocks for Convolutional layer
    
    This module is used to capture the common features of multiple teachers and calculate mmd with features of student.

    **Parameters:**
        - cs (int): channel number of student features
        - channel_ts (list or tuple): channel number list of teacher features
        - ch (int): channel number of hidden features
    """
    def __init__(self, cs, cts, ch, k_size=5):
        super(CFLBlock, self).__init__()
        
        self.align_t = nn.ModuleList()
        for ct in cts:
            self.align_t.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=ct, out_channels=2*ch,
                              kernel_size=1, bias=False),
                    nn.ReLU(inplace=True)
                )
            )

        self.align_s = nn.Sequential(
            nn.Conv2d(in_channels=cs, out_channels=2*ch,
                      kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.extractor = nn.Sequential(
            ResBlock(inplanes=2*ch, planes=ch, stride=1),
            ResBlock(inplanes=ch, planes=ch, stride=1),
            ResBlock(inplanes=ch, planes=ch, stride=1),
        )

        self.dec_t = nn.ModuleList()
        for ct in cts:
            self.dec_t.append(
                nn.Sequential(
                    nn.Conv2d(ch, ct, kernel_size=3,
                              stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ct, ct, kernel_size=1,
                              stride=1, padding=0, bias=False)
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


class CommonFeatureAmalgamator(engine.trainer.TrainerBase):
    
    def setup(
        self, 
        student,
        teachers,
        layer_groups: typing.Sequence[typing.Sequence],
        layer_channels: typing.Sequence[typing.Sequence],
        data_loader:  torch.utils.data.DataLoader, 
        optimizer:    torch.optim.Optimizer, 
        weights = [1.0, 1.0, 1.0],
        device = None,
    ):
        self._data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self.device = device
        self.model = self.student = student.to(self.device)
        self.teachers = nn.ModuleList(teachers).to(self.device) 
        self.optimizer = optimizer
        self._weights = weights

        amal_blocks = []
        block_params = []
        for group, C in zip( layer_groups, layer_channels ):
            hooks = [ engine.hooks.FeatureHook(layer) for layer in group ]
            amal_block = CFLBlock(cs=C[0], cts=C[1:], ch=256).to(self.device).train()
            amal_blocks.append( (amal_block, hooks, C)  )
            block_params.extend( list(amal_block.parameters()) )
        self._amal_blocks = amal_blocks
        self._amal_optimimizer = torch.optim.Adam( block_params, lr=1e-3, weight_decay=1e-4 )
        self._cfl_criterion = criterion.CFLLoss( sigmas=[0.001, 0.01, 0.05, 0.1, 0.2, 1, 2] )

    def run(self, start_iter, max_iter):
        with set_mode(self.student, training=True), \
             set_mode(self.teachers, training=False):
            super( CommonFeatureAmalgamator, self ).run( start_iter, max_iter)

    def _get_data(self):
        try:
            data = next( self._data_loader_iter )
        except StopIteration:
            self._data_loader_iter = iter(self._data_loader) # reset iterator
            data = next( self._data_loader_iter )
        if not isinstance( data, typing.Sequence ):
            data = [data, ]
        return data

    def step(self):
        self.optimizer.zero_grad()
        self._amal_optimimizer.zero_grad()

        start_time = time.perf_counter()

        data = self._get_data()
        data = [ d.to(self.device) for d in data ] # move to device

        s_out = self.student( data[0] )
        with torch.no_grad():
            t_out = [ teacher( data[0] ) for teacher in self.teachers ]
        
        loss_amal = 0
        loss_recons = 0
        for amal_block, hooks, C in self._amal_blocks:
            features = [ h.feat_out for h in hooks ]
            fs, fts = features[0], features[1:]
            (hs, hts), (_fts, fts) = amal_block( fs, fts )
            _loss_amal, _loss_recons = self._cfl_criterion( hs, hts, _fts, fts ) 
            loss_amal += _loss_amal
            loss_recons += _loss_recons
        loss_kd = criterion.kldiv( s_out, torch.cat( t_out, dim=1 ) )
        loss = self._weights[0]*loss_kd + self._weights[1]*loss_amal + self._weights[2]*loss_recons
        loss.backward()
        self.optimizer.step()
        self._amal_optimimizer.step()
        step_time = time.perf_counter() - start_time

        # record training info
        info = {
            'total_loss': loss.item(),
            'loss_kd': loss_kd.item(),
            'loss_amal': loss_amal.item(),
            'loss_recons': loss_amal.item(),

            'step_time': step_time,
            'lr': float( self.optimizer.param_groups[0]['lr'] )
        }
        self.history.put_scalars( **info )

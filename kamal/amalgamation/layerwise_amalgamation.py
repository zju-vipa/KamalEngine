import torch
import torch.nn as nn
import torch.nn.functional as F

from kamal.core.engine.engine import Engine
from kamal.core.engine.hooks import FeatureHook
from kamal.core import tasks

from kamal.utils import set_mode
import typing
import time
from kamal.utils import move_to_device, set_mode

class AmalBlock(nn.Module):
    def __init__(self, cs, cts):
        super( AmalBlock, self ).__init__()
        self.cs, self.cts = cs, cts
        self.enc = nn.Conv2d( in_channels=sum(self.cts), out_channels=self.cs, kernel_size=1, stride=1, padding=0, bias=True )
        self.fam = nn.Conv2d( in_channels=self.cs, out_channels=self.cs, kernel_size=1, stride=1, padding=0, bias=True )
        self.dec = nn.Conv2d( in_channels=self.cs, out_channels=sum(self.cts), kernel_size=1, stride=1, padding=0, bias=True )
    
    def forward(self, fs, fts):
        rep = self.enc( torch.cat( fts, dim=1 ) )
        _fts = self.dec( rep )
        _fts = torch.split( _fts, self.cts, dim=1 )
        _fs = self.fam( fs )
        return rep, _fs, _fts

class LayerWiseAmalgamator(Engine):
    
    def setup(
        self, 
        student,
        teachers,
        layer_groups: typing.Sequence[typing.Sequence],
        layer_channels: typing.Sequence[typing.Sequence],
        dataloader:  torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer, 
        weights = [1., 1., 1.],
        device=None,
    ):
        if device is None:
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        self._device = device
        self._dataloader = dataloader
        self.model = self.student = student.to(self.device)
        self.teachers = nn.ModuleList(teachers).to(self.device) 
        self.optimizer = optimizer
        self._weights = weights
        amal_blocks = []
        
        for group, C in zip(layer_groups, layer_channels):
            hooks = [ FeatureHook(layer) for layer in group ]
            amal_block = AmalBlock(cs=C[0], cts=C[1:]).to(self.device).train()
            amal_blocks.append( (amal_block, hooks, C)  )
        self._amal_blocks = amal_blocks

    def run(self, max_iter, start_iter=0, epoch_length=None ):
        block_params = []
        for block, _, _ in self._amal_blocks:
            block_params.extend( list(block.parameters()) )
        if isinstance( self.optimizer, torch.optim.SGD ):
            self._amal_optimimizer = torch.optim.SGD( block_params, lr=self.optimizer.param_groups[0]['lr'], momentum=0.9, weight_decay=1e-4 )
        else:
            self._amal_optimimizer = torch.optim.Adam( block_params, lr=self.optimizer.param_groups[0]['lr'], weight_decay=1e-4 )
        self._amal_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self._amal_optimimizer, T_max=max_iter )

        with set_mode(self.student, training=True), \
             set_mode(self.teachers, training=False):
            super( LayerWiseAmalgamator, self ).run(self.step_fn, self._dataloader, start_iter=start_iter, max_iter=max_iter, epoch_length=epoch_length)
    
    @property
    def device(self):
        return self._device
    
    def step_fn(self, engine, batch):
        start_time = time.perf_counter()
        batch = move_to_device(batch, self._device)
        data = batch[0]
        s_out = self.student( data )
        with torch.no_grad():
            t_out = [ teacher( data ) for teacher in self.teachers ]
        loss_amal = 0
        loss_recons = 0
        for amal_block, hooks, C in self._amal_blocks:
            features = [ h.feat_out for h in hooks ]
            fs, fts = features[0], features[1:]
            rep, _fs, _fts = amal_block( fs, fts )
            loss_amal += F.mse_loss( _fs, rep.detach() )
            loss_recons += sum( [ F.mse_loss( _ft, ft ) for (_ft, ft) in zip( _fts, fts ) ] )
        loss_kd = tasks.loss.kldiv( s_out, torch.cat( t_out, dim=1 ) )
        #loss_kd = F.mse_loss( s_out, torch.cat( t_out, dim=1 ) )
        loss_dict = { "loss_kd":      self._weights[0] * loss_kd,
                      "loss_amal":    self._weights[1] * loss_amal,
                      "loss_recons":  self._weights[2] * loss_recons }
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        self._amal_optimimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._amal_optimimizer.step()
        self._amal_scheduler.step()
        step_time = time.perf_counter() - start_time
        metrics = { loss_name: loss_value.item() for (loss_name, loss_value) in loss_dict.items() }
        metrics.update({
            'total_loss': loss.item(),
            'step_time': step_time,
            'lr': float( self.optimizer.param_groups[0]['lr'] )
        })
        return metrics



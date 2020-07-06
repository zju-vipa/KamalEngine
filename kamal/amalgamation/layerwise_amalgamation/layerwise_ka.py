import torch
import torch.nn as nn
import torch.nn.functional as F

from kamal.core import engine, criterions, metrics
from kamal.utils import set_mode
import typing
import time

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

class LayerWiseAmalgamator(engine.trainer.TrainerBase):
    
    def setup(
        self, 
        student,
        teachers,
        layer_groups: typing.Sequence[typing.Sequence],
        layer_channels: typing.Sequence[typing.Sequence],
        data_loader:  torch.utils.data.DataLoader, 
        optimizer:    torch.optim.Optimizer, 
        weights = [1., 1., 1.],
        device=None,
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
        for group, C in zip(layer_groups, layer_channels):
            hooks = [ engine.hooks.FeatureHook(layer) for layer in group ]
            amal_block = AmalBlock(cs=C[0], cts=C[1:]).to(self.device).train()
            amal_blocks.append( (amal_block, hooks, C)  )
            block_params.extend( list(amal_block.parameters()) )
        self._amal_blocks = amal_blocks
        self._amal_optimimizer = torch.optim.Adam( block_params, lr=1e-3 )

    def reset(self):
        self._amal_optimimizer = torch.optim.Adam( block_params, lr=1e-3 )
        if hasattr( self, '_amal_scheduler' ):
            self._amal_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self._amal_optimimizer, T_max=max_iter)
        self._iter = self._start_iter
        self._data_loader_iter = iter(data_loader)
    
    @property
    def data_loader(self):
        return self._data_loader

    def run(self, start_iter, max_iter):
        self._amal_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self._amal_optimimizer, T_max=max_iter)
        with set_mode(self.student, training=True), \
             set_mode(self.teachers, training=False):
            super( LayerWiseAmalgamator, self ).run( start_iter, max_iter)

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
            rep, _fs, _fts = amal_block( fs, fts )
            loss_amal += F.mse_loss( _fs, rep.detach() )
            loss_recons += sum( [ F.mse_loss( _ft, ft ) for (_ft, ft) in zip( _fts, fts ) ] )
        
        loss_kd = criterion.kldiv( s_out, torch.cat( t_out, dim=1 ) )
        #loss_kd = F.mse_loss( s_out, torch.cat( t_out, dim=1 ) )
        loss = self._weights[0]*loss_kd + self._weights[1]*loss_amal + self._weights[2]*loss_recons
        
        loss.backward()
        self.optimizer.step()
        self._amal_optimimizer.step()
        self._amal_scheduler.step()
        step_time = time.perf_counter() - start_time

        # record training info
        info = {
            'total_loss': loss.item(),
            'loss_kd': loss_kd.item(),
            'loss_amal': loss_amal.item(),
            'loss_recons': loss_recons.item(),

            'step_time': step_time,
            'lr': float( self.optimizer.param_groups[0]['lr'] )
        }
        self.history.put_scalars( **info )



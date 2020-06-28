import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskModel(nn.Module):
    def __init__(self, encoder, decoders):
        super( MultiTaskModel, self ).__init__()
        self.encoder = encoder
        self.decoders = nn.ModuleList( decoders )
    
    def forward( self, x ):
        rep = self.encoder( x )
        outputs = [ dec(rep) for dec in self.decoders ]
        return outputs

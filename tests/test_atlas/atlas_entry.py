import torch, os, sys
from atlas import meta
# User Module
import lenet

class AtlasEntry(meta.AtlasEntryBase):
    
    @staticmethod
    def init():
        model = lenet.LeNet5()
        return model
    



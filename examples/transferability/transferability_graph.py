from kamal.transferability.trans_graph import TransferabilityGraph
from kamal.transferability.trans_metric import AttrMapMetric

import kamal
from kamal.vision import sync_transforms as sT
import os
import torch
from PIL import Image

if __name__=='__main__':
    zoo = './zoo'
    TG = TransferabilityGraph(zoo)
    probe_set_root = './probe_data'
    for probe_set in os.listdir( probe_set_root ):
        print("Add %s"%(probe_set))
        imgs_set = list( os.listdir( os.path.join( probe_set_root, probe_set ) ) )
        images = [ Image.open( os.path.join(probe_set_root, probe_set, img) ) for img in imgs_set ]
        metric = AttrMapMetric(images, device='cpu')
        TG.add_metric( probe_set, metric)
        
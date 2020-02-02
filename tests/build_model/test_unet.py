from kamal.vision.models import segmentation
import torch

model = segmentation.unet().eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

model = segmentation.UNet(channel_list=[ 32, 64, 128, 256, 512 ]).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

from kamal.vision.models import segmentation
import torch

model = segmentation.linknet_resnet18(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

model = segmentation.linknet_resnet34(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

model = segmentation.linknet_resnet50(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

model = segmentation.linknet_resnet101(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

model = segmentation.linknet_resnet152(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )
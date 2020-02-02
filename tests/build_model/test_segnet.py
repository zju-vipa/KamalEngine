from kamal.vision.models import segmentation
import torch

# Customized model
model = segmentation.SegNet(arch=[1, 1, 1, 1, 1], pretrained_backbone=False).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

# SegNet without BN
model = segmentation.segnet_vgg11(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

model = segmentation.segnet_vgg13(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

model = segmentation.segnet_vgg16(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

model = segmentation.segnet_vgg19(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

# SegNet with BN
model = segmentation.segnet_vgg11_bn(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

model = segmentation.segnet_vgg13_bn(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

model = segmentation.segnet_vgg16_bn(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

model = segmentation.segnet_vgg19_bn(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

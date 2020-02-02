from kamal.vision.models import segmentation
import torch

# deeplab v3
model = segmentation.deeplabv3_mobilenetv2(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

model = segmentation.deeplabv3_resnet50(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

model = segmentation.deeplabv3_resnet101(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

# deeplab v3+
model = segmentation.deeplabv3plus_mobilenetv2(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

model = segmentation.deeplabv3plus_resnet50(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )

model = segmentation.deeplabv3plus_resnet50(pretrained_backbone=True).eval()
print( model( torch.randn( 1,3,256,256 ) ).shape )
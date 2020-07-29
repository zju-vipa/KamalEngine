import thop, torch
from torchvision.models import resnet34, resnet18

totoal_macs = 0
print(sum( [196, 102]) )
for num_classes in [ 196, 102]:
    res18 = resnet18(num_classes=num_classes)
    macs, params = thop.profile(res18, inputs=(torch.randn(1, 3, 224, 224), ))
    totoal_macs+=macs
    print(macs)
print("total macs=%f G"% (totoal_macs/1e9) )


res18 = resnet18(num_classes=sum([ 196, 102, 120, 200 ]))
macs, params = thop.profile(res18, inputs=(torch.randn(1, 3, 224, 224), ))
print(macs/1e9)

from kamal.vision.models.classification import darknet
from kamal.core.engine.evaluator import ClassificationEvaluator
from kamal.vision import datasets, sync_transforms as transforms
import torch


darknet19 = darknet.darknet19(pretrained=True).eval()
print(darknet19( torch.randn( 1,3,224,224 ) ).shape)

darknet19_448 = darknet.darknet19_448(pretrained=True).eval()
print(darknet19_448( torch.randn( 1,3,224,224 ) ).shape)
del darknet19_448

darknet53 = darknet.darknet53(pretrained=True).eval()
print(darknet53( torch.randn( 1,3,224,224 ) ).shape)

darknet53_448 = darknet.darknet53_448(pretrained=True).eval()
print(darknet53_448( torch.randn( 1,3,224,224 ) ).shape)
del darknet53_448

val_loader = torch.utils.data.DataLoader(
                datasets.ImageNet(
                            '~/Datasets/ILSVRC2012', 
                            split='val', 
                            download=True, 
                            transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(256),
                                transforms.ToTensor()
                            ]))
                ,batch_size=64, num_workers=2, shuffle=False, pin_memory=True
            )

print(darknet19)
evaluator = ClassificationEvaluator( data_loader = val_loader )
results = evaluator.eval( darknet19 ) 
print(results)

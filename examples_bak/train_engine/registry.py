import torch
from kamal.vision import datasets as kamaldatasets
from torchvision import datasets as torchdatasets

from torchvision import datasets, transforms as T
from kamal.vision import sync_transforms as sT

from kamal.vision import models as kamalmodels

from PIL import Image

_MODEL_DICT = {
    'deeplabv3_resnet50': kamalmodels.segmentation.deeplabv3_resnet50,
    'deeplabv3_mobilenet': kamalmodels.segmentation.deeplabv3plus_mobilenetv2,
    'deeplabv3plus_resnet50': kamalmodels.segmentation.deeplabv3plus_resnet50,
    'deeplabv3plus_mobilenet': kamalmodels.segmentation.deeplabv3plus_mobilenetv2,

    'segnet_vgg11': kamalmodels.segmentation.segnet_vgg11,
    'segnet_vgg13': kamalmodels.segmentation.segnet_vgg13,
    'segnet_vgg16': kamalmodels.segmentation.segnet_vgg16,
    'segnet_vgg19': kamalmodels.segmentation.segnet_vgg19,
    'segnet_vgg11_bn': kamalmodels.segmentation.segnet_vgg11_bn,
    'segnet_vgg13_bn': kamalmodels.segmentation.segnet_vgg13_bn,
    'segnet_vgg16_bn': kamalmodels.segmentation.segnet_vgg16_bn,
    'segnet_vgg19_bn': kamalmodels.segmentation.segnet_vgg19_bn,

    'unet': kamalmodels.segmentation.UNet,
    
    'linknet_resnet18': kamalmodels.segmentation.linknet.linknet_resnet18,
    'linknet_resnet34': kamalmodels.segmentation.linknet.linknet_resnet34,
    'linknet_resnet50': kamalmodels.segmentation.linknet.linknet_resnet50,
}

def get_model(model_name, num_classes):
    if 'unet' not in model_name:
        return _MODEL_DICT[model_name](num_classes=num_classes, pretrained_backbone=True)
    else:
        return _MODEL_DICT[model_name](num_classes=num_classes)

def get_dataloader(name, data_cfg):
    data_root = data_cfg['data_root']
    num_classes = data_cfg['num_classes']

    if name.lower()=='mnist':
        train_loader = torch.utils.data.DataLoader( 
            torchdatasets.MNIST(data_root, train=True, download=True,
                       transform=T.Compose([
                           T.Resize((32, 32)),
                           T.ToTensor(),
                           T.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=128, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader( 
            torchdatasets.MNIST(data_root, train=False, download=True,
                      transform=T.Compose([
                          T.Resize((32, 32)),
                          T.ToTensor(),
                          T.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=128, shuffle=False, num_workers=2)
    elif name.lower()=='cifar10':
        train_loader = torch.utils.data.DataLoader( 
            torchdatasets.CIFAR10(data_root, train=True, download=True,
                       transform=T.Compose([
                            T.RandomCrop(32, padding=4),
                            T.RandomHorizontalFlip(),
                            T.ToTensor(),
                            T.Normalize(data_cfg['mean'], data_cfg['std']),
                        ])),
            batch_size=128, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader( 
            torchdatasets.CIFAR10(data_root, train=False, download=True,
                       transform=T.Compose([
                            T.ToTensor(),
                            T.Normalize(data_cfg['mean'], data_cfg['std']),
                        ])),
            batch_size=128, shuffle=False, num_workers=2)
    elif name.lower()=='cifar100':
        train_loader = torch.utils.data.DataLoader( 
            torchdatasets.CIFAR100(data_root, train=True, download=True,
                       transform=T.Compose([
                            T.RandomCrop(32, padding=4),
                            T.RandomHorizontalFlip(),
                            T.ToTensor(),
                            T.Normalize(data_cfg['mean'], data_cfg['std']),
                        ])),
            batch_size=128, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader( 
            torchdatasets.CIFAR100(data_root, train=False, download=True,
                       transform=T.Compose([
                            T.ToTensor(),
                            T.Normalize(data_cfg['mean'], data_cfg['std']),
                        ])),
            batch_size=128, shuffle=False, num_workers=2)
    elif name.lower()=='caltech101':
        train_loader = torch.utils.data.DataLoader(
            Caltech101(data_root, train=True, download=False,
                        transform=T.Compose([
                            T.RandomResizedCrop(128),
                            T.RandomHorizontalFlip(),
                            T.ToTensor(),
                            T.Normalize(data_cfg['mean'], data_cfg['std'])
                        ])),
            batch_size=128, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            Caltech101(data_root, train=False, download=False, 
                        transform=transforms.Compose([
                            transforms.Resize(128),
                            transforms.CenterCrop(128),
                            transforms.ToTensor(),
                            transforms.Normalize(data_cfg['mean'], data_cfg['std'])
                        ])), 
            batch_size=128, shuffle=False, num_workers=2)
    elif name.lower()=='imagenet':
        train_loader = None # not required
        test_loader = torch.utils.data.DataLoader( 
            torchdatasets.ImageNet(data_root, split='val', download=True,
                      transform=T.Compose([
                            T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                        ])),
            batch_size=64, shuffle=False, num_workers=4) # shuffle for visualization
    
    ############ Segmentation       
    elif name.lower()=='camvid':
        train_loader = torch.utils.data.DataLoader(
            kamaldatasets.CamVid(data_root, split='trainval',
                        transforms=sT.Compose([
                            sT.Multi( sT.Resize(data_cfg['input_size']), 
                                sT.Resize(data_cfg['input_size'], interpolation=Image.NEAREST) ),
                            sT.Sync( sT.RandomCrop(data_cfg['crop_size'], pad_if_needed=True) ),
                            sT.Sync( sT.RandomHorizontalFlip() ),
                            sT.Multi( sT.ColorJitter(0.5, 0.5, 0.5), None  ),
                            sT.Multi( sT.ToTensor(), sT.ToTensor(normalize=False, dtype=torch.long) ),
                            sT.Multi( sT.Normalize( data_cfg['mean'], data_cfg['std']), None )
                        ])),
            batch_size=data_cfg['batch_size'], shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            kamaldatasets.CamVid(data_root, split='test',
                        transforms=sT.Compose([
                            sT.Multi( sT.Resize(data_cfg['input_size']), 
                                sT.Resize(data_cfg['input_size'], interpolation=Image.NEAREST) ),
                            sT.Multi( sT.ToTensor(), sT.ToTensor(normalize=False, dtype=torch.long) ),
                            sT.Multi( sT.Normalize( data_cfg['mean'], data_cfg['std']), None)
                        ])),
            batch_size=data_cfg['batch_size'], shuffle=False, num_workers=2)
    elif name in ['nyuv2']:
        train_loader = torch.utils.data.DataLoader(
            kamaldatasets.NYUv2(data_root, split='train',
                        transforms=sT.Compose([
                            sT.Multi( sT.Resize(data_cfg['input_size']), 
                                sT.Resize(data_cfg['input_size'], interpolation=Image.NEAREST) ),
                            sT.Sync( sT.RandomCrop(data_cfg['crop_size'], pad_if_needed=True) ),
                            sT.Sync( sT.RandomHorizontalFlip() ),
                            sT.Multi( sT.ColorJitter(0.5, 0.5, 0.5), None  ),
                            sT.Multi( sT.ToTensor(), sT.ToTensor(normalize=False, dtype=torch.long) ),
                            sT.Multi( sT.Normalize( data_cfg['mean'], data_cfg['std'] ), None )
                        ])),
            batch_size=data_cfg['batch_size'], shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            kamaldatasets.NYUv2(data_root, split='test',
                        transforms=sT.Compose([
                            sT.Multi( sT.Resize(data_cfg['input_size']), 
                                sT.Resize(data_cfg['input_size'], interpolation=Image.NEAREST) ),
                            sT.Multi( sT.ToTensor(), sT.ToTensor(normalize=False, dtype=torch.long) ),
                            sT.Multi( sT.Normalize( data_cfg['mean'], data_cfg['std']), None )
                        ])),
            batch_size=data_cfg['batch_size'], shuffle=False, num_workers=2)
    elif name.lower() in ['cityscapes']:
        train_loader = torch.utils.data.DataLoader(
            kamaldatasets.Cityscapes(data_root, split='train',
                        transforms=sT.Compose([
                            sT.Multi( sT.Resize(data_cfg['input_size']), 
                                sT.Resize(data_cfg['input_size'], interpolation=Image.NEAREST) ),
                            sT.Sync( sT.RandomCrop(data_cfg['crop_size'], pad_if_needed=True) ),
                            sT.Sync( sT.RandomHorizontalFlip() ),
                            sT.Multi( sT.ColorJitter(0.5, 0.5, 0.5), None  ),
                            sT.Multi( sT.ToTensor(), sT.ToTensor(normalize=False, dtype=torch.long) ),
                            sT.Multi( sT.Normalize( data_cfg['mean'], data_cfg['std'] ), None )
                        ])),
            batch_size=data_cfg['batch_size'], shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            kamaldatasets.Cityscapes(data_root, split='val',
                        transforms=sT.Compose([
                            sT.Multi( sT.Resize(data_cfg['input_size']), 
                                sT.Resize(data_cfg['input_size'], interpolation=Image.NEAREST) ),
                            sT.Multi( sT.ToTensor(), sT.ToTensor(normalize=False, dtype=torch.long) ),
                            sT.Multi( sT.Normalize( data_cfg['mean'], data_cfg['std'] ), None )
                        ])),
            batch_size=4, shuffle=False, num_workers=2)
    elif name.lower() in ['voc2012', 'voc2012aug']:
        year = name.lower()[3:]
        train_loader = torch.utils.data.DataLoader(
            kamaldatasets.VOCSegmentation(data_root,  year=year, image_set='train',
                        transforms=sT.Compose([
                            sT.Multi( sT.Resize(data_cfg['input_size']), 
                                sT.Resize(data_cfg['input_size'], interpolation=Image.NEAREST) ),
                            sT.Sync( sT.RandomCrop(data_cfg['crop_size'], pad_if_needed=True) ),
                            sT.Sync( sT.RandomHorizontalFlip() ),
                            sT.Multi( sT.ColorJitter(0.5, 0.5, 0.5), None  ),
                            sT.Multi( sT.ToTensor(), sT.ToTensor(normalize=False, dtype=torch.long) ),
                            sT.Multi( sT.Normalize( data_cfg['mean'], data_cfg['std']), None )
                        ])),
            batch_size=data_cfg['batch_size'], shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            kamaldatasets.VOCSegmentation(data_root,  year=year, image_set='val',
                        transforms=sT.Compose([
                            sT.Multi( sT.Resize(data_cfg['input_size']), 
                                sT.Resize(data_cfg['input_size'], interpolation=Image.NEAREST) ),
                            #sT.Multi( sT.CenterCrop(data_cfg['input_size']), sT.CenterCrop(data_cfg['input_size']) ),
                            sT.Multi( sT.ToTensor(), sT.ToTensor(normalize=False, dtype=torch.long) ),
                            sT.Multi( sT.Normalize( data_cfg['mean'], data_cfg['std'] ), None )
                        ])),
            batch_size=1, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, num_classes

def get_optimizer_and_scheduler(opt_cfg, sch_cfg, model):
    opt_cfg, sch_cfg = opt_cfg.copy(), sch_cfg.copy()
    opt_name = opt_cfg.pop('name')
    if opt_name.lower()=='sgd':
        optimizer = torch.optim.SGD( model.parameters(), **opt_cfg )
    elif opt_name.lower()=='adam':
        optimizer = torch.optim.Adam( model.parameters(), **opt_cfg )

    sch_name = sch_cfg.pop('name')
    if sch_name.lower()=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sch_cfg )
    elif sch_name.lower()=='cosineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **sch_cfg )
    elif sch_name.lower()=='none':
        scheduler = None
    
    return optimizer, scheduler

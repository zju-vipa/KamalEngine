import kamal
from kamal import engine
from kamal import vision

import torch
from kamal.vision import sync_transforms as sT
from copy import deepcopy

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['stanford_dogs', 'cub200'])
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--total_iters', default=20000, type=int)
args = parser.parse_args()


if args.dataset=='stanford_dogs':
    num_classes=120
    train_dst = vision.datasets.StanfordDogs( '~/Datasets/StanfordDogs', split='train')
    val_dst = vision.datasets.StanfordDogs( '~/Datasets/StanfordDogs', split='test')
elif args.dataset=='cub200':
    num_classes=200
    train_dst = vision.datasets.CUB200( '~/Datasets/CUB200', split='train')
    val_dst = vision.datasets.CUB200( '~/Datasets/CUB200', split='test')
else:
    raise NotImplementedError

model = vision.models.classification.resnet18( num_classes=num_classes, pretrained=True )
task = engine.task.ClassificationTask()
trainer = engine.trainer.SimpleTrainer( task=task, model=model )
train_dst.transform = sT.Compose( [
                        sT.Resize(224),
                        sT.RandomCrop( 224 ),
                        sT.RandomHorizontalFlip(),
                        sT.ToTensor(),
                        sT.Normalize( mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225] )
                    ] )
val_dst.transform = sT.Compose( [
                        sT.Resize(224),
                        sT.CenterCrop( 224 ),
                        sT.ToTensor(),
                        sT.Normalize( mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225] )
                    ] )
train_loader = torch.utils.data.DataLoader( dataset=train_dst, batch_size=32, num_workers=4, shuffle=True )
val_loader = torch.utils.data.DataLoader( dataset=val_dst, batch_size=32, num_workers=4, shuffle=False )
evaluator = engine.evaluator.ClassificationEvaluator( val_loader )

optimizer = torch.optim.SGD( model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4 )
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=args.total_iters )

trainer.add_callbacks([
    engine.callbacks.ValidationCallback( 
        interval=len(train_loader), 
        evaluator=evaluator,
        ckpt_tag=args.dataset
    ),
    engine.callbacks.LRSchedulerCallback(
        scheduler=[ scheduler ]
    )
])
trainer.train(0, args.total_iters, train_loader, optimizer)

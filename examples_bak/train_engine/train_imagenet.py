import argparse
import torch
import torch.nn as nn

from kamal import engine, metrics, vision
from kamal.vision import sync_transforms as sT

from visdom import Visdom
import random
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--data_root', type=str, default='./data/ILSVRC2012')
    parser.add_argument('--model', type=str, required=True, choices=['darknet19', 'darknet53'])
    args = parser.parse_args()

    train_loader = torch.utils.data.DataLoader(
        vision.datasets.ImageNet(args.data_root, split='train', transform=sT.Compose([
            sT.RandomResizedCrop(224),
            sT.RandomHorizontalFlip(),
            sT.ToTensor(),
            sT.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])
        ), batch_size=64, num_workers=4, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        vision.datasets.ImageNet(args.data_root, split='val', transform=sT.Compose([
                sT.Resize(256),
                sT.CenterCrop(224),
                sT.ToTensor(),
                sT.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])]) 
        ), batch_size=64, num_workers=4
    )
    
    # Prepare model
    task = engine.task.ClassificationTask( criterion=nn.CrossEntropyLoss(ignore_index=255) )
    if args.model=='darknet19':
        model = vision.models.classification.darknet19(num_classes=1000)
    elif args.model=='darknet53':
        model = vision.models.classification.darknet53(num_classes=1000)
    
    # prepare trainer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader)*30, gamma=0.1)
    evaluator = engine.evaluator.ClassificationEvaluator( val_loader )
    trainer = engine.trainer.SimpleTrainer( task=task, model=model, train_loader=train_loader, optimizer=optimizer )

    viz = Visdom(port='29999', env=args.model)

    trainer.add_callbacks( [
        engine.callbacks.LoggingCallback(
            interval=100,  
            names=('total_loss', 'lr'), 
            smooth_window_sizes=( 100, None ),
            viz=viz),
        engine.callbacks.LRSchedulerCallback(scheduler=scheduler),
        engine.callbacks.SimpleValidationCallback(
            interval=5000, 
            evaluator=evaluator, 
            save_model=('best', 'latest'), 
            ckpt_dir='checkpoints',
            ckpt_tag=args.model,
            viz = viz)  
    ] )

    trainer.train(0, len(train_loader)*90)

if __name__=='__main__':
    main()
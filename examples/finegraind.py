from kamal import vision, engine, callbacks
from kamal.vision import sync_transforms as sT
import kamal

import torch, time
from torch.utils.tensorboard import SummaryWriter


import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--dataset', required=True )
parser.add_argument( '--lr', type=float, default=0.01)
args = parser.parse_args()

def main():
    if args.dataset=='stanford_dogs':
        num_classes=120
        train_dst = vision.datasets.StanfordDogs( 'data/StanfordDogs', split='train')
        val_dst = vision.datasets.StanfordDogs( 'data/StanfordDogs', split='test')
    elif args.dataset=='cub200':
        num_classes=200
        train_dst = vision.datasets.CUB200( 'data/CUB200', split='train')
        val_dst = vision.datasets.CUB200( 'data/CUB200', split='test')
    elif args.dataset=='fgvc_aircraft':
        num_classes=102  
        train_dst = vision.datasets.FGVCAircraft( 'data/FGVCAircraft/', split='trainval')
        val_dst = vision.datasets.FGVCAircraft( 'data/FGVCAircraft/', split='test')
    elif args.dataset=='stanford_cars':
        num_classes=196
        train_dst = vision.datasets.StanfordCars( 'data/StanfordCars/', split='train')
        val_dst = vision.datasets.StanfordCars( 'data/StanfordCars/', split='test')
    else:
        raise NotImplementedError
    model = vision.models.classification.resnet18( num_classes=num_classes, pretrained=False )
    train_dst.transform = sT.Compose( [
                            sT.RandomResizedCrop(224),
                            sT.RandomHorizontalFlip(),
                            sT.ToTensor(),
                            sT.Normalize( mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225] )
                        ] )
    val_dst.transform = sT.Compose( [
                            sT.Resize(256),
                            sT.CenterCrop(224),
                            sT.ToTensor(),
                            sT.Normalize( mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225] )
                        ] )
    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=32, shuffle=True, num_workers=4 )
    val_loader = torch.utils.data.DataLoader( val_dst, batch_size=32, num_workers=4 )
    TOTAL_ITERS=len(train_loader) * 100
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.SGD( model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4 )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )

    metric = kamal.tasks.StandardMetrics.classification()
    evaluator = engine.evaluator.BasicEvaluator( val_loader, metric=metric, progress=False )

    task = kamal.tasks.StandardTask.classification()
    trainer = engine.trainer.BasicTrainer( 
        logger=kamal.utils.logger.get_logger(args.dataset), 
        tb_writer=SummaryWriter( log_dir='run/%s-%s'%(args.dataset, time.asctime().replace( ' ', '_' ) ) ) 
    )
    trainer.setup( model=model, 
                   task=task,
                   dataloader=train_loader,
                   optimizer=optim,
                   device=device )
    
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_STEP(every=10), 
        callbacks=callbacks.MetricsLogging(keys=('total_loss', 'lr')))
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_EPOCH, 
        callbacks=callbacks.EvalAndCkpt(model=model, evaluator=evaluator, metric_name='acc', ckpt_prefix=args.dataset) )
    trainer.add_callback(
        engine.DefaultEvents.AFTER_STEP,
        callbacks=callbacks.LRSchedulerCallback(schedulers=[sched]))
    trainer.run(start_iter=0, max_iter=TOTAL_ITERS)
    
if __name__=='__main__':
    main()
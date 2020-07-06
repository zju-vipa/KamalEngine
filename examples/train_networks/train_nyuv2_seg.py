from kamal import vision, engine, utils, metrics
from kamal.vision import sync_transforms as sT

import torch, time
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

def main():
    # PyTorch Part
    num_classes = 13
    model = vision.models.segmentation.segnet_vgg19_bn(num_classes=num_classes, pretrained_backbone=True)
    train_dst = vision.datasets.NYUv2( 
        '../data/NYUv2', split='train', target_type='semantic', transforms=sT.Compose([
            sT.Multi( sT.Resize(240),     sT.Resize(240, interpolation=Image.NEAREST)),
            sT.Sync(  sT.RandomCrop(240), sT.RandomCrop(240) ),
            sT.Sync(  sT.RandomHorizontalFlip(), sT.RandomHorizontalFlip() ),
            sT.Multi( sT.ColorJitter(0.4, 0.4, 0.4), None),
            sT.Multi( sT.ToTensor(), sT.ToTensor( normalize=False, dtype=torch.long ) ),
            sT.Multi( sT.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), None )
        ]) )
    val_dst = vision.datasets.NYUv2( 
        '../data/NYUv2', split='test', target_type='semantic', transforms=sT.Compose([
            sT.Multi( sT.Resize(240), sT.Resize(240, interpolation=Image.NEAREST)),
            sT.Multi( sT.ToTensor(),  sT.ToTensor( normalize=False, dtype=torch.long ) ),
            sT.Multi( sT.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), None )
        ]) )
    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=16, shuffle=True, num_workers=4 )
    val_loader = torch.utils.data.DataLoader( val_dst, batch_size=16, num_workers=4 )
    TOTAL_ITERS=len(train_loader) * 200
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.SGD( model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4 )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )

    # KAE Part
    metric = metrics.StandardTaskMetrics.segmentation(num_classes=num_classes)
    evaluator = engine.evaluator.BasicEvaluator( data_loader=val_loader, metric=metric, progress=False )
    task = engine.task.StandardTask.segmentation()
    trainer = engine.trainer.BasicTrainer( 
        logger=utils.logger.get_logger('nyuv2_seg'), 
        viz=SummaryWriter( log_dir='run/nyuv2_seg-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    )
    trainer.add_callbacks([
        engine.callbacks.EvalAndCkptCallback( 
            len(train_loader), 
            evaluator, 
            save_type=('best', ),
            ckpt_prefix='nyuv2_seg',
            verbose=False ),
        engine.callbacks.LoggingCallback( interval=10, keys=('total_loss', 'lr') ),
        engine.callbacks.LRSchedulerCallback( schedulers=[sched] ),
        engine.callbacks.VisualizeSegmentationCallBack( 
            interval=len(train_loader), 
            dataset=val_dst, 
            idx_list_or_num_vis=10,
            normalizer=utils.Normalizer( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], ) 
        )
    ])
    trainer.setup( model=model, task=task,
                   data_loader=train_loader,
                   optimizer=optim,
                   device=device )
    trainer.run( start_iter=0, max_iter=TOTAL_ITERS )

if __name__=='__main__':
    main()
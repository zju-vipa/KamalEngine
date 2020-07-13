from kamal import vision, engine, callbacks
from kamal.vision import sync_transforms as sT
import kamal

import torch, time
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

def main():
    # PyTorch Part
    model = vision.models.segmentation.segnet_vgg16_bn(num_classes=1, pretrained_backbone=True)
    train_dst = vision.datasets.NYUv2( 
        'data/NYUv2', split='train', target_type='depth', transforms=sT.Compose([
            sT.Multi( sT.Resize(240),  sT.Resize(240)),
            sT.Sync(  sT.RandomCrop(240),  sT.RandomCrop(240)),
            sT.Sync(  sT.RandomHorizontalFlip(), sT.RandomHorizontalFlip() ), 
            sT.Multi( sT.ToTensor(), sT.ToTensor( normalize=False, dtype=torch.float ) ),
            sT.Multi( sT.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), sT.Lambda(lambda x: x/1000 ) )
        ]) )
    val_dst = vision.datasets.NYUv2( 
        'data/NYUv2', split='test', target_type='depth', transforms=sT.Compose([
            sT.Multi( sT.Resize(240),  sT.Resize(240)),
            sT.Multi( sT.ToTensor(),  sT.ToTensor( normalize=False, dtype=torch.float ) ),
            sT.Multi( sT.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), sT.Lambda(lambda x: x/1000) )
        ]) )
    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=16, shuffle=True, num_workers=4 )
    val_loader = torch.utils.data.DataLoader( val_dst, batch_size=16, num_workers=4 )
    TOTAL_ITERS=len(train_loader) * 200
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.SGD( model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4 )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )

    # KAE Part
    metric = kamal.tasks.StandardMetrics.monocular_depth()
    evaluator = engine.evaluator.BasicEvaluator( dataloader=val_loader, metric=metric, progress=False )
    
    task = kamal.tasks.StandardTask.monocular_depth()
    trainer = engine.trainer.BasicTrainer( 
        logger=kamal.utils.logger.get_logger('nyuv2_depth'), 
        tb_writer=SummaryWriter( log_dir='run/nyuv2_depth-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
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
        engine.DefaultEvents.AFTER_STEP,
        callbacks=callbacks.LRSchedulerCallback(schedulers=[sched]))
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_EPOCH, 
        callbacks=[ 
            callbacks.EvalAndCkpt(model=model, evaluator=evaluator, metric_name='rmse', metric_mode='min', ckpt_prefix='nyuv2_depth'),
            callbacks.VisualizeDepth(
                model=model,
                dataset=val_dst, 
                idx_list_or_num_vis=10,
                max_depth=10,
                normalizer=kamal.utils.Normalizer( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], reverse=True),
            )])
    trainer.run( start_iter=0, max_iter=TOTAL_ITERS )

if __name__=='__main__':
    main()
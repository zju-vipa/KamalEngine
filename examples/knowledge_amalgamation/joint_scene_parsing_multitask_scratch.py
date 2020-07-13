from kamal import vision, engine, metrics, callbacks, tasks
from kamal.vision import sync_transforms as sT
import kamal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import time

class MultiTaskSegNet(nn.ModuleList):
    def __init__(self, out_channel_list, segnet_build_fn=vision.models.segmentation.segnet_vgg16_bn):
        super( MultiTaskSegNet, self ).__init__()
        encoder = None
        decoders = []
        for oc in out_channel_list:
            segnet = segnet_build_fn( pretrained_backbone=True, num_classes=oc )
            if encoder is None:
                encoder = nn.ModuleList( [ getattr( segnet, 'down%d'%i ) for i in range(1,6) ] )
            decoders.append( nn.ModuleList( [ getattr( segnet, 'up%d'%i ) for i in range(1,6) ] ) ) 
        self.encoder = encoder
        self.decoders = nn.ModuleList( decoders )

    def forward(self, inputs):
        down1, indices_1, unpool_shape1 = self.encoder[0](inputs)
        down2, indices_2, unpool_shape2 = self.encoder[1](down1)
        down3, indices_3, unpool_shape3 = self.encoder[2](down2)
        down4, indices_4, unpool_shape4 = self.encoder[3](down3)
        down5, indices_5, unpool_shape5 = self.encoder[4](down4)

        outputs = []
        for decoder in self.decoders:
            up5 = decoder[4](down5, indices_5, unpool_shape5)
            up4 = decoder[3](up5, indices_4, unpool_shape4)
            up3 = decoder[2](up4, indices_3, unpool_shape3)
            up2 = decoder[1](up3, indices_2, unpool_shape2)
            up1 = decoder[0](up2, indices_1, unpool_shape1)
            outputs.append( up1 )
        return outputs

def main():
    # Seg + Depth
    model = MultiTaskSegNet(out_channel_list=[13, 1])
    seg_train_dst = vision.datasets.NYUv2( '../data/NYUv2', split='train', target_type='semantic' )
    seg_val_dst = vision.datasets.NYUv2( '../data/NYUv2', split='test', target_type='semantic' )
    depth_train_dst = vision.datasets.NYUv2( '../data/NYUv2', split='train', target_type='depth' )
    depth_val_dst = vision.datasets.NYUv2( '../data/NYUv2', split='test', target_type='depth' )
    train_dst = vision.datasets.LabelConcatDataset( 
            datasets=[seg_train_dst, depth_train_dst], 
            transforms=sT.Compose([
                sT.Multi( sT.Resize(240), sT.Resize(240, interpolation=Image.NEAREST), sT.Resize(240)),
                sT.Sync(  sT.RandomCrop(240), sT.RandomCrop(240), sT.RandomCrop(240) ),
                sT.Sync(  sT.RandomHorizontalFlip(), sT.RandomHorizontalFlip(), sT.RandomHorizontalFlip() ),
                sT.Multi( sT.ToTensor(), sT.ToTensor( normalize=False, dtype=torch.long ), sT.ToTensor( normalize=False, dtype=torch.float )),
                sT.Multi( sT.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), sT.Lambda(lambd=lambda x: x.squeeze()), sT.Lambda( lambd=lambda x: x/1e3 ) )
            ]))
    val_dst = vision.datasets.LabelConcatDataset( 
            datasets=[seg_val_dst, depth_val_dst], 
            transforms=sT.Compose([
                sT.Multi( sT.Resize(240), sT.Resize(240, interpolation=Image.NEAREST), sT.Resize(240)),
                sT.Multi( sT.ToTensor(), sT.ToTensor( normalize=False, dtype=torch.long ), sT.ToTensor( normalize=False, dtype=torch.float ) ),
                sT.Multi( sT.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), sT.Lambda(lambd=lambda x: x.squeeze()), sT.Lambda( lambd=lambda x: x/1e3 ) )
            ]))

    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=16, shuffle=True, num_workers=4 )
    val_loader = torch.utils.data.DataLoader( val_dst, batch_size=16, num_workers=4 )
    TOTAL_ITERS=len(train_loader) * 200
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.Adam( model.parameters(), lr=1e-4, weight_decay=1e-4 )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )

    
    confusion_matrix = metrics.ConfusionMatrix(num_classes=13, ignore_idx=255, attach_to=0)
    metric = metrics.MetricCompose(
        metric_dict={ 'acc': metrics.Accuracy( attach_to=0 ),
                      'cm': confusion_matrix,
                      'mIoU': metrics.mIoU(confusion_matrix),
                      'rmse': metrics.RootMeanSquaredError(attach_to=1) })
    evaluator = engine.evaluator.BasicEvaluator( dataloader=val_loader, metric=metric, progress=False )
    
    task = [ kamal.tasks.StandardTask.segmentation(attach_to=[0,0]), 
                kamal.tasks.StandardTask.monocular_depth(attach_to=[1,1]) ]
    trainer = engine.trainer.BasicTrainer( 
        logger=kamal.utils.logger.get_logger('nyuv2_multitasking'), 
        tb_writer=SummaryWriter( log_dir='run/nyuv2_multitasking-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    )
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_STEP(every=10), 
        callbacks=callbacks.MetricsLogging(keys=('total_loss', 'lr')))
    trainer.add_callback(
        engine.DefaultEvents.AFTER_STEP,
        callbacks=callbacks.LRSchedulerCallback(schedulers=[sched]))
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_EPOCH, 
        callbacks=[ 
            callbacks.EvalAndCkpt(model=model, evaluator=evaluator, metric_name='rmse', metric_mode='min', ckpt_prefix='nyuv2_multitasking'),
            callbacks.VisualizeSegmentation( 
                model=model,
                dataset=val_dst, 
                idx_list_or_num_vis=10,
                attach_to=0,
                normalizer=kamal.utils.Normalizer( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], reverse=True),
            ),
            callbacks.VisualizeDepth( 
                model=model,
                dataset=val_dst, 
                idx_list_or_num_vis=10,
                max_depth=10,
                attach_to=1,
                normalizer=kamal.utils.Normalizer( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], reverse=True),
            ),
        ] )
    trainer.setup( model=model, task=task,
                   dataloader=train_loader,
                   optimizer=optim,
                   device=device )
    trainer.run( start_iter=0, max_iter=TOTAL_ITERS )

if __name__=='__main__':
    main()
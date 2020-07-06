from kamal import vision, engine, utils, metrics, criterions
from kamal.vision import sync_transforms as sT

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import time

class MultiTaskSegNet(nn.ModuleList):
    def __init__(self, out_channel_list, segnet_build_fn=vision.models.segmentation.segnet_vgg19_bn):
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
                sT.Multi( sT.ColorJitter(0.4, 0.4, 0.4), None, None),
                sT.Multi( sT.ToTensor(), sT.ToTensor( normalize=False, dtype=torch.long ), sT.ToTensor( normalize=False, dtype=torch.float )),
                sT.Multi( sT.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), None, sT.Lambda( lambd=lambda x: x/1e3 ) )
            ]))
    val_dst = vision.datasets.LabelConcatDataset( 
            datasets=[seg_val_dst, depth_val_dst], 
            transforms=sT.Compose([
                sT.Multi( sT.Resize(240), sT.Resize(240, interpolation=Image.NEAREST), sT.Resize(240)),
                sT.Multi( sT.ToTensor(), sT.ToTensor( normalize=False, dtype=torch.long ), sT.ToTensor( normalize=False, dtype=torch.float ) ),
                sT.Multi( sT.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), None, sT.Lambda( lambd=lambda x: x/1e3 ) )
            ]))

    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=16, shuffle=True, num_workers=4 )
    val_loader = torch.utils.data.DataLoader( val_dst, batch_size=16, num_workers=4 )
    TOTAL_ITERS=len(train_loader) * 200
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.SGD( model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4 )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )

    def get_seg(outputs, targets):
        return outputs[0], targets[0].squeeze(1)

    def get_depth(outputs, targets):
        return outputs[1].view_as(targets[1]), targets[1]

    confusion_matrix = metrics.ConfusionMatrix(num_classes=13, ignore_idx=255, output_target_transform=get_seg)
    metric = metrics.MetricCompose(
        metric_dict={ 'acc': metrics.Accuracy( output_target_transform=get_seg ),
                      'cm': confusion_matrix,
                      'mIoU': metrics.mIoU(confusion_matrix),
                      'rmse': metrics.RootMeanSquaredError(output_target_transform=get_depth) },
        primary_metric='mIoU',
    )
    evaluator = engine.evaluator.BasicEvaluator( data_loader=val_loader, metric=metric, progress=False )
    task = engine.task.Task(criterion_dict={
        'ce': criterions.Criterion( nn.CrossEntropyLoss(ignore_index=255), scale=1, output_target_transform=get_seg ),
        'l1': criterions.Criterion( nn.L1Loss(), scale=1, output_target_transform=get_depth ),
    })
    trainer = engine.trainer.BasicTrainer( 
        logger=utils.logger.get_logger('nyuv2_multitasking'), 
        viz=SummaryWriter( log_dir='run/nyuv2_multitasking-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    )
    trainer.add_callbacks([
        engine.callbacks.EvalAndCkptCallback( 
            len(train_loader), 
            evaluator, 
            ckpt_prefix='nyuv2_multitasking',
            verbose=False ),
        engine.callbacks.LoggingCallback( interval=10, keys=('total_loss', 'ce', 'l1', 'lr') ),
        engine.callbacks.LRSchedulerCallback( schedulers=[sched] ),
        engine.callbacks.VisualizeSegmentationCallBack( 
            interval=len(train_loader), 
            dataset=val_dst, 
            idx_list_or_num_vis=10,
            normalizer=utils.Normalizer( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ),
            decode_fn=seg_train_dst.decode_fn,
            output_target_transform=get_seg
        ),
        engine.callbacks.VisualizeDepthCallBack( 
            interval=len(train_loader), 
            dataset=val_dst, 
            idx_list_or_num_vis=10,
            normalizer=utils.Normalizer( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ),
            output_target_transform=get_depth,
            max_depth=10
        )  
    ])
    trainer.setup( model=model, task=task,
                   data_loader=train_loader,
                   optimizer=optim,
                   device=device )
    trainer.run( start_iter=0, max_iter=TOTAL_ITERS )

if __name__=='__main__':
    main()
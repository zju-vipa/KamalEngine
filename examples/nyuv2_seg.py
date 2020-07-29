# Copyright 2020 Zhejiang Lab. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================

from kamal import vision, engine, callbacks
from kamal.vision import sync_transforms as sT
import kamal

import torch, time
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

def main():
    # PyTorch Part
    num_classes = 13
    model = vision.models.segmentation.segnet_vgg16_bn(num_classes=num_classes, pretrained_backbone=True)
    train_dst = vision.datasets.NYUv2( 
        'data/NYUv2', split='train', target_type='semantic', transforms=sT.Compose([
            sT.Multi( sT.Resize(240), sT.Resize(240, interpolation=Image.NEAREST)),
            sT.Sync(  sT.RandomRotation(5),  sT.RandomRotation(5)),
            sT.Multi( sT.ColorJitter(0.2, 0.2, 0.2), None),
            sT.Sync(  sT.RandomCrop(240),  sT.RandomCrop(240)),
            sT.Sync(  sT.RandomHorizontalFlip(), sT.RandomHorizontalFlip() ),
            sT.Multi( sT.ToTensor(), sT.ToTensor( normalize=False, dtype=torch.long) ),
            sT.Multi( sT.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), sT.Lambda(lambd=lambda x: x.squeeze()) )
        ]) )
    val_dst = vision.datasets.NYUv2( 
        'data/NYUv2', split='test', target_type='semantic', transforms=sT.Compose([
            sT.Multi( sT.Resize(240), sT.Resize(240, interpolation=Image.NEAREST)),
            sT.Multi( sT.ToTensor(),  sT.ToTensor( normalize=False, dtype=torch.long ) ),
            sT.Multi( sT.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), sT.Lambda(lambd=lambda x: x.squeeze()) )
        ]) )
    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=16, shuffle=True, num_workers=4 )
    val_loader = torch.utils.data.DataLoader( val_dst, batch_size=16, num_workers=4 )
    TOTAL_ITERS=len(train_loader) * 200
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.SGD( model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4 )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )

    # KAE Part
    metric = kamal.tasks.StandardMetrics.segmentation(num_classes=num_classes)
    evaluator = engine.evaluator.BasicEvaluator( dataloader=val_loader, metric=metric, progress=False )
    task = kamal.tasks.StandardTask.segmentation()
    trainer = engine.trainer.BasicTrainer( 
        logger=kamal.utils.logger.get_logger('nyuv2_seg'), 
        tb_writer=SummaryWriter( log_dir='run/nyuv2_seg-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
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
            callbacks.EvalAndCkpt(model=model, evaluator=evaluator, metric_name='miou', ckpt_prefix='nyuv2_seg'),
            callbacks.VisualizeSegmentation(
                model=model,
                dataset=val_dst, 
                idx_list_or_num_vis=10,
                normalizer=kamal.utils.Normalizer( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], reverse=True),
            )])
    trainer.run( start_iter=0, max_iter=TOTAL_ITERS )

if __name__=='__main__':
    main()
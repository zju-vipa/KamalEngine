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

from kamal import vision, amalgamation, metrics, callbacks, tasks, engine
from kamal.vision import sync_transforms as sT
import kamal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--seg_ckpt', required=True )
parser.add_argument( '--depth_ckpt', required=True )
parser.add_argument( '--lr', type=float, default=1e-3)
parser.add_argument( '--test_only', action='store_true', default=False)
parser.add_argument( '--ckpt', type=str, default=None)
args = parser.parse_args()

def main():
    seg_teacher = vision.models.segmentation.segnet_vgg16_bn(num_classes=13, pretrained_backbone=True)
    depth_teacher = vision.models.segmentation.segnet_vgg16_bn(num_classes=1, pretrained_backbone=True)
    seg_teacher.load_state_dict( torch.load(args.seg_ckpt) )
    depth_teacher.load_state_dict( torch.load( args.depth_ckpt ) )
    student = vision.models.segmentation.segnet_vgg16_bn(num_classes=1, pretrained_backbone=True)
    student = amalgamation.task_branching.BranchySegNet( out_channels=[13, 1], segnet_fn=vision.models.segmentation.segnet_vgg16_bn )
    
    # target is not necessary
    train_dst = vision.datasets.NYUv2( '../data/NYUv2', split='train', target_type='semantic' )
    seg_val_dst = vision.datasets.NYUv2( '../data/NYUv2', split='test', target_type='semantic' )
    depth_val_dst = vision.datasets.NYUv2( '../data/NYUv2', split='test', target_type='depth' )
    train_dst.transforms = sT.Compose([
            sT.Multi( sT.Resize(240),  sT.Resize(240, interpolation=Image.NEAREST)),
            sT.Sync(  sT.RandomCrop(240),  sT.RandomCrop(240)),
            sT.Sync(  sT.RandomHorizontalFlip(), sT.RandomHorizontalFlip() ),  
            sT.Multi( sT.ToTensor(), sT.ToTensor( normalize=False, dtype=torch.long ) ),
            sT.Multi( sT.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ), sT.Lambda(lambd=lambda x: x.squeeze()) )
        ])

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
    optim = torch.optim.Adam( student.parameters(), lr=1e-4, weight_decay=1e-4 )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )

    confusion_matrix = metrics.ConfusionMatrix(num_classes=13, ignore_idx=255, attach_to=0)
    metric = metrics.MetricCompose(
        metric_dict={ 'acc': metrics.Accuracy( attach_to=0 ),
                      'cm': confusion_matrix,
                      'mIoU': metrics.mIoU(confusion_matrix),
                      'rmse': metrics.RootMeanSquaredError(attach_to=1) },
    )
    evaluator = engine.evaluator.BasicEvaluator( dataloader=val_loader, metric=metric, progress=False )
    student_tasks = [ kamal.tasks.StandardTask.distillation(attach_to=0),
                        kamal.tasks.StandardTask.monocular_depth(attach_to=1) ]
    trainer = amalgamation.TaskBranchingAmalgamator( 
        logger=kamal.utils.logger.get_logger('nyuv2_task_branching'), 
        tb_writer=SummaryWriter( log_dir='run/nyuv2_task_branching-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    )
    trainer.setup( joint_student=student, 
                   teachers=[seg_teacher, depth_teacher], tasks=student_tasks,
                   dataloader=train_loader,
                   optimizer=optim,
                   device=device )

    trainer.add_callback( 
        engine.DefaultEvents.AFTER_STEP(every=10), 
        callbacks=callbacks.MetricsLogging(keys=('total_loss', 'seg_kld', 'depth_l1', 'branch', 'lr')))
    trainer.add_callback(
        engine.DefaultEvents.AFTER_STEP,
        callbacks=callbacks.LRSchedulerCallback(schedulers=[sched]))
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_EPOCH, 
        callbacks=[ 
            callbacks.EvalAndCkpt(model=student, evaluator=evaluator, metric_name='mIoU', metric_mode='max', ckpt_prefix='nyuv2_task_branching'),
            callbacks.VisualizeDepth( 
                model=student,
                dataset=val_dst, 
                idx_list_or_num_vis=10,
                max_depth=10,
                attach_to=1,
                normalizer=kamal.utils.Normalizer( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], reverse=True),
            ),
            callbacks.VisualizeSegmentation( 
                model=student,
                dataset=val_dst, 
                idx_list_or_num_vis=10,
                attach_to=0,
                normalizer=kamal.utils.Normalizer( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], reverse=True),
            )
        ])
    
    trainer.run( start_iter=0, max_iter=TOTAL_ITERS )

if __name__=='__main__':
    main()
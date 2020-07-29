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

def main():
    # Pytorch Part
    model = vision.models.classification.cifar.wrn.wrn_40_2(num_classes=100)
    train_dst = vision.datasets.torchvision_datasets.CIFAR100( 
        'data/torchdata', train=True, download=True, transform=sT.Compose([
            sT.RandomCrop(32, padding=4),
            sT.RandomHorizontalFlip(),
            sT.ToTensor(),
            sT.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) )
        ]) )
    val_dst = vision.datasets.torchvision_datasets.CIFAR100( 
        'data/torchdata', train=False, download=True, transform=sT.Compose([
            sT.ToTensor(),
            sT.Normalize( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) )
        ]) )
    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=128, shuffle=True, num_workers=4 )
    val_loader = torch.utils.data.DataLoader( val_dst, batch_size=128, num_workers=4 )
    TOTAL_ITERS=len(train_loader) * 200
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.SGD( model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4 )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )

    # KAE Part
    # prepare evaluator
    metric = kamal.tasks.StandardMetrics.classification()
    evaluator = engine.evaluator.BasicEvaluator(dataloader=val_loader, metric=metric, progress=False)
    
    # prepare trainer
    task = kamal.tasks.StandardTask.classification()
    trainer = engine.trainer.BasicTrainer( 
        logger=kamal.utils.logger.get_logger('cifar100'), 
        tb_writer=SummaryWriter( log_dir='run/cifar100-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    )
    trainer.setup( model=model, 
                   task=task, 
                   dataloader=train_loader,
                   optimizer=optim,
                   device=device )

    # add callbacks
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_EPOCH, 
        callbacks=callbacks.EvalAndCkpt(model=model, evaluator=evaluator, metric_name='acc', ckpt_prefix='cifar100') )
    trainer.add_callback(
        engine.DefaultEvents.AFTER_STEP,
        callbacks=callbacks.LRSchedulerCallback(schedulers=[sched]))
    # run
    trainer.run(start_iter=0, max_iter=TOTAL_ITERS)

if __name__=='__main__':
    main()
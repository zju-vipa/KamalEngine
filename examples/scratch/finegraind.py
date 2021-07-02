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
import torch.nn as nn

import torch, time
from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--dataset', required=True )
parser.add_argument( '--lr', type=float, default=0.01)
parser.add_argument( '--epochs', type=int, default=200)
parser.add_argument( '--pretrained', default=False, action='store_true')
args = parser.parse_args()

def main():
    # Pytorch Part
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
    
    model = vision.models.classification.resnet18( num_classes=num_classes, pretrained=args.pretrained )
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
    TOTAL_ITERS=len(train_loader) * args.epochs
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.SGD( model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4 )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )

    # KAE Part
    # Predefined task & metrics
    criterion = nn.CrossEntropyLoss()
    metric = kamal.predefined.predefined_metrics.classification()
    # Evaluator and Trainer
    evaluator = engine.evaluator.BasicEvaluator( val_loader, metric=metric )
    trainer = engine.trainer.BasicTrainer( 
        logger=kamal.utils.logger.get_logger(args.dataset), 
        tb_writer=SummaryWriter(log_dir='run/%s-%s'%(args.dataset, time.asctime().replace( ' ', '_' )) ) 
    )
    # setup trainer
    trainer.setup( model=model, 
                   criterion=criterion,
                   dataloader=train_loader,
                   optimizer=optim,
                   device=device )
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_STEP(every=10), 
        callbacks=callbacks.MetricsLogging(keys=(trainer.TOTAL_LOSS_METRIC, trainer.LR_METRIC)))
    trainer.add_callback(
        engine.DefaultEvents.AFTER_STEP,
        callbacks=callbacks.LRSchedulerCallback(schedulers=[sched]))
    ckpt_callback = trainer.add_callback( 
        engine.DefaultEvents.AFTER_EPOCH, 
        callbacks=callbacks.EvalAndCkpt(model=model, evaluator=evaluator, metric_name='Acc', ckpt_prefix=args.dataset) )
    trainer.run(start_iter=0, max_iter=TOTAL_ITERS)
    ckpt_callback.callback.final_ckpt(ckpt_dir='pretrained', add_md5=True)
    
if __name__=='__main__':
    main()
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

from torch.utils import data
import kamal
from kamal import vision, engine, utils, amalgamation, metrics, callbacks
from kamal.vision import sync_transforms as sT
from kamal.amalgamation.subtask import TransferHeadTrainer

import torch, time
from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--sub_list', type=int, nargs='+')
parser.add_argument( '--dataset', type=str)
parser.add_argument( '--lr', type=float, default=1e-3)
parser.add_argument( '--test_only', action='store_true', default=False)
parser.add_argument( '--ckpt', type=str, default=None)
args = parser.parse_args()

from torch.utils.data import Dataset
from kamal.vision.datasets.subclstask import SubClsDataset

def main():
    
    train_transform = sT.Compose( [
                            sT.RandomResizedCrop(224),
                            sT.RandomHorizontalFlip(),
                            sT.ToTensor(),
                            sT.Normalize( mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225] )
                        ] )
    val_transform = sT.Compose( [
                            sT.Resize(256),
                            sT.CenterCrop( 224 ),
                            sT.ToTensor(),
                            sT.Normalize( mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225] )
                        ] )

    if args.sub_dataset=='stanford_dogs':
        num_classes=120
        train_dst = vision.datasets.StanfordDogs( '../data/StanfordDogs', split='train', transform=train_transform)
        val_dst = vision.datasets.StanfordDogs( '../data/StanfordDogs', split='test', transform=val_transform)
    elif args.sub_dataset=='cub200':
        num_classes=200
        train_dst = vision.datasets.CUB200( '../data/CUB200', split='train', transform=train_transform)
        val_dst = vision.datasets.CUB200( '../data/CUB200', split='test', transform=val_transform)
    elif args.sub_dataset=='fgvc_aircraft':
        num_classes=102  
        train_dst = vision.datasets.FGVCAircraft( '../data/FGVCAircraft/', split='trainval', transform=train_transform)
        val_dst = vision.datasets.FGVCAircraft( '../data/FGVCAircraft/', split='test', transform=val_transform)
    elif args.sub_dataset=='stanford_cars':
        num_classes=196
        train_dst = vision.datasets.StanfordCars( '../data/StanfordCars/', split='train', transform=train_transform)
        val_dst = vision.datasets.StanfordCars( '../data/StanfordCars/', split='test', transform=val_transform)
    else:
        raise NotImplementedError

    if args.sub_list is not None:
        train_dst = SubClsDataset(train_dst, args.sub_list)
        val_dst = SubClsDataset(val_dst, args.sub_list)
        num_classes = len(args.sub_list)

    model = vision.models.classification.resnet18( num_classes=num_classes, pretrained=False )

    train_dst, _ = data.random_split(train_dst, [(int)(0.1*len(train_dst)), len(train_dst) - (int)(0.1*len(train_dst))])
    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=32, shuffle=True, num_workers=4 )
    val_loader = torch.utils.data.DataLoader( val_dst, batch_size=32, num_workers=4 )
    TOTAL_ITERS=len(train_loader) * 50

    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim = torch.optim.SGD( filter(lambda p: p.requires_grad, model.parameters()), lr=0.02, momentum=0.9, weight_decay=5e-4 )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( optim, T_max=TOTAL_ITERS )

    metric = kamal.tasks.StandardMetrics.classification()
    val_evaluator = engine.evaluator.BasicEvaluator( val_loader, metric )
 
    if args.ckpt is not None:
        pretrained_dict = torch.load( args.ckpt )
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.split('.')[0] != "fc"}
        model.load_state_dict(pretrained_dict, strict=False )
        print("Load original model from %s"%args.ckpt)

    if args.test_only:
        model.load_state_dict( torch.load( args.ckpt ) )
        results= val_evaluator.eval( model )
        print("Average acc of transferred model: %s"%( results ))
        return 
    
    trainer = TransferHeadTrainer(  # KD can be achieved by scaling other losses to zero
        logger=utils.logger.get_logger('subtask'), 
        tb_writer=SummaryWriter( log_dir='run/subtask-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    )
    
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_STEP(every=10), 
        callbacks=callbacks.MetricsLogging(keys=('loss_kd', 'lr')))
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_EPOCH, 
        callbacks=[
            callbacks.EvalAndCkpt(model=model, evaluator=val_evaluator, metric_name='acc', ckpt_prefix='sub'),
        ] )
    trainer.add_callback(
        engine.DefaultEvents.AFTER_STEP,
        callbacks=callbacks.LRSchedulerCallback(schedulers=[sched]))

    trainer.setup( model=model, 
                   dataloader=train_loader,
                   optimizer=optim,
                   device=device,
                   weights=[1., 0., 0.] )
    trainer.run(start_iter=0, max_iter=TOTAL_ITERS)

if __name__=='__main__':
    main()
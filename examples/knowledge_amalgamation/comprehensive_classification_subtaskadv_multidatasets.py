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
from kamal.amalgamation.subtask import SubTaskAdvTrainer

import torch, time
from torch.utils.tensorboard import SummaryWriter

import argparse

from torch.utils.data import Dataset
from kamal.vision.datasets.subclstask import SubClsDataset


parser = argparse.ArgumentParser()

parser.add_argument( '--lr', type=float, default=1e-3)
parser.add_argument( '--sub_dataset', type=str)
parser.add_argument( '--sub_class', type=int, nargs='+')
parser.add_argument( '--adv_dataset', type=str, nargs='+')
parser.add_argument( '--test_only', action='store_true', default=False)
parser.add_argument( '--ckpt', type=str, default=None)
parser.add_argument( '--ckpt_sub', type=str, default=None)
args = parser.parse_args()

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

    total_num_classes=0

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

    total_num_classes += num_classes
    if args.sub_class is not None:
        sub_list = args.sub_class
        adv_list = list(set(range(num_classes)).difference(sub_list))
        val_dst = SubClsDataset(val_dst, sub_list)
    else:
        sub_list = range(num_classes)

    if args.adv_dataset is not None:
        for dataset in args.adv_dataset:
            if dataset=='stanford_dogs':
                num_classes=120
                train_dst_adv = vision.datasets.StanfordDogs( '../data/StanfordDogs', split='train', transform=train_transform)
                train_dst = data.ConcatDataset( [train_dst, train_dst_adv] )
            elif dataset=='cub200':
                num_classes=200
                train_dst_adv = vision.datasets.CUB200( '../data/CUB200', split='train', transform=train_transform)
                train_dst = data.ConcatDataset( [train_dst, train_dst_adv] )
            elif dataset=='fgvc_aircraft':
                num_classes=102  
                train_dst_adv = vision.datasets.FGVCAircraft( '../data/FGVCAircraft/', split='trainval', transform=train_transform)
                train_dst = data.ConcatDataset( [train_dst, train_dst_adv] )
            elif dataset=='stanford_cars':
                num_classes=196
                train_dst_adv = vision.datasets.StanfordCars( '../data/StanfordCars/', split='train', transform=train_transform)
                train_dst = data.ConcatDataset( [train_dst, train_dst_adv] )
            else:
                raise NotImplementedError
            adv_list.extend([i for i in range(total_num_classes, total_num_classes + num_classes)])
            total_num_classes += num_classes

    outputdim_sub = len(sub_list)
    outputdim_adv = len(adv_list)

    ori_model = vision.models.classification.resnet18( num_classes=total_num_classes, pretrained=False )
    sub_model = vision.models.classification.resnet18( num_classes=outputdim_sub, pretrained=False )
    adv_head = torch.nn.Linear(512, outputdim_adv)

    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=32, shuffle=True, num_workers=4 )
    val_loader = torch.utils.data.DataLoader( val_dst, batch_size=32, shuffle=False, num_workers=4 )    

    TOTAL_ITERS=len(train_loader) * 200

    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optim_sub = torch.optim.Adam( sub_model.parameters(), lr=5e-4, weight_decay=5e-4 )
    sched_sub = torch.optim.lr_scheduler.CosineAnnealingLR( optim_sub, T_max=TOTAL_ITERS )

    optim_adv = torch.optim.Adam( adv_head.parameters(), lr=5e-4, weight_decay=5e-4 )
    sched_adv = torch.optim.lr_scheduler.CosineAnnealingLR( optim_adv, T_max=TOTAL_ITERS )

    metric = kamal.tasks.StandardMetrics.classification()
    val_evaluator = engine.evaluator.BasicEvaluator( val_loader, metric )
 
    if args.ckpt is not None:
        ori_model.load_state_dict( torch.load( args.ckpt ) )
        print("Load original model from %s"%args.ckpt)

    if args.test_only:
        sub_model.load_state_dict( torch.load( args.ckpt_sub ) )
        results= val_evaluator.eval( sub_model )
        print("Average acc of sub-model: %s"%( results ))

        return 
    
    trainer = SubTaskAdvTrainer(  # KD can be achieved by scaling other losses to zero
        logger=utils.logger.get_logger('subtask'), 
        tb_writer=SummaryWriter( log_dir='run/subtask-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    )
    
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_STEP(every=10), 
        callbacks=callbacks.MetricsLogging(keys=('loss_kd', 'lr')))
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_EPOCH, 
        callbacks=[
            callbacks.EvalAndCkpt(model=sub_model, evaluator=val_evaluator, metric_name='acc', ckpt_prefix='sub'),
        ] )
    trainer.add_callback(
        engine.DefaultEvents.AFTER_STEP,
        callbacks=callbacks.LRSchedulerCallback(schedulers=[sched_sub, sched_adv]))

    trainer.setup( student=sub_model, 
                   teacher=ori_model,
                   adv_head = adv_head,
                   dataloader=train_loader,
                   optimizer_sub=optim_sub,
                   optimizer_adv=optim_adv,
                   sub_list=sub_list,
                   adv_list=adv_list,
                   device=device)
    trainer.run(start_iter=0, max_iter=TOTAL_ITERS)

if __name__=='__main__':
    main()
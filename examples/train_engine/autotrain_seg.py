import argparse
import os, sys
import torch
import torch.nn as nn

from PIL import Image


from kamal import engine, metrics
import time

from registry import get_model, get_dataloader, get_optimizer_and_scheduler
from kamal.vision.models.segmentation import deeplabv3plus_mobilenetv2
from kamal.vision.datasets import CamVid
from kamal.vision import sync_transforms as sT
import kamal

from ruamel_yaml import YAML
from visdom import Visdom
import random

import hyperopt

from atlas import serialize, meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--model_list', nargs='+', type=str, required=True)
    parser.add_argument('--data_list', nargs='+', type=str, required=True)

    args = parser.parse_args()

    print("Model: %s"%args.model_list)
    print("Data: %s"%args.data_list)

    for model_name in args.model_list:
        for data_name in args.data_list:
            print(model_name, data_name)
            with open( 'configs/dataset/%s.yml'%data_name, 'r') as f:
                data_cfg = YAML().load(f)
            
            model_root = 'checkpoints/%s/%s/%s'%( model_name, data_name, time.asctime().replace(' ', '_') )
            os.makedirs( model_root, exist_ok=True )

            train_loader, val_loader, num_classes = get_dataloader( data_name, data_cfg )
            model = get_model( model_name, num_classes=num_classes )
            # Prepare trainer
            task = engine.task.SegmentationTask(criterion=nn.CrossEntropyLoss(ignore_index=255))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            evaluator = engine.evaluator.SegmentationEvaluator( num_classes, val_loader )
            trainer = engine.trainer.SimpleTrainer( 
                task=task, 
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer,
                logger=kamal.utils.get_logger(name='%s_%s'%(model_name, data_name), output=os.path.join(model_root, 'logs.txt' ))
            )

            hpo = engine.hpo.HPO(trainer, evaluator, os.path.join(model_root, 'hp.yml'))
            hp = hpo.optimize(max_evals=20, max_iters=400)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, 20000)
            viz = Visdom(port='29999', env='%s_%s-%s'%(model_name, data_name, time.asctime().replace(' ', '_')))

            trainer.add_callbacks( [
                engine.callbacks.LoggingCallback(
                    interval=10,
                    names=('total_loss', 'lr' ), 
                    smooth_window_sizes=( 20, None ), # no smooth for lr
                    viz=viz),
                engine.callbacks.LRSchedulerCallback(scheduler=scheduler),
                engine.callbacks.SimpleValidationCallback(
                    interval=200, 
                    evaluator=evaluator,
                    save_model=('best', 'latest'), 
                    ckpt_dir=model_root,
                    ckpt_tag='%s_%s'%( model_name, data_name ),
                    viz = viz,
                ),
                engine.callbacks.SegVisualizationCallback(
                    interval=200,
                    viz=viz,
                    dst=val_loader.dataset,
                    idx_list_or_num_vis=10, # select 5 images for visualization
                    scale_to_255=True,     # 0~1 => 0~255
                    mean=data_cfg['mean'],  # for denormalization
                    std=data_cfg['std'])
            ] )
            trainer.train(0, 20000)

            metadata = meta.MetaData(
                name=model_name, dataset=data_name, task=meta.TASK.SEGMENTATION,
                url='https://github.com/zju-vipa/KamalEngine', 
                input=meta.ImageInput(
                    size=data_cfg['input_size'],
                    range=[0, 1],
                    space='rgb',
                    normalize=dict(mean=data_cfg['mean'], std=data_cfg['std']),
                ),
                other_meta_data=dict(
                    num_classes=data_cfg['num_classes'], 
                    crop_size=data_cfg['crop_size'],
                    batch_size=data_cfg['batch_size']
                )
            )

            deps = ['torch', 'torchvision']
            code = ['../../kamal']
            os.makedirs('exported', exist_ok=True)
            
            serialize.save( trainer.model, path='exported/%s_%s_%s_segmentation'%(model_name, time.asctime().replace(' ', '_'), data_name),
                            deps=deps, code=code, metadata=metadata )
            

if __name__=='__main__':
    main()
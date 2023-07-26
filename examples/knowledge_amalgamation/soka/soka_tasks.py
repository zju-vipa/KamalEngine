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
import torch
torch.multiprocessing.set_start_method('spawn', force=True)
import time
import sys
import os
import os.path as osp
# os.environ['CUDA_VISIBLE_DEVICES']='1'
cur_dir = osp.dirname( __file__ )
main_path = osp.join( cur_dir, '..', '..','..')
sys.path.insert( 0, main_path )
# sys.path.insert(0, "/home/yyc/KA_task/06_27_kamal")
print(sys.path)

from kamal import vision, engine, utils, amalgamation, metrics, callbacks
from kamal.vision import sync_transforms as sT
from kamal.vision.models.taskonomy import TaskonomyNetworkStudent
from kamal.amalgamation.taskonomy_amalgamation import eval_task1_fn, eval_task2_fn, TaskonomyAmalgamation



from torch.utils.tensorboard import SummaryWriter



import argparse
parser = argparse.ArgumentParser()
parser.add_argument( '--data_root', required=True )
parser.add_argument( '--task1', default='normal' )
parser.add_argument( '--task2', default='edge_occlusion')
parser.add_argument( '--lr', type=float, default=1e-4)
parser.add_argument( '--test_only', action='store_true', default=False)
parser.add_argument( '--ckpt', type=str, default=None)
args = parser.parse_args()

def main():
    train_dst = vision.datasets.Taskonomy(root=args.data_root, split='train', target_type1=args.task1, target_type2=args.task2)
    task1_test_dst = vision.datasets.Taskonomy(root=args.data_root, split='test', target_type1=args.task1)
    task2_test_dst = vision.datasets.Taskonomy(root=args.data_root, split='test', target_type1=args.task2)
    student = vision.models.taskonomy.TaskonomyNetworkStudent(task1=args.task1, task2=args.task2, eval_only=False)
    
    metric_dict={}

    metric_dict['rmse'] = metrics.RootMeanSquaredError(attach_to=lambda o, t: (o ,t), log_scale=True )
    metric_dict['rel'] = metrics.MeanRelativeError(attach_to=lambda o, t: (o ,t) )
    metric_dict['pore125_1'] = metrics.PercentageOfRelativeErrors_125(attach_to=lambda o, t: (o ,t) )
    metric_dict['pore125_2'] = metrics.PercentageOfRelativeErrors_1252(attach_to=lambda o, t: (o ,t) )
    metric_dict['pore125_3'] = metrics.PercentageOfRelativeErrors_1253(attach_to=lambda o, t: (o ,t) )

    full_metric = metrics.MetricCompose(metric_dict=metric_dict)
    
    train_loader = torch.utils.data.DataLoader( train_dst, batch_size=32, shuffle=True, num_workers=0 )
    task1_test_loader = torch.utils.data.DataLoader( task1_test_dst, batch_size=16, shuffle=False, num_workers=0 )
    task2_test_loader = torch.utils.data.DataLoader( task2_test_dst, batch_size=16, shuffle=False, num_workers=0 )

    task1_evaluator = engine.evaluator.BasicEvaluator( task1_test_loader, full_metric, eval_fn=eval_task1_fn, tag=args.task1 )
    task2_evaluator = engine.evaluator.BasicEvaluator( task2_test_loader, full_metric, eval_fn=eval_task2_fn, tag=args.task2 )
    # aircraft_evaluator = engine.evaluator.BasicEvaluator( aircraft_loader, aircraft_metric )
    
    if args.ckpt is not None:
        student.load_state_dict( torch.load( args.ckpt ) )
        print("Load student model from %s"%args.ckpt)
    if args.test_only:
        results_task1 = task1_evaluator.eval( student )
        results_task2 = task2_evaluator.eval( student )
        print("task1: %s"%( results_task1 ))
        print("task2: %s"%( results_task2 ))
        return 
    
    TOTAL_ITERS=len(train_loader) * 100
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    print(torch.cuda.current_device())
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=10e-6)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=TOTAL_ITERS, power=0.9)
    trainer = TaskonomyAmalgamation( 
        logger=utils.logger.get_logger('soka'), 
        tb_writer=SummaryWriter( log_dir='run/soka-%s'%( time.asctime().replace( ' ', '_' ) ) ) 
    )
    
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_STEP(every=10), 
        callbacks=[
            callbacks.MetricsLogging(keys=('total_loss_kd', 'loss_kd1', 'loss_kd2', 'loss_la', 'lr', 'step_time')),
            # callbacks.EvalAndCkpt(model=student, evaluator=edge3d_evaluator, metric_name='edge3d_rmse', ckpt_prefix='cfl_depth')
        ])
        
    trainer.add_callback( 
        engine.DefaultEvents.AFTER_EPOCH, 
        callbacks=[
            callbacks.EvalAndCkpt(model=student, evaluator=task1_evaluator, metric_name='pore125_1', metric_mode='max', ckpt_prefix=args.task1),
            callbacks.EvalAndCkpt(model=student, evaluator=task2_evaluator, metric_name='pore125_1', metric_mode='max', ckpt_prefix=args.task2),
        ] )
    trainer.add_callback(
        engine.DefaultEvents.AFTER_STEP,
        callbacks=callbacks.LRSchedulerCallback(schedulers=[scheduler]))



    trainer.setup( student=student, 
                  dataloader=train_loader,
                  optimizer=optimizer,
                  alpha=1.5,
                  device=device
                    )
    trainer.run(start_iter=0, max_iter=TOTAL_ITERS)

if __name__=='__main__':
    main()
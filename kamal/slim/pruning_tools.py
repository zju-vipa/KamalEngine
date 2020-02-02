import torch
import torch.nn as nn 
import numpy as np
from copy import deepcopy
from hyperopt import fmin, tpe, hp
import pickle, os

from ..core.engine.trainer import train, eval
from ..core.metrics import *

class AutoPruner(object):
    def __init__(self, strategy, task_info, output_dir, train_loader, test_loader=None):
        self.strategy = strategy
        self.criterion = self.get_criterion(task_info)
        self.metrics = self.get_metrics(task_info)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.output_dir = output_dir
        
    def compress(self, model, rate=0.1, seaching=False, **training_kargs):
        ori_num_params = sum( [ torch.numel(p) for p in model.parameters() ] )
        target_score_percentage = training_kargs.get( 'target_score_percentage', None )

        if target_score_percentage is not None:
            (metric_name, score), val_loss = eval(model=model,
                                                criterion=self.criterion, 
                                                test_loader=self.test_loader, 
                                                metric=self.metrics)
            print( "Init score: %s=%.4f"%( metric_name, score ) )
            target_score = score * target_score_percentage
            print("Target score: %.4f"%target_score)
        else:
            target_score = None

        model = deepcopy(model).cpu()
        self.prune( model, rate=rate)
        new_num_params = sum( [ torch.numel(p) for p in model.parameters() ] )
        print( "%d=>%d, %.2f%% params were pruned"%( ori_num_params, new_num_params, 100*(ori_num_params-new_num_params)/ori_num_params ) )
        return self.finetune( model, seaching=seaching, target_score=target_score, **training_kargs)

    def prune(self, model, **kargs):
        return self.strategy( model, **kargs)

    def finetune(self, model, seaching, **training_kargs):
        if seaching==True:
            optimizer = self.search_best_optimizer(model)
        else:
            optimizer = self.get_default_optimizer(model)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training_kargs['total_itrs']//3)

        temp_pth_path = os.path.join(self.output_dir,'temp.pth')
        best_score, best_val_loss = train(model, criterion=self.criterion, 
                     optimizer=optimizer,
                     scheduler=scheduler,
                     train_loader=self.train_loader,
                     test_loader=self.test_loader,
                     metric=self.metrics,
                     pth_path=temp_pth_path, 
                     verbose=True,
                     weights_only=False, **training_kargs)

        model = torch.load(temp_pth_path).cpu()
        return model, best_score, best_val_loss

    def get_default_optimizer(self, model):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        
        return optimizer, scheduler

    def search_best_optimizer( self, model ):

        def objective_fn(space):
            searching_model = deepcopy(model)
            wd = space['wd']
            opt, lr = space['opt']
            if opt=='Adam':
                optimizer = torch.optim.Adam(searching_model.parameters(), lr=lr, weight_decay=wd)
            elif opt=='SGD':
                optimizer = torch.optim.SGD(searching_model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
            best_score, best_val_loss = train(searching_model, 
                    criterion=self.criterion, 
                    optimizer=optimizer,
                    scheduler=None,
                    train_loader=self.train_loader,
                    test_loader=self.test_loader,
                    metric=self.metrics,
                    pth_path=None, total_itrs=500, verbose=True, val_interval=500)
            print(space, best_score, best_val_loss)
            return best_val_loss

        hp_pickle = os.path.join(self.output_dir, "hp.pkl")
        if not os.path.exists(hp_pickle):
            space = {
                'wd': hp.uniform('wd', 0, 1e-3),
                'opt': hp.choice('opt', [
                        ('Adam', hp.uniform('lr_adam', 1e-6, 1e-2) ),
                        ('SGD', hp.uniform('lr_sgd', 1e-6, 0.1) )
                ]),
            }
            best = fmin(fn=objective_fn,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=20, verbose=1)
            with open(hp_pickle, "wb") as f:
                pickle.dump(best,f)
        else: 
            with open(hp_pickle, "rb") as f:
                best = pickle.load( f )
            print("Use the existing hp config: %s"%best)

        wd = best['wd']
        opt = best['opt']

        if opt==0:
            optimizer = torch.optim.Adam(model.parameters(), lr=best['lr_adam'], weight_decay=wd)
        elif opt==1:
            optimizer = torch.optim.SGD(model.parameters(), lr=best['lr_sgd'], weight_decay=wd, momentum=0.9)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=itrs//3)
        return optimizer

    def get_criterion(self, task_info):
        if task_info['name'] in ['classification', 'segmentation']:
            return nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
    
    def get_metrics(self, task_info):
        if task_info['name'] == 'classification':
            return StreamClassificationMetrics()
        elif task_info['name'] == 'segmentation':
            return StreamSegmentationMetrics( num_classes=task_info['num_classes'] )
        
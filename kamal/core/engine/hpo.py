from hyperopt import hp, fmin, tpe
import torch
from copy import deepcopy
from ruamel_yaml import YAML
import os
# optimizer

class HPO(object):
    def __init__(self, trainer, saved_hp=None):
        self.trainer = trainer
        self.saved_hp = saved_hp
        self.init_model = deepcopy(self.trainer.model)

    def optimize( self, max_evals=50, max_iters=200 ):
        
        def objective_fn(space):
            trainer = self.trainer
            trainer.logger.info("[HPO] hp: %s"%space)
            model = deepcopy( self.init_model )
            trainer.model = model
            try:
                self.trainer.reset()
            except: pass 
            trainer.callbacks_on = False
            opt_params = space['opt']
            name = opt_params.pop('type')
            optimizer = self._prepare_optimizer(trainer.model, name, opt_params)
            trainer.optimizer = optimizer # set optimizer

            trainer.train(0, max_iters)

            total_loss = trainer.history.get_scalar('total_loss')
            trainer.logger.info("[HPO] loss: %.4f"%total_loss)
            return total_loss

        if self.saved_hp is not None and os.path.exists(self.saved_hp):
            with open(self.saved_hp, 'r')as f:
                hp = YAML().load( f )
        else:
            hpspace = self._get_space()
            best_hp = fmin(  fn=objective_fn,
                        space=hpspace,
                        algo=tpe.suggest,
                        max_evals=max_evals, 
                        verbose=1)
            hp = dict()
            for k, v in best_hp.items():
                if '-' in k:
                    hp[ k.split('-')[1] ] = float(v)
            hp['opt'] = 'Adam' if best_hp['opt']==0 else 'SGD'
            with open(self.saved_hp, 'w')as f:
                YAML().dump( dict(hp), f )
        name = hp.pop('opt')
        self.trainer.model = self.init_model
        optimizer = self._prepare_optimizer( self.trainer.model, name, hp )
        self.trainer.optimizer = optimizer
        self.trainer.callbacks_on = True
        return hp
        
    def _prepare_optimizer(self, model, name, params):
        if name.lower()=='adam':
            return torch.optim.Adam( model.parameters(), **params)    
        elif name.lower()=='sgd':
            return torch.optim.SGD(model.parameters(), **params)   

    def _get_space(self):
        space = {
                'opt': hp.choice('opt', [
                        { 
                            'type': 'Adam', 
                            'lr': hp.quniform(label='adam-lr', low=1e-5, high=1e-2, q=1e-4),
                            'eps': hp.quniform(label='adam-eps', low=1e-9, high=1e-6, q=1e-8),
                            'weight_decay': hp.quniform('adam-weight_decay', low=0, high=1e-3, q=1e-8),
                        },
                        {
                            'type': 'SGD',
                            'lr': hp.quniform(label='sgd-lr', low=1e-3, high=0.2, q=1e-5),
                            'momentum': hp.quniform(label='sgd-momentum', low=0, high=0.99, q=0.1),
                            'weight_decay': hp.quniform('sgd-weight_decay', low=0, high=1e-3, q=1e-8),
                        }
                ])
            }
        return space
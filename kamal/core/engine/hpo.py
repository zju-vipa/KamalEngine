from hyperopt import hp, fmin, tpe
import torch
from copy import deepcopy
from ruamel_yaml import YAML
import os
# optimizer

class HPO(object):
    def __init__(self, trainer, evaluator, saved_hp=None):
        self.trainer = trainer
        self.evaluator = evaluator
        assert len(self.trainer.callbacks)==0, "HPO should be applied before adding callbacks"
        self.saved_hp = saved_hp
        self._init_model = self.trainer.model

    def optimize( self, max_evals=50, max_iters=200, hpo_space=None, minimize=True):
        
        def objective_fn(space):
            trainer = self.trainer
            trainer.logger.info("[HPO] hp: %s"%space)
            trainer.model = deepcopy( self._init_model ) # point to a copy of init model
            try:
                self.trainer.reset()
            except: pass 
            opt_params = space['opt']
            name = opt_params.pop('type')
            optimizer = self._prepare_optimizer(trainer.model, name, opt_params)
            trainer.optimizer = optimizer # set optimizer

            trainer.train(0, max_iters)

            score = self.evaluator.eval( trainer.model )
            if isinstance(score, dict):
                score = score[ self.evaluator.metrics.PRIMARY_METRIC ]
            trainer.logger.info("[HPO] score: %.4f"%score)
            if minimize:
                return score
            else:
                return -score
        
        self.trainer.callbacks_on = False
        if self.saved_hp is not None and os.path.exists(self.saved_hp):
            with open(self.saved_hp, 'r')as f:
                hp = YAML().load( f )
        else:
            hpspace = self._get_default_space() if hpo_space is None else hpo_space
            best_hp = fmin(  fn=objective_fn,
                            space=hpspace,
                            algo=tpe.suggest,
                            max_evals=max_evals, 
                            verbose=1)
            hp = dict()
            for k, v in best_hp.items():
                if ':' in k:
                    hp[ k.split(':')[1] ] = float(v)
            hp['opt'] = 'Adam' if best_hp['opt']==0 else 'SGD'
            with open(self.saved_hp, 'w')as f:
                YAML().dump( dict(hp), f )
        name = hp.pop('opt')
        self.trainer.model = self._init_model
        optimizer = self._prepare_optimizer( self.trainer.model, name, hp )
        self.trainer.optimizer = optimizer
        self.trainer.callbacks_on = True
        return hp
        
    def _prepare_optimizer(self, model, name, params):
        if name.lower()=='adam':
            return torch.optim.Adam( model.parameters(), **params)    
        elif name.lower()=='sgd':
            return torch.optim.SGD(model.parameters(), **params)   

    def _get_default_space(self):
        space = {
                'opt': hp.choice('opt', [
                        { 
                            'type': 'adam', 
                            'lr':  hp.quniform('adam:lr', 1e-5, 1e-2, 5e-5 ), 
                            'weight_decay': hp.quniform('adam:weight_decay', 0, 1e-3, 1e-5),
                        },
                        {
                            'type': 'sgd',
                            'lr': hp.quniform('sgd:lr', 1e-3, 0.2, 2e-3),
                            'momentum': hp.choice('sgd:momentum', [0.5, 0.9]),
                            'weight_decay': hp.quniform('sgd:weight_decay', 0, 1e-3, 1e-5),
                        }
                ])
            }
        return space
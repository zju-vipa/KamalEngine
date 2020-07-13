from typing import Tuple

class MultiTaskingHelpers(object):
    @staticmethod
    def get_mapped_loss_fn( mapping_tuple: Tuple[int, str, callable,  ] ):
        
        def wrapper(engine, batch):
            inputs, *targets = batch
            outputs = engine.model(inputs)
            loss_dict = {}
            for loss_fn, idx in mapping_tuple:
                loss_dict.update( loss_fn( engine, [outputs[], targets] ) )


from .base import Callback
import numbers
from tqdm import tqdm

class MetricsLogging(Callback):
    def __init__(self, keys):
        super(MetricsLogging, self).__init__()
        self._keys = keys

    def __call__(self, engine):
        if engine.logger==None:
            return
        state = engine.state
        content = "Iter %d/%d (Epoch %d/%d, Batch %d/%d)"%(
            state.iter, state.max_iter, 
            state.current_epoch, state.max_epoch, 
            state.current_batch_index, state.max_batch_index
        )
        for key in self._keys:
            value = state.metrics.get(key, None)
            if value is not None:
                if isinstance(value, numbers.Number):
                    content += " %s=%.4f"%(key, value)
                    if engine.tb_writer is not None:
                        engine.tb_writer.add_scalar(key, value, global_step=state.iter)
                elif isinstance(value, (list, tuple)):
                    content += " %s=%s"%(key, value)
                
        engine.logger.info(content)
    
class ProgressCallback(Callback):
    def __init__(self, max_iter=100, tag=None):
        self._max_iter = max_iter
        self._tag = tag
        #self._pbar = tqdm(total=self._max_iter, desc=self._tag)
    
    def __call__(self, engine):
        self._pbar.update(1)
        if self._pbar.n==self._max_iter:
            self._pbar.close()
    
    def reset(self):
        self._pbar = tqdm(total=self._max_iter, desc=self._tag)
    
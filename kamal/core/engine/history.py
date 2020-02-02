from collections import defaultdict
import numpy as np 

class HistoryBuffer(object):
    def __init__(self, max_length:int=10000):
        self._max_length = max_length
        self._data = [  ]
        self._count = 0
        self._global_avg = 0

    def update(self, value, iteration):
        if len(self._data) == self._max_length:
            self._data.pop(0)
        self._data.append( (value, iteration) )
        self._count+=1
        self._global_avg += (value-self._global_avg) / self._count
    
    def latest(self):
        return self._data[-1][0]
    
    def median(self, window_size: int):
        """
        Return the median of the latest `window_size` values in the buffer.
        """
        return np.median([x[0] for x in self._data[-window_size:]])
    
    def avg(self, window_size: int):
        return np.mean( [x[0] for x in self._data[-window_size:]] )
    
    def global_avg(self):
        return self.global_avg()

    def values(self):
        return self._data

class HistoryStorage(object):
    def __init__(self, start_iter=0):
        self._history = defaultdict( HistoryBuffer )

        self._iter = start_iter
        self._latest_scalars = dict()
        self._vis_data = list()
        
    def put_scalar(self, name, value):
        self._history[ name ].update( float(value), self._iter )
        self._latest_scalars[ name ] = float(value)

    def put_scalars(self, **kwargs):
        for k, v in kwargs.items():
            self.put_scalar(k, v)

    def put_image(self, img_name, img_tensor):
        self._vis_data.append((img_name, img_tensor, self._iter))

    def put_images( self, **kwargs ):
        for k, v in kwargs.items():
            self.put_image( k, v )
    
    def clear_images(self):
        self._vis_data.clear()

    def step(self):
        self._iter+=1
        self._latest_scalars.clear()

    def history(self, name):
        return self._history.get(name, None)
    
    def all_histories(self):
        return self._history

    def latest(self):
        return self._latest_scalars

    def get_scalar(self, name, window_size=None):
        if window_size is not None:
            return self._history[name].avg( window_size=window_size )
        else:
            return self._history[name].latest()
    
    @property
    def vis_data(self):
        return self._vis_data

    @property
    def iter(self):
        return self._iter



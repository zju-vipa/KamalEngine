from collections import defaultdict
import numpy as np 

class HistoryBuffer(object):
    def __init__(self, max_length:int=10000):
        self._max_length = max_length
        self._data = [ ]
        self._count = 0
        self._global_avg = 0

    def update(self, value, iteration):
        if len(self._data) == self._max_length:
            self._data = self._data[self._max_length//2:] # remove old records
        self._data.append( (iteration, value) )
        self._count+=1
    
    def latest(self):
        assert len(self._data)>0
        return self._data[-1][1]
    
    def max(self):
        assert len(self._data)>0
        return np.max( [x[1] for x in self._data] )
    
    def min(self):
        assert len(self._data)>0
        return np.min( [x[1] for x in self._data] )

    def avg(self):
        assert len(self._data)>0
        return np.mean( [x[1] for x in self._data] )
    

class History(object):
    def __init__(self, start_iter=0):
        self._history = defaultdict( HistoryBuffer )
        self._vis_data = dict()

        self._iter = start_iter
        
    def put_scalar(self, name, value):
        self._history[ name ].update( float(value), self._iter )

    def put_scalars(self, **kwargs):
        for k, v in kwargs.items():
            self.put_scalar(k, v)

    def put_image(self, img_name, img_tensor):
        self._vis_data[img_name] = img_tensor

    def put_images( self, **kwargs ):
        for k, v in kwargs.items():
            self.put_image( k, v )
    
    def reset(self):
        self._history = defaultdict( HistoryBuffer )
        self._vis_data = dict()

    def step(self):
        self._iter+=1
    
    def history(self, name):
        return self._history.get(name, None)
    
    def all_histories(self):
        return self._history

    def get_scalar(self, name):
        return self._history[name].latest()

    def get_image(self, name):
        return self._vis_data[name]

    @property
    def vis_data(self):
        return self._vis_data

    @property
    def iter(self):
        return self._iter



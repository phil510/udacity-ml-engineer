import abc
import numpy as np

class Space(abc.ABC):
    def __init__(self):
        self.shape = None
        self.seed()
    
    @abc.abstractmethod
    def __contains__(self, item):
        pass
    
    def sample(self):
        raise NotImplementedError
        
    def seed(self, seed = None):
        self._rng = np.random.RandomState(seed)

class TradeSpace(Space):
    def __init__(self, n_securities):
        super().__init__()
        
        self._n = n_securities
        self.shape = (n_securities, )
        
    def __contains__(self, item):
        try:
            item = np.asarray(item)
        except:
            return False
            
        right_shape = (item.shape == self.shape)
        all_int = (item.dtype.kind in np.typecodes['AllInteger'])
        
        return (right_shape and all_int)
    
    def sample(self):
        raise NotImplementedError
        
class MarketSpace(Space):
    def __init__(self, n_securities, n_datapoints):
        super().__init__()
        
        self._n = n_securities
        self._m = n_datapoints
        self.shape = (1 + n_securities * 2 + n_datapoints, )
        
    def __contains__(self, item):
        try:
            item = np.asarray(item)
        except:
            return False
            
        right_shape = (item.shape == self.shape)
        all_int = ((item.dtype.kind in np.typecodes['AllInteger']) 
                   or (item.dtype.kind in np.typecodes['AllFloat']))
        
        return (right_shape and all_int)
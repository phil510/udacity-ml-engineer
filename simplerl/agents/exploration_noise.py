import abc
import numpy as np

class RandomProcess(abc.ABC):
    def __init__(self):
        self.seed()
        
    @abc.abstractmethod
    def sample(self):
        pass
    
    def seed(self, seed = None):
        self._rng = np.random.RandomState(seed)

class OrnsteinUhlenbeckProcess(RandomProcess):
    '''
    from https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl
    '''
    def __init__(self, size, std, theta = 0.15, dt = 1e-2, x0 = None):
        self.theta = theta
        self.mu = 0.0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()
        
        super().__init__()

    def sample(self):
        x = (self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt
             + self.std() * np.sqrt(self.dt) * self._rng.randn(*self.size))
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
        
class GaussianProcess(RandomProcess):
    def __init__(self, size, std):
        self.size = size
        self.std = std
        
        super().__init__()

    def sample(self):
        return self._rng.randn(*self.size) * self.std()
        
class BinaryNoise(RandomProcess):
    def __init__(self, size, prob):
        self.size = size
        self.prob = prob
        
        super().__init__()
        
    def sample(self):
        x = self._rng.rand(*self.size)
        x = (x > self.prob()) * 2 - 1
        
        return x
import abc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import copy

from .parallel_envs import ParallelEnvironment, VectorEnvWrapper
from .replay_buffers import (ReplayBuffer, 
                             TrajectoryBuffer, 
                             PrioritizedReplayBuffer)

class Agent(abc.ABC):
    @abc.abstractmethod
    def action(self, obs):
        pass
    
    @abc.abstractmethod
    def update(self, obs, action, reward, next_obs, terminal):
        pass
        
class RandomAgent(Agent):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space
        self.seed()
        
    @property
    def training(self):
        return False
        
    def action(self, obs):
        # TODO just make this return action_space.sample()
        return self._rng.uniform(-1, 1, self.action_space.shape)
        
    def update(self, obs, action, reward, next_obs, terminal):
        pass
        
    def seed(self, seed = None):
        self._rng = np.random.RandomState(seed)

class LearningAgent(Agent):
    def __init__(self, env_fn = None, 
                 model_fn = None, 
                 n_actors = 1):
        
        assert (callable(model_fn)), 'TODO'
        assert (n_actors > 0), 'TODO'
        assert (callable(env_fn)), 'TODO'
        
        self._model_fn = model_fn
        self._env_fn = env_fn
        self._n_actors = n_actors
        
        self._torch_objs = {}
        self._device = torch.device('cuda' if torch.cuda.is_available()
                                    else 'cpu')
        
        self.reset_current_step()
        
    @abc.abstractmethod
    def action(self, obs):
        pass
    
    @abc.abstractmethod
    def update(self, obs, action, reward, next_obs, terminal):
        pass
    
    @property
    def env(self):
        return self._env
    
    @property
    def training(self):
        return self._training
        
    def train(self):
        self._training = True
        self.open_env()
        
    def eval(self):
        self._training = False
        self.open_env()
        
    def open_env(self):
        self.close_env()
        if self.training and (self._n_actors > 1):
            self._env = ParallelEnvironment([self._env_fn for _ 
                                             in range(self._n_actors)])
        else:
            self._env = VectorEnvWrapper(self._env_fn()) 
    
    def close_env(self):
        try:
            self._env.close()
        except AttributeError:
            pass
            
    def reset_current_step(self):
        self.current_step = 0
    
    def to(self, device):
        self._device = device
        for obj in self._torch_objs.values():
            try:
                obj.to(self._device) # optimizer objects don't have a to method
            except AttributeError:
                pass
                
    def register_torch_obj(self, obj, obj_name):
        # make sure that the object has a state_dict method for saving
        assert (hasattr(obj, 'state_dict')), 'TODO'
        assert (callable(getattr(obj, 'state_dict'))), 'TODO'
        
        self._torch_objs[obj_name] = obj
    
    def save(self, directory_path, **kwargs):
        torch_path = os.path.join(directory_path, 'torch_objs.tar')
        state_dicts = {obj_name: obj.state_dict() for obj_name, obj 
                       in self._torch_objs.items()}
        
        torch.save(state_dicts, torch_path)
        
    def load(self, directory_path):
        torch_path = os.path.join(directory_path, 'torch_objs.tar')
        model = torch.load(torch_path, map_location = self._device)
        
        for obj_name, state_dict in model.items():
            self._torch_objs[obj_name].load_state_dict(state_dict)
            
    def seed(self, seed = None):
        self._rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        
    def ready_to_update(self):
        return True
        
class ExplorationNoiseMixin(object):
    def __init__(self, *args, noise_fn = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert ((noise_fn is None) or callable(noise_fn)), 'TODO'
        if noise_fn is not None:
            self.exploration_noise = [noise_fn() for _ 
                                      in range(self._n_actors)]
        else:
            self.exploration_noise = None
        
    def generate_noise(self):
        if self.exploration_noise is not None:
            noise = [process.sample() for process in self.exploration_noise]
            noise = np.stack(noise)
        else:
            noise = None
            
        return noise
    
    def train(self):
        super().train()
        self._reset_states()
            
    def eval(self):
        super().eval()
        self._reset_states()
        
    def _reset_states(self):
        try:
            for process in self.exploration_noise:
                process.reset_states()
        except AttributeError:
            pass
    
class ReplayBufferMixin(object):
    def __init__(self, *args, 
                 replay_memory = 100000,
                 use_per = False,
                 alpha = 0.4,
                 beta = lambda: 0.6,
                 replay_start = 1000,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.replay_start = replay_start
        self._use_per = use_per
        
        if self._use_per:
            assert (callable(beta)), 'TODO'
            self.replay_buffer = PrioritizedReplayBuffer(replay_memory, 
                                                         alpha = alpha)
            self.alpha = alpha
            self.beta = beta
        else:
            self.replay_buffer = ReplayBuffer(replay_memory)
        
    def add_to_memory(self, *args, **kwargs):
        raise NotImplementedError
    
    def sample_from_memory(self, batch_size):
        if self._use_per:
            samples = self.replay_buffer.sample(batch_size, beta = self.beta())
            experiences, is_weights, indices = zip(*samples)
        else:
            experiences = self.replay_buffer.sample(batch_size)    
            is_weights = None
            indices = None
            
        obs, action, reward, next_obs, terminal = zip(*experiences)
        
        obs = torch.stack(obs, dim = 0)
        action = torch.stack(action, dim = 0)
        reward = torch.stack(reward, dim = 0)
        next_obs = torch.stack(next_obs, dim = 0)
        terminal = torch.stack(terminal, dim = 0)
        
        return obs, action, reward, next_obs, terminal, is_weights, indices
        
    def update_priorities(self, indices, priorities):
        if self._use_per:
            self.replay_buffer.update_priorities(indices, priorities)
        else:
            pass
    
    def save(self, directory_path, save_memory = False, **kwargs):
        super().save(directory_path, **kwargs)
        
        if save_memory:
            memory_path = os.path.join(directory_path, 'replay_memory.pt')
            torch.save(self.replay_buffer._memory, memory_path)
            
    def load(self, directory_path):
        super().load(directory_path)
        
        memory_path = os.path.join(directory_path, 'replay_memory.pt')
        if os.path.isfile(memory_path):
            memory = torch.load(memory_path)
            self.replay_memory._memory = memory
            
class LocalBufferMixin(object):
    def __init__(self, *args, trajectory_len = 1, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert (trajectory_len >= 1), 'TODO'
        self._trajectory_len = trajectory_len
        
    def add_to_local_buffer(self, *args, **kwargs):
        raise NotImplementedError
    
    def reset_local_buffer(self):
        if self._trajectory_len > 1:
            n = self._n_actors if self.training else 1
            self.local_buffer = [TrajectoryBuffer(self._trajectory_len) 
                                 for _ in range(n)]
                             
    def train(self):
        super().train()
        self.reset_local_buffer()
        
    def eval(self):
        super().eval()
        self.reset_local_buffer()
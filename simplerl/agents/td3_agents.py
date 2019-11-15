import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .agents import (LearningAgent, 
                     ExplorationNoiseMixin, 
                     ReplayBufferMixin,
                     LocalBufferMixin)
from .torch_utils import calc_n_step_returns, to_tensors

class TD3Agent(ExplorationNoiseMixin, LocalBufferMixin,
               ReplayBufferMixin, LearningAgent):
    
    def __init__(self, env_fn = None, 
                 model_fn = None,
                 n_actors = 1,
                 action_range = 1.0,
                 gamma = 0.99,
                 exploration_noise = None,
                 regularization_noise = None,
                 noise_clip = (-0.5, 0.5),
                 batch_size = 64,
                 n_steps = 1,
                 replay_memory = 100000,
                 use_per = False,
                 alpha = 0.6,
                 beta = lambda: 0.4,
                 replay_start = 1000,
                 parameter_tau = 1e-3,
                 buffer_tau = 1e-3,
                 optimizer = optim.Adam,
                 policy_learning_rate = 1e-4,
                 critic_learning_rate = 1e-3,
                 weight_decay = 1e-4,
                 clip_gradients = None,
                 update_freq = 1,
                 policy_update_freq = 3):
        
        super().__init__(env_fn = env_fn,
                         model_fn = model_fn,
                         n_actors = n_actors,
                         noise_fn = exploration_noise,
                         replay_memory = replay_memory,
                         use_per = use_per,
                         alpha = alpha,
                         beta = beta,
                         replay_start = replay_start,
                         trajectory_len = n_steps)
        
        # create online and target networks
        self.online_network = self._model_fn()
        self.target_network = self._model_fn()
        
        assert (hasattr(self.online_network, 'action')), 'TODO'
        assert (hasattr(self.online_network, 'critic_value')), 'TODO'
        assert (hasattr(self.online_network, 'critic_value_1')), 'TODO'
        assert (hasattr(self.online_network, 'policy_params')), 'TODO'
        assert (hasattr(self.online_network, 'critic_params_1')), 'TODO'
        assert (hasattr(self.online_network, 'critic_params_2')), 'TODO'
        
        self.register_torch_obj(self.online_network, 'online_network')
        self.register_torch_obj(self.target_network, 'target_network')
        self.online_network.eval()
        self.target_network.eval()
        
        # create the optimizers for the online_network
        self.policy_optimizer = optimizer(self.online_network.policy_params, 
                                          lr = policy_learning_rate,
                                          weight_decay = weight_decay)
        self.critic_optimizer_1 = optimizer(self.online_network.critic_params_1, 
                                            lr = critic_learning_rate,
                                            weight_decay = weight_decay)
        self.critic_optimizer_2 = optimizer(self.online_network.critic_params_2, 
                                            lr = critic_learning_rate,
                                            weight_decay = weight_decay)
        self.clip_gradients = clip_gradients
        self.register_torch_obj(self.policy_optimizer, 'policy_optimizer')
        self.register_torch_obj(self.critic_optimizer_2, 'critic_optimizer_1')
        self.register_torch_obj(self.critic_optimizer_2, 'critic_optimizer_2')
        
        self.regularization_noise = regularization_noise
        
        self.gamma = gamma
        self.action_range = action_range
        self.batch_size = batch_size
        self.parameter_tau = parameter_tau
        self.buffer_tau = buffer_tau
        self.update_freq = update_freq
        self.policy_update_freq = policy_update_freq
        self.noise_clip = noise_clip
        
        self.eval()
        
    @property 
    def n_steps(self):
        return self._trajectory_len
        
    def soft_update(self):
        for target, online in zip(self.target_network.parameters(), 
                                  self.online_network.parameters()):
            target.detach_()
            target.copy_(target * (1.0 - self.parameter_tau) 
                         + online * self.parameter_tau)
        
        # this is for things like batch norm and other PyTorch objects
        # that have buffers and/instead of learnable parameters
        for target, online in zip(self.target_network.buffers(), 
                                  self.online_network.buffers()):
            # detach is probably unnecessary since buffers are not learnable
            target.detach_() 
            target.copy_(target * (1.0 - self.buffer_tau) 
                         + online * self.buffer_tau)
            
    def generate_reg_noise(self):
        if self.regularization_noise is not None:
            noise = self.regularization_noise.sample()
            assert (noise.shape[0] == self.batch_size), 'TODO'
            noise = np.clip(noise, *self.noise_clip)
        else:
            noise = 0.0
        
        noise = torch.as_tensor(noise, dtype = torch.float32, 
                                device = self._device)
        
        return noise
            
    def ready_to_update(self):
        return ((self.current_step >= self.replay_start) and
                (self.current_step % self.update_freq == 0) and
                self.training)
                
    def action(self, obs):
        obs = torch.as_tensor(obs, dtype = torch.float32,
                              device = self._device)
        
        with torch.no_grad():
            action = self.online_network.action(obs).cpu().numpy()
        
        if (self.exploration_noise is not None) and (self.training):
            noise = self.generate_noise()
            assert (noise.shape == action.shape), 'TODO'
            action += noise
            action = np.clip(action, -self.action_range, self.action_range)
        
        return action
        
    def update_target(self, obs, action, reward, next_obs, terminal):
        with torch.no_grad():
            if self.n_steps > 1:
                Q_sa_1 = []
                Q_sa_2 = []
                for i in range(self.n_steps):
                    next_action = self.target_network.action(next_obs[:, i])
                    noise = self.generate_reg_noise()
                    if self.regularization_noise is not None:
                        assert (noise.shape == next_action.shape), 'TODO'
                    next_action += noise
                    Q_1, Q_2 = self.target_network.critic_value(next_obs[:, i], 
                                                                next_action)
                    Q_sa_1.append(Q_1)
                    Q_sa_2.append(Q_2)
                
                Q_sa_1 = torch.stack(Q_sa_1, dim = 1)
                Q_sa_2 = torch.stack(Q_sa_2, dim = 1)
                
            else:
                next_action = self.target_network.action(next_obs)
                noise = self.generate_reg_noise()
                if self.regularization_noise is not None:
                    assert (noise.shape == next_action.shape), 'TODO'
                next_action += noise
                Q_sa_1, Q_sa_2 = self.target_network.critic_value(next_obs, 
                                                                  next_action)
        
        Q_sa_next = torch.min(Q_sa_1, Q_sa_2)
        
        # check if reward is 1 dimensional, if it is, we need to 
        # insert the seq_len dimension at dim 1 for the calc_n_step_returns
        # function
        if (reward.dim() == 1):
            reward = reward.unsqueeze(1)
            terminal = terminal.unsqueeze(1)
            Q_sa_next = Q_sa_next.unsqueeze(1)
            
        update_target = calc_n_step_returns(reward, Q_sa_next, terminal,
                                            gamma = self.gamma, 
                                            n_steps = self.n_steps,
                                            seq_model = False)
        update_target = update_target.squeeze(1)
        
        assert (update_target.shape[0] == obs.shape[0]), 'TODO'
        assert (not update_target.requires_grad), 'TODO'

        return update_target
        
    def add_to_memory(self, obs, action, reward, next_obs, terminal,
                      *args, **kwargs): 
        assert (obs.shape[0] == self._n_actors), 'TODO'
        if (not hasattr(self, 'local_buffer')):
            for i in range(self._n_actors):
                experience = to_tensors(obs[i], action[i], reward[i], 
                                        next_obs[i], terminal[i],
                                        dtype = torch.float32, 
                                        device = self._device)
                self.replay_buffer.add(experience)
            
        else:
            for i, buffer in enumerate(self.local_buffer):
                if (len(buffer) == self.n_steps):
                    trajectory = buffer.get_trajectory()
                    experience = to_tensors(*trajectory,
                                            dtype = torch.float32,
                                            device = self._device)
                    self.replay_buffer.add(experience)
                    buffer.reset()
                
                elif terminal[i]:
                    length = len(buffer)
                    trajectory = buffer.get_trajectory()
                    for item in trajectory:
                        pad = np.zeros(np.asarray(item[0]).shape, 
                                       dtype = np.asarray(item[0]).dtype)
                        for _ in range(self.n_steps - length):
                            item.append(pad)
                    
                    experience = to_tensors(*trajectory,
                                            dtype = torch.float32,
                                            device = self._device)
                    self.replay_buffer.add(experience)
                    buffer.reset()
                    
    def add_to_local_buffer(self, obs, action, reward, next_obs, terminal,
                            *args, **kwargs): 
        assert (len(self.local_buffer) == obs.shape[0]), 'TODO'
        for i, buffer in enumerate(self.local_buffer):
            buffer.add((obs[i], action[i], reward[i], 
                        next_obs[i], terminal[i]))
        
    def update(self, obs, action, reward, next_obs, terminal):        
        if self.training:
            if hasattr(self, 'local_buffer'):
                self.add_to_local_buffer(obs, action, reward, 
                                         next_obs, terminal)
            self.add_to_memory(obs, action, reward, next_obs, terminal)
        
        for i, done in enumerate(terminal):
            if done:
                try:
                    self.local_buffer[i].reset()
                except AttributeError:
                    pass
                if self.training:
                    try:
                        self.exploration_noise[i].reset_states()
                    except AttributeError:
                        pass
            
        if self.ready_to_update():
            self.online_network.eval()
            
            # Maybe not the best way to do a line break, 
            # but I like it more than \
            (obs, action, reward, next_obs, terminal, 
             weight, indices) = self.sample_from_memory(self.batch_size)

            update_target = self.update_target(obs, action, reward, 
                                               next_obs, terminal)
            
            self.online_network.train()
            
            if self.n_steps > 1:
                obs = obs[:, 0]
                action = action[:, 0]
            
            if weight is None:
                weight = torch.ones(self.batch_size, device = self._device)
            else:
                weight = torch.as_tensor(weight, dtype = torch.float32, 
                                         device = self._device)
                assert (weight.dim() == 1), 'TODO'
            
            Q_sa_1, Q_sa_2 = self.online_network.critic_value(obs, action)
            
            Qs = [Q_sa_1, Q_sa_2]
            optimizers = [self.critic_optimizer_1, self.critic_optimizer_2]
            params = [self.online_network.critic_params_1,
                      self.online_network.critic_params_2]
            
            for Q_sa, optimizer, param in zip(Qs, optimizers, params):
                td_error = Q_sa - update_target
                critic_loss = (td_error).pow(2).mul(0.5).squeeze(-1)
                critic_loss = (critic_loss * weight).mean()

                optimizer.zero_grad()
                critic_loss.backward()
                if self.clip_gradients:
                    nn.utils.clip_grad_norm_(param, self.clip_gradients)
                optimizer.step()
            
            if (self.current_step % self.policy_update_freq == 0):
                action = self.online_network.action(obs)
                policy_loss = -self.online_network.critic_value_1(obs, action)
                policy_loss = policy_loss.mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                if self.clip_gradients:
                    nn.utils.clip_grad_norm_(self.online_network.policy_params,
                                             self.clip_gradients)
                self.policy_optimizer.step()
                
                self.soft_update()
            
            self.online_network.eval()
            
            # update the priorities using the td errors from the 2nd critic
            updated_p = (np.abs(td_error.detach().cpu().numpy().squeeze()) 
                         + 1e-8)
            self.update_priorities(indices, updated_p)
        
        if self.training:
            self.current_step += 1
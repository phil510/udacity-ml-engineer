import abc
import numpy as np
import os
import sys
from functools import partial
from collections import defaultdict

class TrainTestHook(abc.ABC):
    @abc.abstractmethod
    def begin(self, env, agent):
        pass
            
    @abc.abstractmethod
    def reset(self, env, agent):
        pass
        
    @abc.abstractmethod
    def step(self, env, agent, obs, action, reward, next_obs, terminal):
        pass
        
    @abc.abstractmethod
    def end(self, env, agent):
        pass

class SaverHook(TrainTestHook):
    def __init__(self, save_directory, save_every = None):
        self._save_directory = save_directory
        self._save_every = save_every
        self._began = False
        
    def begin(self, env, agent):
        self.current_step = 0
        if not os.path.exists(self._save_directory):
            os.mkdir(self._save_directory)
        self._began = True
            
    def reset(self, env, agent):
        pass
        
    def step(self, env, agent, obs, action, reward, next_obs, terminal):
        assert (self._began), 'TODO'
        self.current_step += 1
        
        if self._save_every and (self.current_step % self._save_every == 0):
            self._save_agent(agent)
        
    def end(self, env, agent):
        assert (self._began), 'TODO'
        self._save_agent(agent)
        
    def _save_agent(self, agent):
        model_dir = os.path.join(self._save_directory, 
                                 'model_{}'.format(self.current_step))
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        
        agent.save(model_dir)
        print('Model saved to {}'.format(model_dir))

class GradientMonitorHook(TrainTestHook):
    def __init__(self):
        self.gradients = None
        self.current_step = None
        self._began = False
    
    def begin(self, env, agent):
        self.gradients = defaultdict(list)
        self.current_step = 0
        self._handles = []
        
        def gradient_norm_hook(gradient, param_name):
            self.gradients[param_name].append((self.current_step, gradient.norm()))
        
        for obj_name, obj in agent._torch_objs.items():
            if obj_name == 'target_network':
                continue
                
            try:
                for name, param in obj.named_parameters():
                    param.register_hook(partial(gradient_norm_hook,
                                                param_name = name))
            except AttributeError: 
                pass
                
        self._began = True
    
    def reset(self, env, agent):
        pass
        
    def step(self, env, agent, obs, action, reward, next_obs, terminal):
        assert (self._began), 'TODO'
        self.current_step += 1
    
    def end(self, env, agent):
        assert (self._began), 'TODO'
        for handle in self._handles:
            handle.remove()
            
    def plot_gradients(self, parameters = None):
        assert (self.gradients is not None), 'TODO'
        
        if parameters is None:
            parameters = (self.gradients.keys())
        
        raise NotImplementedError

class GymMonitorHook(TrainTestHook):
    def __init__(self, eval_window = 100, verbose = 10, vector_env = False):
        self._eval_window = eval_window
        self._verbose = verbose
        self._vector_env = vector_env
        
        self.episode = None
        self.archive = []
        
    def begin(self, env, agent):
        self.episode = 0
        self.scores = []
            
    def reset(self, env, agent):
        assert (self.episode is not None), 'TODO'
        if self.episode == 0:
            self.time_step = 0
            self.total_reward = 0
            self.episode += 1
            return
        
        self.scores.append(self.total_reward)
        avg_score = np.mean(self.scores[-self._eval_window: ])
            
        if self._verbose and (self.episode % self._verbose == 0):
            print('\rEpisode {}'.format(self.episode)
                  + ' | Time Steps: {}'.format(self.time_step)
                  + ' | Average Score: {:.2f}'.format(avg_score))
        
        self.time_step = 0
        self.total_reward = 0
        self.episode += 1
        
    def step(self, env, agent, obs, action, reward, next_obs, terminal):
        assert (self.episode is not None), 'TODO'
        if self._vector_env:
            reward = reward[0]
            
        self.total_reward += reward
        self.time_step += 1
        
    def end(self, env, agent):
        assert (self.episode is not None), 'TODO'
        if self._verbose and (self.time_step != 0):
            avg_score = np.mean(self.scores[-self._eval_window: ])
            print('\rEpisode {}'.format(self.episode)
                  + ' | Time Steps: {}'.format(self.time_step)
                  + ' | Average Score: {:.2f}'.format(avg_score))
        
        self.archive.append(self.scores)
        self.episode = None
    
class MarketMonitorHook(TrainTestHook):
    def __init__(self, eval_window = 100, eval_start = 1, 
                 verbose = 10, vector_env = False):
        self._eval_window = eval_window
        self._eval_start = eval_start
        self._vector_env = vector_env
        self._verbose = verbose
        
        self.episode = None
        self.archive = []
        
    def begin(self, env, agent):
        self._beginning_cash = env.env_spec['beginning_cash']
        self._n = env.env_spec['n_tradable']
        self.episode = 0
        self.market_returns = []
        self.agent_returns = []
        self.agent_wins = []
            
    def reset(self, env, agent):
        assert (self.episode is not None), 'TODO'
        
        try:
            simulation = env._simulation
        except AttributeError:
            simulation = env.get_env_attr('_simulation')[0]
            
        market_return = ((simulation[-1, :self._n] 
                          / simulation[self._eval_start - 1, :self._n] 
                          - 1.0).mean() * 100.0)
        self.market_returns.append(market_return)
        
        if self.episode == 0:
            self.time_step = 0
            self.total_reward = 0
            self.episode += 1
            return
        
        agent_return = (self.total_reward / self._beginning_cash) * 100.0
        self.agent_returns.append(agent_return)
        
        # get the second to last market return since we add the current market
        # return at the beginning of each episode
        won = (agent_return > self.market_returns[-2])
        self.agent_wins.append(won)
        
        spread = (np.asarray(self.agent_returns[-self._eval_window:])
                  - np.asarray(self.market_returns[-self._eval_window - 1:-1]))
        avg_spread = spread.mean()
        
        win_rate = (sum(self.agent_wins[-self._eval_window:]) 
                    / len(self.agent_wins[-self._eval_window:])) * 100.0
        
        if self._verbose and (self.episode % self._verbose == 0):
            print('\rEpisode {}'.format(self.episode)
                  + ' | Market Return: {:.2f}'.format(self.market_returns[-2])
                  + ' | Agent Return: {:.2f}'.format(self.agent_returns[-1])
                  + ' | Win Rate: {:.2f}'.format(win_rate)
                  + ' | Average Spread {:.2f}'.format(avg_spread))
        
        self.time_step = 0
        self.total_reward = 0
        self.episode += 1
        
    def step(self, env, agent, obs, action, reward, next_obs, terminal):
        assert (self.episode is not None), 'TODO'
        if self._vector_env:
            reward = reward[0]
            
        self.total_reward += reward
        self.time_step += 1
        
    def end(self, env, agent):
        assert (self.episode is not None), 'TODO'
        
        spread = (np.asarray(self.agent_returns[-self._eval_window:])
                  - np.asarray(self.market_returns[-self._eval_window - 1:-1]))
        avg_spread = spread.mean()
        
        win_rate = (sum(self.agent_wins[-self._eval_window:]) 
                    / len(self.agent_wins[-self._eval_window:])) * 100.0
        
        if self._verbose and (self.time_step != 0):
            print('\rEpisode {}'.format(self.episode)
                  + ' | Market Return: {:.2f}'.format(self.market_returns[-2])
                  + ' | Agent Return: {:.2f}'.format(self.agent_returns[-1])
                  + ' | Win Rate: {:.2f}'.format(win_rate)
                  + ' | Average Spread {:.2f}'.format(avg_spread))
        
        if len(self.market_returns) != len(self.agent_returns):
            self.market_returns = self.market_returns[:-1]
        
        self.archive.append((self.market_returns,
                             self.agent_returns,
                             self.agent_wins))
        self.episode = None
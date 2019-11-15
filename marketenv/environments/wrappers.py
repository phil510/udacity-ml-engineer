import abc
import numpy as np
from collections import deque

from ..common.utils import (get_current_prices,
                            get_positions, 
                            get_cash_balance,
                            get_portfolio_value,
                            get_sales, 
                            get_purchases)

class Wrapper(object):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range
           
    def step(self, action):
        return self.env.step(action)
        
    def reset(self):
        return self.env.reset()
        
    def close(self):
        return self.env.close()
    
    def seed(self, seed = None):
        return self.env.seed(seed = seed)
        
    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.env.__str__())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
    
    # this allows us to only have to override methods/attributes
    def __getattr__(self, attr):
        return getattr(self.env, attr)
    
class MarketMonitor(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset_histories()
        
    def reset(self):
        if self.obs_history is not None:
            episode = (self.obs_history, self.action_history,
                       self.reward_history, self.value_history,
                       self.info_history)
            self.episode_history.append(episode)
    
        self.obs_history = []
        self.action_history = []
        self.reward_history = []
        self.value_history = []
        self.info_history = []
        
        obs = self.env.reset()
        self.obs_history.append(obs)
        
        return obs
        
    def step(self, action):
        obs, reward, terminal, info = self.env.step(action)
        
        self.obs_history.append(obs)
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.value_history.append(self.portfolio_value)
        self.info_history.append(info)
        
        return obs, reward, terminal, info
        
    def reset_histories(self):
        self.episode_history = []
        self.obs_history = None
        self.action_history = None
        self.reward_history = None
        self.value_history = None
        self.info_history = None
        
class WarmResetWrapper(Wrapper):
    def __init__(self, env, warm_up = 0):
        super().__init__(env)
        self.warm_up = warm_up
        
    def reset(self):
        obs = self.env.reset()
        zero_action = np.zeros(self.action_space.shape)
        for _ in range(self.warm_up):
            obs, _, _, _ = self.env.step(zero_action)
            
        return obs

class TradeActionWrapper(Wrapper):
    def __init__(self, env, action_scale = 1.0):
        super().__init__(env)
        self.action_scale = action_scale
        
    def step(self, action):
        action = np.asarray(action)
        action = action * self.action_scale
        action = self.clip_action(action, self.env.observation)
        
        return self.env.step(action)
        
    def clip_action(self, raw_action, obs):
        cash = obs[0]
        positions = obs[1: self.env._n + 1].astype(int)
        prices = obs[(self.env._n + 1): (self.env._n * 2 + 1)]
        
        action = np.round(raw_action).astype(int)
        sale = get_sales(action)
        purchase = get_purchases(action)

        valid_sale = np.minimum(sale, positions)
        sale_commission = self.env._commission * sum(valid_sale > 0)
        
        updated_cash_bal = cash + np.dot(valid_sale, prices) - sale_commission
        if updated_cash_bal < 0.0:
            valid_sale = np.zeros(action.shape[0], dtype = int)
            updated_cash_bal = cash
            
        purchase_commission = self.env._commission * sum(purchase > 0)
        total_purchase_price = np.dot(purchase, prices) + purchase_commission
        
        if total_purchase_price > updated_cash_bal:
            cash_available = max(updated_cash_bal - purchase_commission, 0.0)
            
            cash_split = ((purchase * prices) / np.dot(purchase, prices) 
                          * cash_available)
            valid_purchase = np.floor(cash_split / prices).astype(int)
            
            purchase_commission = (self.env._commission 
                                   * sum(valid_purchase > 0))
            total_purchase_price = (np.dot(valid_purchase, prices) 
                                   + purchase_commission)
            assert (total_purchase_price <= updated_cash_bal), 'TODO'
        else:
            valid_purchase = purchase

        valid_action = -valid_sale + valid_purchase
        
        return valid_action
        
class PriceOnlyWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        obs, reward, terminal, info = self.env.step(action)
        return obs[self.env._n + 1: ], reward, terminal, info
        
    def reset(self):
        obs = self.env.reset()
        return obs[self.env._n + 1: ]
        
class ObsStackWrapper(Wrapper):
    def __init__(self, env, stack_size = 4):
        super().__init__(env)
        self.stack_size = stack_size
        self.obs_buffer = None
        
    def reset(self):
        self.obs_buffer = deque([], maxlen = self.stack_size)
        obs = self.env.reset()
        
        for _ in range(self.stack_size):
            self.obs_buffer.append(np.zeros(*obs.shape))
        self.obs_buffer.append(obs)
        
        return np.array(self.obs_buffer)
        
    def step(self, action):
        obs, reward, terminal, info = self.env.step(action)
        self.obs_buffer.append(obs)
        
        return np.array(self.obs_buffer), reward, terminal, info
        
class PortfolioWeightWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self):
        obs = self.env.reset()
        
        positions = self.env.positions
        current_prices = self.env.current_prices
        
        obs[0] = obs[0] / self.env.portfolio_value
        position_weights = ((self.env.positions * self.env.current_prices)
                            / self.portfolio_value)
        obs[1: self.env._n + 1] = position_weights
        
        return obs
    
    def step(self, action):
        obs, reward, terminal, info = self.env.step(action)
        
        positions = self.env.positions
        current_prices = self.env.current_prices
        
        obs[0] = obs[0] / self.env.portfolio_value
        position_weights = ((self.env.positions * self.env.current_prices)
                            / self.portfolio_value)
        obs[1: self.env._n + 1] = position_weights
        
        return obs, reward, terminal, info
        
class ActionBasedReward(Wrapper):
    def __init__(self, env, sell_at_terminal = True):
        super().__init__(env)
        self._sell_at_terminal = sell_at_terminal
        
    def step(self, action):
        reward = np.dot(self.env.current_prices, -action)
        obs, _, terminal, info = self.env.step(action)
        
        if terminal and self._sell_at_terminal:
            reward += np.dot(self.env.positions, self.env.current_prices)
        
        return obs, reward, terminal, info
        
class NoActionPenalty(Wrapper):
    def __init__(self, env, beta = 1e-3):
        super().__init__(env)
        self._beta = beta
        
    def step(self, action):
        obs, reward, terminal, info = self.env.step(action)
        
        if (action == 0.0).all():
            reward -= self.env.env_spec['beginning_cash'] * self._beta
        
        return obs, reward, terminal, info
        
class SimNormalization(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert (hasattr(env._simulator, 'historical_data')), 'TODO'
        self._mean = np.mean(env._simulator.historical_data, axis = 0)
        self._std = np.std(env._simulator.historical_data, axis = 0)
        
    def reset(self):
        obs = self.env.reset()
        obs[self.env._n + 1:] = ((obs[self.env._n + 1:] - self._mean) 
                                 / (self._std + 1e-6))
    
        return obs
        
    def step(self, action):
        obs, reward, terminal, info = self.env.step(action)
        obs[self.env._n + 1:] = ((obs[self.env._n + 1:] - self._mean) 
                                 / (self._std + 1e-6))
        
        return obs, reward, terminal, info
import abc
import numpy as np
import pandas as pd
import warnings
import os

from .simulators import (MultiRegimeGBM, 
                         GBMRegime, 
                         HistoricalMarket,
                         UniformPrices)
from .inventory import Inventory
from .spaces import TradeSpace, MarketSpace
from ..common.utils import (get_current_prices,
                            get_positions, 
                            get_cash_balance,
                            get_portfolio_value)

path = os.path.abspath(os.path.dirname(__file__))

def make_env(env_name, episode_len = 252):
    if env_name == 'MiniMarket-v1':
        init_prices = UniformPrices(0.8, 1.2, (4, ))
        simulator = MultiRegimeGBM(init_prices, episode_len = episode_len)
        
        return_period = 'annual'
        
        bull_regime = GBMRegime()
        mu = np.array([0.1321738, 0.1340978, 0.06562815, -0.03456354])
        sigma = np.array([0.12709004, 0.15593475, 0.06269818, 0.35404054])
        rho = np.array([[ 1.0,        0.82590478,  0.2785641,  -0.3353591 ],
                        [ 0.82590478, 1.0,         0.28292836, -0.52011784],
                        [ 0.2785641,  0.28292836,  1.0,        -0.34717838],
                        [-0.3353591, -0.52011784, -0.34717838,  1.0       ]])
        bull_regime.set_params(mu, sigma, rho, return_period)
        
        bear_regime = GBMRegime()
        mu = np.array([-0.12503673, -0.08069799, 0.03991433, 0.23415451])
        sigma = np.array([0.18568538, 0.21337199, 0.08138472, 0.38094824])
        rho = np.array([[ 1.0,         0.91221021,  0.32619213, -0.43998601],
                        [ 0.91221021,  1.0,         0.45842168, -0.62724839],
                        [ 0.32619213,  0.45842168,  1.0,        -0.50523724],
                        [-0.43998601, -0.62724839, -0.50523724,  1.0       ]])
        bear_regime.set_params(mu, sigma, rho, return_period)
        
        init_probs = np.array([0.84713644, 0.15286356])
        transition_probs = np.array([[9.99668545e-01, 3.31455088e-04],
                                     [1.83654729e-03, 9.98163453e-01]])
        
        simulator.set_params([bull_regime, bear_regime], 
                              transition_probs, init_probs)
        env = MarketEnvironment(simulator, 3, beginning_cash = 10000.0, 
                                commission = 0.0)
                                
    elif env_name == 'MiniMarket-v2':
        init_prices = UniformPrices(0.8, 1.2, (4, ))
        simulator = MultiRegimeGBM(init_prices, episode_len = episode_len)
        
        return_period = 'annual'
        
        bull_regime = GBMRegime()
        mu = np.array([0.15, 0.10, 0.05, -0.05])
        sigma = np.array([0.15, 0.2, 0.04, 0.35])
        rho = np.array([[ 1.00,  0.80,  0.30, -0.35],
                        [ 0.80,  1.00,  0.30, -0.50],
                        [ 0.30,  0.30,  1.00, -0.35],
                        [-0.35, -0.50, -0.35,  1.00]])
        bull_regime.set_params(mu, sigma, rho, return_period)
        
        bear_regime = GBMRegime()
        mu = np.array([-0.125, -0.075, 0.04, 0.20])
        sigma = np.array([0.2, 0.225, 0.08, 0.40])
        rho = np.array([[ 1.00,  0.90,  0.35, -0.50],
                        [ 0.90,  1.00,  0.45, -0.65],
                        [ 0.35,  0.45,  1.00, -0.60],
                        [-0.50, -0.65, -0.60,  1.00]])
        bear_regime.set_params(mu, sigma, rho, return_period)
        
        init_probs = np.array([0.80, 0.20])
        transition_probs = np.array([[0.9995, 0.0005],
                                     [0.0020, 0.9980]])
        
        simulator.set_params([bull_regime, bear_regime], 
                              transition_probs, init_probs)
        env = MarketEnvironment(simulator, 4, beginning_cash = 10000.0, 
                                commission = 0.0)
    
    elif env_name == 'Market-v1':
        stock_tickers = ['PG', 'JNJ', 'AAPL', 'MSFT', 'XOM', 
                         'JPM', 'BA', 'KO', 'WMT', 'DIS']
        data_file = os.path.join(path, 'market_data/historical_data.csv')
        historical_data = pd.read_csv(data_file, index_col = 0)
        historical_data.index = pd.to_datetime(historical_data.index)
        historical_data = historical_data.loc[:, stock_tickers]
        
        init_prices = UniformPrices(0.8, 1.2, (historical_data.shape[1], ))
        
        historical_data.loc['2000-03-01':'2002-10-31', 'regime'] = 1
        historical_data.loc['2007-10-01':'2009-03-31', 'regime'] = 1
        historical_data['regime'] = historical_data['regime'].fillna(0)
        historical_data['regime'] = historical_data['regime'].astype(int)
        
        simulator = MultiRegimeGBM(init_prices, episode_len = episode_len)
        simulator.fit_params(historical_data, regimes = 'last')
        
        env = MarketEnvironment(simulator, len(stock_tickers), 
                                beginning_cash = 10000.0, commission = 0.0)
    
    elif env_name == 'Market-v2':
        stock_tickers = ['PG', 'JNJ', 'AAPL', 'MSFT', 'XOM', 
                         'JPM', 'BA', 'KO', 'WMT', 'DIS']
        data_file = os.path.join(path, 'market_data/historical_data.csv')
        historical_data = pd.read_csv(data_file, index_col = 0)
        
        market_columns = list(historical_data.columns[90:])
        volume_columns = [ticker + '_VOLUME' for ticker in stock_tickers]
        range_columns = [ticker + '_DAYRANGE' for ticker in stock_tickers]
        columns = (stock_tickers + volume_columns 
                   + range_columns + market_columns)
        
        historical_data = historical_data.loc[:, columns].to_numpy()

        simulator = HistoricalMarket(historical_data, 
                                     episode_len = episode_len,
                                     scale = False)
        env = MarketEnvironment(simulator, len(stock_tickers), 
                                beginning_cash = 100000.0, commission = 0.0)
                                
    elif env_name == 'Market-v3':
        data_file = os.path.join(path, 'market_data/historical_data.csv')
        historical_data = pd.read_csv(data_file, index_col = 0)
        
        historical_data = historical_data.iloc[:, 90:]
        columns = (['WILL5000INDFC'] 
                   + list(historical_data.columns.drop('WILL5000INDFC')))
        
        historical_data = historical_data.loc[:, columns].to_numpy()

        simulator = HistoricalMarket(historical_data, 
                                     episode_len = episode_len,
                                     scale = False)
        env = MarketEnvironment(simulator, 1, 
                                beginning_cash = 10000.0, commission = 0.0)
                                
    else:
        raise NotImplementedError('Cannot create {}'.format(env_name))
                  
    return env
                   
class Environment(abc.ABC):
    def __init__(self):
        self.action_space = None
        self.observation_space = None
        self.reward_range = (-np.inf, np.inf)
        self.env_spec = {}
    
    @abc.abstractmethod
    def step(self, action):
        pass
    
    @abc.abstractmethod
    def reset(self):
        pass
    
    @abc.abstractmethod
    def close(self):
        pass
    
    @abc.abstractmethod
    def seed(self, seed = None):
        pass
        
    def render(self):
        raise NotImplementedError

    def __str__(self):
        return '{} Instance'.format(type(self).__name__)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        
    @property
    def unwrapped(self):
        return self
        
class MarketEnvironment(Environment):
    def __init__(self,
                 simulator,
                 n_tradable,
                 n_non_tradable = 0,
                 commission = 10,
                 beginning_cash = 10000):
        super().__init__()
        
        assert (n_tradable > 0), 'TODO'
        assert (n_non_tradable >= 0), 'TODO'
        self._n = n_tradable
        self._m = n_non_tradable
        
        self.action_space = TradeSpace(self._n)
        self.observation_space = MarketSpace(self._n, self._m)
        
        assert (hasattr(simulator, 'simulate')), 'TODO'
        self._simulator = simulator
        
        self._commission = commission
        self._inventory = Inventory(self._n, beginning_cash)
        
        self._time_step = None
        self._obs = None
        self._terminal = False
        
        self.seed()
        
        self.env_spec['n_tradable'] = self._n
        self.env_spec['n_non_tradable'] = self._m
        self.env_spec['commission'] = self._commission
        self.env_spec['beginning_cash'] = beginning_cash
    
    @property
    def observation(self):
        assert (self._obs is not None), 'TODO'
        return self._obs
    
    @property
    def positions(self):
        assert (self.observation is not None), 'TODO'
        return get_positions(self._obs, self._n)
    
    @property
    def cash_balance(self):
        assert (self.observation is not None), 'TODO'
        return get_cash_balance(self._obs)
    
    @property    
    def current_prices(self):
        assert (self.observation is not None), 'TODO'
        return get_current_prices(self._obs, self._n)
    
    @property
    def portfolio_value(self):
        assert (self.observation is not None), 'TODO'
        return get_portfolio_value(self._obs, self._n)
        
    @property
    def terminal(self):
        return self._terminal
    
    def seed(self, seed = None):
        self._rng = np.random.RandomState(seed)
        
        self._simulator.seed(seed = seed)
        self.action_space.seed(seed = seed)
        self.observation_space.seed(seed = seed)
    
    def reset(self):
        self._time_step = 0
        self._inventory.reset()
        self._terminal = False
        
        self._simulation, self._simulation_info = self._simulator.simulate()
        self._episode_len  = self._simulation.shape[0] - 1
        self._obs = self._create_obs()
        
        return np.array(self._obs)
     
    def close(self):
        self._time_step = None
        self._current_pos = None
    
    def step(self, action):
        assert (self._time_step is not None), 'TODO'
        assert (action in self.action_space), 'TODO'
        
        if self.terminal:
            warnings.warn('You are calling step() after the environment has'
                          + 'reached a terminal state; You should always call'
                          + 'reset() after reaching a terminal state')
            return np.array(self._obs), 0.0, self.terminal, {}
            
        current_value = self.portfolio_value
        
        self._inventory.update(action, self.current_prices, self._commission)
        self._time_step += 1
        self._obs = self._create_obs()
        
        new_value = self.portfolio_value
        reward = new_value - current_value
        
        if self._time_step >= self._episode_len:
            self._terminal = True
        
        return np.array(self._obs), reward, self._terminal, {}
        
    def _create_obs(self):
        obs = np.concatenate([np.array([self._inventory.cash_balance]),
                              self._inventory.positions,
                              self._simulation[self._time_step, :].squeeze()])
    
        return obs
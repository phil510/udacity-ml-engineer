import abc
import numpy as np
import itertools
import warnings

from ..common.utils import log_returns

PERIOD_DICT = {'annual': (252, 365),
               'quarterly': (63, 91),
               'monthly': (21, 30),
               'weekly': (5, 7),
               'daily': (1, 1)}

class Simulator(abc.ABC):
    def __init__(self):
        self.seed()

    @abc.abstractmethod
    def simulate(self):
        pass
        
    def seed(self, seed = None):
        self._rng = np.random.RandomState(seed)
        
class HistoricalMarket(Simulator):
    def __init__(self,
                 historical_data,
                 episode_len = 365,
                 scale = False):    
        self._historical_data = historical_data
        self.episode_len = episode_len
        self._scale = scale
            
        super().__init__()
        
    @property
    def historical_data(self):
        return self._historical_data
    
    def simulate(self):
        self._init_pos = self._rng.randint(0, self.historical_data.shape[0] 
                                              - self.episode_len - 1)
        self._last_pos = self._init_pos + self.episode_len + 1
        
        assert (self._last_pos < self.historical_data.shape[0]), 'TODO'
        
        simulation = self.historical_data[self._init_pos: self._last_pos, :]
        
        if self._scale:
            simulation = simulation / (simulation[0, :] + 1e-4)
        
        return np.array(simulation), {}
        
class MultiRegimeGBM(Simulator):
    def __init__(self, 
                 init_prices,
                 episode_len = 365):
        self._init_prices = init_prices
        assert (callable(self._init_prices)), 'TODO'
        
        self.episode_len = episode_len
        
        self._regimes = None
        self._transition_probs = None
        self._init_probs = None
        
        super().__init__()
        
    @property
    def regimes(self):
        return self._regimes
    
    @property    
    def init_probs(self):
        return self._init_probs
        
    @property    
    def transition_probs(self):
        return self._transition_probs
        
    def seed(self, seed = None):
        super().seed(seed = seed)
        try:
            self._init_prices.seed(seed = seed)
        except AttributeError as e:
            print(e)
            warnings.warn('init_prices object does not have a seed method')
    
    def simulate(self):
        ''' 
        GBM based on:
        http://www.columbia.edu/~ks20/4703-Sigman/4703-07-Notes-BM-GBM-I.pdf
        http://www.goddardconsulting.ca/matlab-monte-carlo-assetpaths-corr.html
        http://comisef.wikidot.com/tutorial:correlation
        '''
        
        assert (self._regimes is not None), 'TODO'
        assert (self._transition_probs is not None), 'TODO'
        assert (self._init_probs is not None), 'TODO'
        
        if len(self._regimes) == 1:
            regimes = [0] * self.episode_len
        
        else:
            regimes = []
            current_regime = np.argmax(self._rng.multinomial(1, 
                                       self._init_probs))
            regimes.append(current_regime)
            
            for e in range(self.episode_len - 1):
                probs = self._transition_probs[current_regime, :]
                current_regime = np.argmax(self._rng.multinomial(1, probs))
                regimes.append(current_regime)
        
        simulation = (np.ones(self._n) * self._init_prices()).reshape(1, -1)
        
        for r, group in itertools.groupby(regimes):
            regime = self._regimes[r]
            S_0 = simulation[-1, :] 
            
            T = len(list(group))
            dt = (1 / regime.return_period)
        
            mu = regime.mu
            sigma = regime.sigma
            C = regime._cholskey_lower.T
 
            nu = mu - (sigma * sigma) / 2.0
 
            Z = self._rng.normal(0, 1, size = (T, self._n))
            Y = np.matmul(Z, C)
            X = (sigma * np.sqrt(dt)) * Y  + (nu * dt)
            
            S = S_0 * np.cumprod(np.exp(X), axis = 0)

            simulation = np.concatenate([simulation, S], axis = 0)
        
        return simulation, {'regimes': regimes}
        
    def set_params(self, regimes, transition_probs, init_probs):
        transition_probs = np.asarray(transition_probs)
        init_probs = np.asarray(init_probs).squeeze()
        
        assert (all(isinstance(r, GBMRegime) for r in regimes)), 'TODO1'
        try:
            assert (all(r._n == len(self._init_prices) 
                    for r in regimes)), 'TODO2'
        except TypeError:
            pass
        
        assert (len(regimes) == init_probs.shape[0]), 'TODO'
        assert (transition_probs.shape == (init_probs.shape[0], 
                                           init_probs.shape[0])), 'TODO'
        
        assert ((transition_probs >= 0.0).all()), 'TODO'
        assert ((transition_probs <= 1.0).all()), 'TODO'
        assert (np.isclose(transition_probs.sum(axis = 1), 1.0).all()), 'TODO'
        assert ((init_probs >= 0.0).all()), 'TODO'
        assert ((init_probs <= 1.0).all()), 'TODO'
        assert (np.isclose(init_probs.sum(), 1.0)), 'TODO'
        
        self._regimes = regimes
        self._transition_probs = transition_probs
        self._init_probs = init_probs
        
        self._n = self._regimes[0]._n
        
    def fit_params(self, data, return_period = 'annual' , regimes = None):
        pos_dict = {'first': 0, 'last': -1}
        
        if regimes is None:
            regime = GBMRegime().fit_params(data, 
                                            return_period = return_period)
            
            self._regimes = [regime]
            self._init_probs = np.array([1.0])
            self._transition_probs = np.array([1.0]).reshape(1, 1)
            
            self._n = self._regimes[0]._n
        
        elif regimes in pos_dict:
            pos = pos_dict[regimes]
            regime_seq = data.iloc[:, pos]
            regime_seq.columns = 'regime' # to make referencing easier
            
            missing_regimes = regime_seq.isnull().sum()
            if missing_regimes > 0:
                warnings.warn('Found {} missing regime'.format(missing_regimes)
                              + ' values in data; data will be removed for'
                              + ' fitting parameters')
                regime_seq = regime_seq.dropna()
            
            # initial state probabilities
            p = regime_seq.value_counts().values / regime_seq.shape[0] 
            assert (np.isclose(p.sum(), 1.0)), 'TODO'
            
            n = p.shape[0] # number of regimes
            t = np.zeros((n, n)) # state transition probability matrix
            
            for i in range(regime_seq.shape[0] - 1):
                t[regime_seq[i], regime_seq[i + 1]] += 1    
            t = t / t.sum(axis = 1, keepdims = 1)
            
            for row in t:
                assert (np.isclose(row.sum(), 1.0)), 'TODO'
                
            r = [] # list of GBMRegime objects
            
            data_cols = data.columns.drop(data.columns[pos])
            day_offset =  PERIOD_DICT[return_period][1]
            growth_rates = log_returns(data.loc[:, data_cols], 
                                       day_offset = day_offset, 
                                       fill = True)
            growth_rates = growth_rates.join(regime_seq, how = 'inner')
            
            for i in range(n):
                current_regime = (growth_rates['regime'] == i)
                regime_data = growth_rates.loc[current_regime, data_cols]
                
                regime = GBMRegime()
                regime.fit_params(regime_data, calc_growth_rates = False, 
                                  return_period = return_period)
                r.append(regime)
                
            self._init_probs = p
            self._transition_probs = t
            self._regimes = r
            self._n = self._regimes[0]._n # number of variables to simulate
            
        else:
            raise ValueError('{} is not a valid regime value'.format(regimes))            
    
class GBMRegime(object):
    def __init__(self):
        self._n = None
        self._return_period = None
        self._mu = None
        self._sigma = None
        self._rho = None
        
    @property
    def return_period(self):
        return self._return_period
        
    @property
    def mu(self):
        return self._mu
        
    @property
    def sigma(self):
        return self._sigma
        
    @property
    def rho(self):
        return self._rho
        
    def fit_params(self, data, calc_growth_rates = True, 
                   return_period = 'annual'):
        self._return_period = PERIOD_DICT[return_period][0]
        
        if calc_growth_rates:
            day_offset = PERIOD_DICT[return_period][1]
            growth_rates = log_returns(data, day_offset = day_offset, 
                                       fill = True).values
        else:
            growth_rates = np.asarray(data)
        
        self._mu = np.mean(growth_rates, axis = 0)
        self._sigma = np.std(growth_rates, axis = 0, ddof = 1)
        self._rho = np.corrcoef(growth_rates, rowvar = False)
        
        self._cholskey_lower = np.linalg.cholesky(self._rho)
        
        self._n = self.mu.shape[0]
        
    def set_params(self, mu, sigma, rho, return_period):
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        rho = np.asarray(rho)
        
        assert (mu.shape == sigma.shape), 'TODO1'
        assert (mu.shape[0] == rho.shape[0]), 'TODO2'
        assert (np.allclose(rho, rho.T)), 'TODO3'
        
        self._cholskey_lower = np.linalg.cholesky(rho)
            
        self._return_period = PERIOD_DICT[return_period][0]
        self._mu = mu
        self._sigma = sigma
        self._rho = rho   

        self._n = self.mu.shape[0]
        
class UniformPrices(object):
    def __init__(self, low, high, shape):
        self._low = low
        self._high = high
        self._shape = shape
        
        self.seed()
        
    def seed(self, seed = None):
        self._rng = np.random.RandomState(seed)
        
    def __call__(self):
        return self._rng.uniform(self._low, self._high, self._shape)
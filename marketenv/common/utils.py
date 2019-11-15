import numpy as np
import pandas as pd

def log_returns(data, day_offset = 365, fill = False):
    df = data.copy()
    
    if fill:
        date_range = pd.date_range(df.index.min(), df.index.max(), freq = 'D')
        df_pp = df.reindex(date_range).fillna(method = 'ffill')
    else:
        df_pp = df.copy()
    
    df_pp.index = df_pp.index + pd.DateOffset(days = day_offset)
    df_pp.columns = [c + '_PP' for c in df_pp.columns] 
    
    growth_rates = df.join(df_pp, how = 'inner')
    
    for c in data.columns:
        growth_rates[c] = np.log(growth_rates[c] / growth_rates[c + '_PP'])
        
    growth_rates = growth_rates.loc[:, data.columns]
    
    return growth_rates
    
def get_sales(action):
    sale = np.array(action)
    sale[np.where(sale > 0)] = 0
    sale = abs(sale)
    
    return sale

def get_purchases(action):
    purchase = np.array(action)
    purchase[np.where(purchase < 0)] = 0
    
    return purchase
    
def get_current_prices(obs, n):
    return obs[(n + 1): (n * 2 + 1)]

def get_positions(obs, n):
    return obs[1: n + 1]

def get_cash_balance(obs):
   return obs[0]

def get_portfolio_value(obs, n):
    v = np.dot(get_positions(obs, n), get_current_prices(obs, n))
    v += get_cash_balance(obs)
    
    return v
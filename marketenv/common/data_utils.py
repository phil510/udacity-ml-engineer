import requests
import pandas as pd
import numpy as np
import os
import warnings
import logging
from datetime import datetime

TIINGO_BASE = 'https://api.tiingo.com/tiingo'
FRED_META_BASE = 'https://api.stlouisfed.org/fred/series?'
FRED_OBS_BASE = 'https://api.stlouisfed.org/fred/series/observations?'

logger = logging.getLogger(__name__)

class TiingoAPIError(Exception):
    pass
    
class FREDAPIError(Exception):
    pass

def has_duplicates(df, column):
    return (df.loc[df[column].duplicated(), :].shape[0] != 0)

def has_nulls(df):
    return (df.loc[df.isnull().any(axis = 1), :].shape[0] != 0)
    
def has_tiingo_errors(response):
    if 'detail' in response:
        return True
    else:
        return False
        
def has_fred_errors(response):
    if 'error_code' in response:
        return True
    else:
        return False
    
def get_tiingo_meta_data(tiingo_api_key, tickers):
    headers = {'Content-Type': 'application/json', 
               'Authorization' : 'Token ' + tiingo_api_key}
    
    columns = ['name', 'ticker', 'exchangeCode', 'startDate', 'endDate', 'description']
    stock_meta_data = pd.DataFrame(columns = columns)
    
    if len(tickers) == 0:
        warnings.warn('tickers is empty; returning an empty DataFrame')
        return stock_meta_data
    
    for ticker in tickers:
        request = '/'.join([TIINGO_BASE, 'daily', ticker + '?'])
        logger.info('Requesting meta data for {}'.format(ticker))
        response = requests.get(request, headers = headers)
        
        if has_tiingo_errors(response.json()):
            raise TiingoAPIError('{}'.format(response.json()['detail']))
            
        data = pd.read_json(response.text, lines = True)
        logger.info('Received meta data for {}'.format(ticker))
        
        stock_meta_data = pd.concat([stock_meta_data, data], sort = False)
        
    stock_meta_data = stock_meta_data.reset_index(drop = True)
    
    logger.info('Cleaning non-ASCII characters from descriptions')
    non_ascii_replacements = {'\xa0': ' ', '©': '', '®': '', '—': '-', '’': "'", '“': '"', '”': '"'}
    
    for k, v in non_ascii_replacements.items():
        stock_meta_data['description'] = stock_meta_data['description'].map(lambda x: x.replace(k, v))
    logger.info('Non-ASCII characters replaced with ASCII character')
    
    return stock_meta_data

def get_tiingo_data(tiingo_api_key, ticker, start_date, end_date):
    headers = {'Content-Type': 'application/json', 
               'Authorization' : 'Token ' + tiingo_api_key}
    
    base = '/'.join([TIINGO_BASE, 'daily', ticker, 'prices?'])
    arguments = '&'.join(['resampleFreq=daily', 'startDate=' + start_date, 'endDate=' + end_date])
    request = base + arguments
    
    logger.info('Requesting data for {}'.format(ticker))
    response = requests.get(request, headers = headers)
    
    if has_tiingo_errors(response.json()):
        raise TiingoAPIError('{}'.format(response.json()['detail']))
    
    data = pd.read_json(response.text)
    logger.info('Received data for {}'.format(ticker))
    
    data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
    numeric_columns = data.columns.drop('date')
    data.loc[:, numeric_columns] = data.loc[:, numeric_columns].apply(pd.to_numeric)
    
    data = data.sort_values('date').reset_index(drop = True)
    
    return data
    
def get_fred_meta_data(fred_api_key, series_ids):
    columns = ['id', 'title', 'last_updated', 'popularity', 
               'frequency', 'frequency_short',
               'observation_start', 'observation_end',
               'seasonal_adjustment', 'seasonal_adjustment_short', 
               'units', 'units_short', 'notes']
    market_meta_data = pd.DataFrame(columns = columns)
    
    if len(series_ids) == 0:
        warnings.warn('series_ids is empty; returning an empty DataFrame')
        return market_meta_data
    
    for series in series_ids:
        request = FRED_META_BASE + '&'.join(['series_id=' + series, 
                                             'api_key=' + fred_api_key, 
                                             'file_type=json'])
        logger.info('Requesting meta data for {}'.format(series))
        response = requests.get(request)
        
        if has_fred_errors(response.json()):
            raise FREDAPIError('{}'.format(response.json()['error_message']))
        
        data = pd.DataFrame.from_dict(response.json()['seriess'])
        logger.info('Received meta data for {}'.format(series))
        
        market_meta_data = pd.concat([market_meta_data, data], sort = False)
        
    market_meta_data = market_meta_data.reset_index(drop = True)
    
    return market_meta_data
    
def create_series_data_request(fred_api_key, series_id, frequency = None, 
                               observation_start = 'first', observation_end = 'last',
                               realtime_start = 'first', realtime_end = 'last'):
    if observation_start == 'first':
        observation_start = '1776-07-04'
    if observation_end == 'last':
        observation_end = '9999-12-31'
    if realtime_start == 'first':
        realtime_start = '1776-07-04'
    if realtime_end == 'last':
        realtime_end = '9999-12-31'
    
    api_key = 'api_key=' + fred_api_key
    series_id = 'series_id=' + series_id
    
    realtime_start = 'realtime_start=' + realtime_start
    realtime_end = 'realtime_end=' + realtime_end
    
    observation_start = 'observation_start=' + observation_start
    observation_end = 'observation_end=' + observation_end
    
    file_type = 'file_type=json'
    
    if frequency:
        frequency = 'frequency=' + frequency
        request = FRED_OBS_BASE + '&'.join([series_id, realtime_start, realtime_end,
                                            observation_start, observation_end, api_key, file_type,
                                            frequency])
    else:
        request = FRED_OBS_BASE + '&'.join([series_id, realtime_start, realtime_end, 
                                            observation_start, observation_end, api_key, file_type])
    
    return request
    
def get_daily_fred_data(fred_api_key, series_id, as_of = 'as_available',
                        observation_start = 'first', observation_end = 'last'):
    
    if as_of == 'today':
        request = create_series_data_request(fred_api_key, series_id, 
                                             observation_start = observation_start, 
                                             observation_end = observation_end,
                                             realtime_start = datetime.today().strftime('%Y-%m-%d'),
                                             realtime_end = datetime.today().strftime('%Y-%m-%d'))

        logger.info('Requesting data for {}'.format(series_id))
        response = requests.get(request)
        
        if has_fred_errors(response.json()):
            raise FREDAPIError('{}'.format(response.json()['error_message']))
        
        data = pd.DataFrame(response.json()['observations'])
        assert (not has_duplicates(data, 'date')), 'Duplicate date values from query'
        logger.info('Received data for {}'.format(series))
    
    elif as_of == 'as_available':
        request = create_series_data_request(fred_api_key, series_id, 
                                             observation_start = observation_start, 
                                             observation_end = observation_end)

        logger.info('Requesting data for {}'.format(series_id))
        response = requests.get(request)
        
        if has_fred_errors(response.json()):
            raise FREDAPIError('{}'.format(response.json()['error_message']))
        
        data = pd.DataFrame(response.json()['observations'])
        
        dates = data.groupby('date', as_index = False)['realtime_start'].min()
        data = data.merge(dates, on = ['date', 'realtime_start'], how = 'inner')
        assert (not has_duplicates(data, 'date')), 'Duplicate date values after join'
        logger.info('Received data for {}'.format(series_id))
        
    else:
        raise ValueError('{} is not a valid as_of value'.format(as_of))
    
    data['value'] = pd.to_numeric(data['value'], errors = 'coerce')
    data = data.dropna()
    
    date_columns = data.columns.drop(['value', 'realtime_end'])
    data.loc[:, date_columns] = data.loc[:, date_columns].apply(pd.to_datetime)
    
    data = data.sort_values('date').reset_index(drop = True)
    
    return data

# the monthly data requests 'as_available' might need some work
def get_monthly_fred_data(fred_api_key, series_id, as_of = 'as_available',
                            observation_start = 'first', observation_end = 'last'):
    
    if as_of == 'today':
        request = create_series_data_request(fred_api_key, series_id, 
                                             observation_start = observation_start, 
                                             observation_end = observation_end,
                                             realtime_start = datetime.today().strftime('%Y-%m-%d'),
                                             realtime_end = datetime.today().strftime('%Y-%m-%d'))
                                             
        logger.info('Requesting data for {}'.format(series_id))
        response = requests.get(request)
        
        if has_fred_errors(response.json()):
            raise FREDAPIError('{}'.format(response.json()['error_message']))
        
        data = pd.DataFrame(response.json()['observations'])
        assert (not has_duplicates(data, 'date')), 'Duplicate date values from query'
        logger.info('Received data for {}'.format(series))
    
    elif as_of == 'as_available':
        request = create_series_data_request(fred_api_key, series_id, 
                                             observation_start = observation_start, 
                                             observation_end = observation_end)
        
        logger.info('Requesting data for {}'.format(series_id))
        response = requests.get(request)
        
        if has_fred_errors(response.json()):
            raise FREDAPIError('{}'.format(response.json()['error_message']))
        
        data = pd.DataFrame(response.json()['observations'])
        
        dates = data.groupby('realtime_start', as_index = False)['date'].max()
        data = data.merge(dates, on = ['date', 'realtime_start'], how = 'inner')
        assert (not has_duplicates(data, 'realtime_start')), 'Duplicate date values after join'
        logger.info('Received data for {}'.format(series))
        
        data = data.sort_values('realtime_start')
        drop_records = []
        for i in range(data.shape[0]):
            if i == 0:
                continue
            if data['date'].iloc[i] < data['date'].iloc[i - 1]:
                drop_records.append(i)
                
        data = data.drop(data.index[drop_records], axis = 0)
        
    else:
        raise ValueError('{} is not a valid as_of value'.format(as_of))
    
    data['value'] = pd.to_numeric(data['value'], errors = 'coerce')
    data = data.dropna()
    
    date_columns = data.columns.drop(['value', 'realtime_end'])
    data.loc[:, date_columns] = data.loc[:, date_columns].apply(pd.to_datetime)
    
    data = data.sort_values('date').reset_index(drop = True)
    
    return data
    
def monthly_to_daily(series_data, as_of_type):
    data = series_data.copy()
    
    if as_of_type == 'as_available':
        data['observation_date'] = data['date']
        data['date'] = data['realtime_start']
    else:
        data['observation_date'] = data['date']
    
    min_date = data['date'].min()
    max_date = data['date'].max()

    date_range = pd.date_range(min_date, max_date, freq = 'B')
    date_range = pd.DataFrame({'date': date_range})

    data = data.merge(date_range, on = ['date'], how = 'outer').sort_values('date')
    assert (not has_duplicates(data, 'date')), 'Duplicate date values after join'

    data = data.fillna(method = 'ffill')
    
    data = data.loc[:, ['date', 'observation_date', 'realtime_start', 'realtime_end', 'value']]
    data = data.reset_index(drop = True)
    
    return data
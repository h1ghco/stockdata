import numpy as np
import pandas as pd
import glob
import shutil
from typing import List
#import matplotlib.pyplot as plt #test
import os
from alpha_vantage.timeseries import TimeSeries
from pandas.core.frame import DataFrame
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from pandas.api.types import is_string_dtype
from IPython.display import *
import bs4
import urllib.request
import json
import re
import datetime
import pandas as pd
import pickle
import requests
import pandas as pd 
import smtplib, ssl
import yagmail
import pandabase
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pandas_datareader import data as pdr
from finviz.screener import Screener
from IPython.display import display_html
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import yfinance as yf
import yaml
from pandas_datareader import data as pdr
from finvizfinance.quote import finvizfinance
import telegram_send
from yahoo_fin import stock_info as si


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 100)
pd.set_option('display.max_colwidth', 100)

def link_finviz(symbols, text):
    link = 'https://finviz.com/screener.ashx?v=351&t=' +",".join(symbols)
    from IPython.display import display, HTML
    # create a string template for the HTML snippet
    link_t = '<a href='+link+' target="_blank">> '+text+'</a>'
    # create HTML object, using the string template
    html = HTML(link_t.format())
    # display the HTML object to put the link on the page:
    display(html)

def get_stocklist(stocklist):
#    from yahoo_fin import stock_info as si
#    from utils.util import *
    from yahoo_fin import stock_info as si

    if stocklist == 'nasdaq':
        symbols = si.tickers_nasdaq()
    elif stocklist == 'sp500':
        symbols = si.tickers_sp500()
    elif stocklist == 'all':
        symbols = si.tickers_sp500()
        symbols += si.tickers_nasdaq()
    elif stocklist == 'tradingview':
        tradingview_wl_rename(path = '/Users/heiko/Downloads/TradingView')
        sym = tradingview_symbols(path = '/Users/heiko/Downloads/TradingView')
        symbols = sym.Symbol.unique()
    elif stocklist == 'tradingview_1':
        tradingview_wl_rename(path = '/Users/heiko/Downloads/TradingView')
        sym = tradingview_symbols(path = '/Users/heiko/Downloads/TradingView')
        sym = sym[sym.watchlist.str.contains('0_Action') | sym.watchlist.str.contains('@1_focus')]
        symbols = sym.Symbol.unique()
    elif stocklist == 'tradingview_2':
        tradingview_wl_rename(path = '/Users/heiko/Downloads/TradingView')
        sym = tradingview_symbols(path = '/Users/heiko/Downloads/TradingView')
        sym = sym[sym.watchlist.str.contains('0_Action') | sym.watchlist.str.contains('@1_focus') | sym.watchlist.str.contains('@2_watch_closely')]
        symbols = sym.Symbol.unique()
    
    print(stocklist, ' ', len(symbols))
    return symbols

def get_daily(symbols, df, interval_='1d', period_='12mo'):
    """[summary]

    Args:
        symbols ([type]): [description]
        interval (str, optional): [description]. Defaults to '1d'.
        period (str, optional): [description]. Defaults to '12mo'.

    Returns:
        [type]: [description]
    """
    import numpy as np
    import time
    yf.pdr_override() # <== that's all it takes :-)

    start_time = time.time()

    if df is None:
        print('none')
        df = pdr.get_data_yahoo(tickers=symbols, interval=interval_, period=period_, progress=True, group_by='ticker')
        df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
        #print(df)
        df = yahoo_prep(df)

    print('Get Index...')
    index_name = '^GSPC' # S&P 500
    index_df = pdr.get_data_yahoo(index_name,groupby='ticker')
    index_df['symbol']= 'sp500'
    index_df = yahoo_prep(index_df)
    index_df['date'] = pd.to_datetime(index_df['date'])
    index_df = index_df.rename({'adj_close':'sp500_adj_close'},axis=1)
    index_df['sp500_adj_close'] = index_df['sp500_adj_close']/10
    
    print('Merging...')
    df = pd.merge(df, index_df[['date','sp500_adj_close']], left_on='date', right_on='date', how='left')
    df['id']  = df['symbol'] + df['date'].astype(str).str.replace('-','')
    df.index = df.id
    df = df.drop('id',axis=1)
    df = df.sort_values('id', ascending=True)
    print("--- %s seconds ---" % (time.time() - start_time))

    df['sma10'] = df.groupby('symbol').apply(lambda g: g['adj_close'].rolling(10).mean()).reset_index().set_index('id').iloc[:,1]
    df['sma4'] = df.groupby('symbol').apply(lambda g: g['adj_close'].rolling(4).mean()).reset_index().set_index('id').iloc[:,1]
    df['sma7'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].rolling(7).mean()).reset_index().set_index('id').iloc[:,1]
    df['sma20'] = df.groupby('symbol').apply(lambda g: g['adj_close'].rolling(20).mean()).reset_index().set_index('id').iloc[:,1]
    df['sma50'] = df.groupby('symbol').apply(lambda g: g['adj_close'].rolling(50).mean()).reset_index().set_index('id').iloc[:,1]
    df['sma65'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].rolling(65).mean()).reset_index().set_index('id').iloc[:,1]
    df['sma100'] = df.groupby('symbol').apply(lambda g: g['adj_close'].rolling(100).mean()).reset_index().set_index('id').iloc[:,1]
    df['sma150'] = df.groupby('symbol').apply(lambda g: g['adj_close'].rolling(150).mean()).reset_index().set_index('id').iloc[:,1]
    df['sma200'] = df.groupby('symbol').apply(lambda g: g['adj_close'].rolling(200).mean()).reset_index().set_index('id').iloc[:,1]
    
    df['ema4'] = df.groupby('symbol').apply(lambda g: g['sma4'].ewm(4, adjust=False).mean()).reset_index().set_index('id').iloc[:,1]
    df['ema10'] = df.groupby('symbol').apply(lambda g: g['sma10'].ewm(10, adjust=False).mean()).reset_index().set_index('id').iloc[:,1]
    df['ema20'] = df.groupby('symbol').apply(lambda g: g['sma20'].ewm(20, adjust=False).mean()).reset_index().set_index('id').iloc[:,1]

    df['ema4_1'] = df[['symbol','ema4']].groupby('symbol').apply(lambda g:g['ema4'].shift(1)).reset_index().set_index('id').iloc[:,1]
    df['ema4_2'] = df[['symbol','ema4']].groupby('symbol').apply(lambda g:g['ema4'].shift(2)).reset_index().set_index('id').iloc[:,1]
    df['ema4_3'] = df[['symbol','ema4']].groupby('symbol').apply(lambda g:g['ema4'].shift(3)).reset_index().set_index('id').iloc[:,1]

    df['ti'] = df['sma7'] / df['sma65']

    df['sma10_diff'] = df.groupby('symbol').apply(lambda g: round(((g['close']/g['sma10'])-1)*100,2)).reset_index().set_index('id').iloc[:,1]
    df['sma20_diff'] = df.groupby('symbol').apply(lambda g: round(((g['close']/g['sma20'])-1)*100,2)).reset_index().set_index('id').iloc[:,1]
    df['sma50_diff'] = df.groupby('symbol').apply(lambda g: round(((g['close']/g['sma50'])-1)*100,2)).reset_index().set_index('id').iloc[:,1]
    df['sma150_diff'] = df.groupby('symbol').apply(lambda g: round(((g['close']/g['sma150'])-1)*100,2)).reset_index().set_index('id').iloc[:,1]
    df['sma200_diff'] = df.groupby('symbol').apply(lambda g: round(((g['close']/g['sma200'])-1)*100,2)).reset_index().set_index('id').iloc[:,1]
    df['ema10_diff'] = df.groupby('symbol').apply(lambda g: round(((g['close']/g['ema10'])-1)*100,2)).reset_index().set_index('id').iloc[:,1]
    df['ema20_diff'] = df.groupby('symbol').apply(lambda g: round(((g['close']/g['ema20'])-1)*100,2)).reset_index().set_index('id').iloc[:,1]

    print('Calc ADR...')
    df = calc_adr(df)

    df['volume_sma20'] = df[['symbol','volume']].groupby('symbol').apply(lambda g: g['volume'].rolling(window=20).mean()).reset_index().set_index('id').iloc[:,1]
    df['volume_sma50'] = df[['symbol','volume']].groupby('symbol').apply(lambda g: g['volume'].rolling(window=50).mean()).reset_index().set_index('id').iloc[:,1]
    df['volume_50_rel'] = df['volume'] / df['volume_sma50']
    df['volume_20_rel'] = df['volume'] / df['volume_sma20']
    df['volume_prev'] = df[['symbol','volume']].groupby('symbol').apply(lambda g: g['volume'].shift(1)).reset_index().set_index('id').iloc[:,1] 


    df['change'] =  df[['symbol','close']].groupby('symbol').apply(lambda g: g['close'].pct_change()*100).reset_index().set_index('id').iloc[:,1]
    df['change_1'] = df[['symbol','change']].groupby('symbol').apply(lambda g: g['change'].shift(2)).reset_index().set_index('id').iloc[:,1]
    df['change_2'] = df[['symbol','change']].groupby('symbol').apply(lambda g: g['change'].shift(3)).reset_index().set_index('id').iloc[:,1]
    df['change_3'] = df[['symbol','change']].groupby('symbol').apply(lambda g: g['change'].shift(4)).reset_index().set_index('id').iloc[:,1]

    df['change_after'] = df[['symbol','change']].groupby('symbol').apply(lambda g: g['change'].shift(-1)).reset_index().set_index('id').iloc[:,1]
    df['change_after2'] = df[['symbol','change']].groupby('symbol').apply(lambda g: g['change'].shift(-2)).reset_index().set_index('id').iloc[:,1]
    df['change_after3'] = df[['symbol','change']].groupby('symbol').apply(lambda g: g['change'].shift(-3)).reset_index().set_index('id').iloc[:,1]
    df['close_prev'] = df[['symbol','close']].groupby('symbol').apply(lambda g: g['close'].shift(1)).reset_index().set_index('id').iloc[:,1]
    df['open_after'] = df[['symbol','open']].groupby('symbol').apply(lambda g: g['open'].shift(-1)).reset_index().set_index('id').iloc[:,1] 

    print("--- %s seconds ---" % (time.time() - start_time))
    #df.to_parquet('yahoo_nasdaq_2022_full_prep_mas.parquet')
    #df = pd.read_parquet('yahoo_nasdaq_2022_full_prep_mas.parquet')
    #start_time = time.time()

    df['min_21'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].rolling(21).min()).reset_index().set_index('id').iloc[:,1]
    df['max_21'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].rolling(21).max()).reset_index().set_index('id').iloc[:,1]
    df['max_63'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].rolling(63).max()).reset_index().set_index('id').iloc[:,1]
    df['max_5'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].rolling(5).max()).reset_index().set_index('id').iloc[:,1]
    df['min_5'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].rolling(5).min()).reset_index().set_index('id').iloc[:,1]

    df['min_63'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].rolling(63, min_periods=1).min()).reset_index().set_index('id').iloc[:,1]
    df['min_126'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].rolling(126, min_periods=1).min()).reset_index().set_index('id').iloc[:,1]
    df['min_252'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].rolling(252, min_periods=1).min()).reset_index().set_index('id').iloc[:,1]
    df['max_252'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].rolling(252, min_periods=1).max()).reset_index().set_index('id').iloc[:,1]

    df['high_low'] = df[['symbol','high','low']].groupby('symbol').apply(lambda g: (g['high'] - g['low'])).reset_index().set_index('id').iloc[:,1]
    df['high_close'] = df[['symbol','high','close']].groupby('symbol').apply(lambda g: np.abs(g['high'] - g['close'].shift())).reset_index().set_index('id').iloc[:,1]
    df['low_close'] = df[['symbol','low','close']].groupby('symbol').apply(lambda g:  np.abs(g['low'] - g['close'].shift())).reset_index().set_index('id').iloc[:,1]
    df['true_range'] = df[['symbol','high_low','high_close','low_close']].groupby('symbol').apply(lambda g: np.max(g[['high_low','high_close','low_close']], axis=1)).reset_index().set_index('id').iloc[:,1]
    df['atr'] = df[['symbol','true_range']].groupby('symbol').apply(lambda g: g['true_range'].rolling(14).sum()/14).reset_index().set_index('id').iloc[:,1]

    df['range'] = df[['symbol','close','low','high']].groupby('symbol').apply(lambda g:(g['close'] - g['low']) / (g['high'] - g['low'])*100).reset_index().set_index('id').iloc[:,1]
    df['range_perc'] = df[['symbol','close','low','high','open','adj_close','range']].groupby('symbol').apply(lambda g:abs(g['adj_close']-g['open'])/g['range']).reset_index().set_index('id').iloc[:,1]

    # RELATIVE STRENGTH
    df['rs12'] = df[['symbol','close']].groupby('symbol').apply(lambda g: 2 * (g['close']/g['close'].shift(63)) + (g['close']/g['close'].shift(126))+ (g['close']/g['close'].shift(189))+ (g['close']/g['close'].shift(252))).reset_index().set_index('id').iloc[:,1]
    df['rs3'] = df[['symbol','close']].groupby('symbol').apply(lambda g: g['close'].rolling(window=7).mean() / g['close'].rolling(window=65).mean()).reset_index().set_index('id').iloc[:,1]
    df['rs_max_252'] = df[['symbol','close','rs12']].groupby('symbol').apply(lambda g: g['rs12'].rolling(252).max()).reset_index().set_index('id').iloc[:,1]
    df['rs_max_21'] = df[['symbol','close','rs12']].groupby('symbol').apply(lambda g: g['rs12'].rolling(21).max()).reset_index().set_index('id').iloc[:,1]
    df['rs_max_63'] = df[['symbol','close','rs12']].groupby('symbol').apply(lambda g: g['rs12'].rolling(63).max()).reset_index().set_index('id').iloc[:,1]
    df['rs_max_5'] = df[['symbol','close','rs12']].groupby('symbol').apply(lambda g: g['rs12'].rolling(5).max()).reset_index().set_index('id').iloc[:,1]

    #df.groupby('symbol').apply(lambda g: np.mean(g.close[0:7]) / np.mean(g.close[0:65]))
    print("--- %s seconds ---" % (time.time() - start_time))

    print('Calc Prev...')
    df['rslinebasic'] = df['adj_close']/df['sp500_adj_close']*100 #s3
    df['mult'] = df[['symbol','adj_close','sp500_adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].shift(60) / g['sp500_adj_close'].shift(60)).reset_index().set_index('id').iloc[:,1] #mult
    df['rsline'] =  df[['symbol','rslinebasic','mult']].groupby('symbol').apply(lambda g: g['rslinebasic'] * g['mult'] * 0.85).reset_index().set_index('id').iloc[:,1]
    df['rsline'] = df[['symbol','rsline']].groupby('symbol').apply(lambda g: g['rsline']*10).reset_index().set_index('id').iloc[:,1]
    #df = pd.read_parquet('yahoo_nasdaq_2022_full_prep_mas.parquet')
    #df = df.groupby('symbol').tail(565)
    #start_time = time.time()

    #df.groupby('symbol').transform(lambda g: g['change'].shift(1))
    df['prev_change'] = df[['symbol','change']].groupby('symbol').apply(lambda g: g['change'].shift(1)).reset_index().set_index('id').iloc[:,1]
    df['prev_day_close'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].shift(1)).reset_index().set_index('id').iloc[:,1]
    df['prev_day_high'] = df[['symbol','high']].groupby('symbol').apply(lambda g: g['high'].shift(1)).reset_index().set_index('id').iloc[:,1]
    df['prev_day_low'] = df[['symbol','low']].groupby('symbol').apply(lambda g: g['low'].shift(1)).reset_index().set_index('id').iloc[:,1]
    df['prev_day_open'] = df[['symbol','open']].groupby('symbol').apply(lambda g: g['open'].shift(1)).reset_index().set_index('id').iloc[:,1]
    df['prev_day_open_3'] = df[['symbol','open']].groupby('symbol').apply(lambda g: g['open'].shift(3)).reset_index().set_index('id').iloc[:,1]

    df['prev_close_1'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].shift(1)).reset_index().set_index('id').iloc[:,1]
    df['prev_close_2'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].shift(2)).reset_index().set_index('id').iloc[:,1]
    df['prev_close_3'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].shift(3)).reset_index().set_index('id').iloc[:,1]
    df['prev_close_4'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].shift(4)).reset_index().set_index('id').iloc[:,1]
    df['prev_close_5'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: g['adj_close'].shift(5)).reset_index().set_index('id').iloc[:,1] 

    df['volume_10d_max'] = df[['symbol','volume']].groupby('symbol').apply(lambda g: g['volume'].rolling(window=10).max()).reset_index().set_index('id').iloc[:,1]
    df['volume_5d_max'] = df[['symbol','volume']].groupby('symbol').apply(lambda g: g['volume'].rolling(window=5).max()).reset_index().set_index('id').iloc[:,1]

    df['pivot_10d'] = df[['symbol','adj_close','prev_day_close','volume','volume_10d_max']].groupby('symbol').apply(lambda g: (g['adj_close'] > g['prev_day_close']) & (g['volume'] >= g['volume_10d_max'])).reset_index().set_index('id').iloc[:,1]
    df['pivot_5d'] = df[['symbol','adj_close','prev_day_close','volume','volume_5d_max']].groupby('symbol').apply(lambda g: (g['adj_close'] > g['prev_day_close']) & (g['volume'] >= g['volume_5d_max'])).reset_index().set_index('id').iloc[:,1]

    df['insideday'] = df[['symbol','adj_close','prev_day_close','prev_day_open']].groupby('symbol').apply(lambda g: (g['adj_close'] <= g['prev_day_close']) & (g['adj_close'] >= g['prev_day_open'])).reset_index().set_index('id').iloc[:,1]

    df['whick'] = df[['symbol','adj_close','prev_day_high','prev_day_close']].groupby('symbol').apply(lambda g:(g['adj_close'] <= g['prev_day_high']) & (g['adj_close'] >= g['prev_day_close'])).reset_index().set_index('id').iloc[:,1]

    #print("--- %s seconds ---" % (time.time() - start_time))
    #df.to_parquet('yahoo_nasdaq_2022_2020_prep_mas.parquet')
    #df.groupby(['symbol']).first(1)
    #start_time = time.time()

    #df['range_'] = df.groupby('symbol').apply(lambda g:(g['adj_close'] > (g['low'] + (g['range']/2)))).reset_index().set_index('id').iloc[:,1]

    df['oops'] = df[['symbol','adj_close','prev_day_close','open']].groupby('symbol').apply(lambda g:(g['adj_close'] > g['prev_day_close']) & (g['open'] < g['prev_day_close'])).reset_index().set_index('id').iloc[:,1]
    df['kicker'] = df[['symbol','prev_change','open','prev_day_high']].groupby('symbol').apply(lambda g:(g['prev_change'] < 1) & (g['open'] > g['prev_day_high'])).reset_index().set_index('id').iloc[:,1]

    df['b3'] = df[['symbol','prev_close_1','adj_close','prev_close_2','prev_close_3','volume_sma20','volume']].groupby('symbol').apply(lambda g:(g['prev_close_1'] < g['adj_close']) &
                            (g['prev_close_2'] < g['adj_close']) &
                            (g['prev_close_3'] < g['adj_close']) &
                            (g['volume_sma20'] > g['volume'])).reset_index().set_index('id').iloc[:,1]

    df['upside_reversal'] = df[['symbol','prev_close_1','low','adj_close','range']].groupby('symbol').apply(lambda g:(g['prev_close_1'] < g['low']) & (
        g['adj_close'] > (g['low'] + (g['range']/2)))).reset_index().set_index('id').iloc[:,1]

    df['power3'] = df.groupby('symbol').apply(lambda g:(g['prev_close_1'] < g['sma10']) & (g['prev_close_1'] < g['sma20']) & (g['prev_close_1'] < g['sma50']) & (
        g['adj_close'] > g['sma10']) & (g['adj_close'] > g['sma20']) & (g['adj_close'] > g['sma50'])).reset_index().set_index('id').iloc[:,1]
    df['power2'] = df.groupby('symbol').apply(lambda g:(g['prev_close_1'] < g['sma10']) & (g['prev_close_1'] < g['sma20']) & (
        g['adj_close'] > g['sma10']) & (g['adj_close'] > g['sma20'])).reset_index().set_index('id').iloc[:,1]
    df['sma_10_sma_20_tight'] = df.groupby('symbol').apply(lambda g:((g['sma10'] / g['sma20']) <= 1.02) & (g['adj_close'] > g['sma10'])).reset_index().set_index('id').iloc[:,1]

    df['insideday'] =  df[['symbol','adj_close','prev_day_close','prev_day_open']].groupby('symbol').apply(lambda g:((g['adj_close']) < g['prev_day_close']) & ((g['adj_close'] > g['prev_day_open'])) |
                                ((g['adj_close'] > g['prev_day_open']) & (g['adj_close'] < g['prev_day_close']))).reset_index().set_index('id').iloc[:,1]

    df['outside_bullish'] = df[['symbol','adj_close','prev_day_high']].groupby('symbol').apply(lambda g: (g['adj_close'] > g['prev_day_high'])).reset_index().set_index('id').iloc[:,1]
    df['outside_bearish'] = df[['symbol','adj_close','prev_day_low']].groupby('symbol').apply(lambda g: (g['adj_close'] < g['prev_day_low'])).reset_index().set_index('id').iloc[:,1]

    df['slingshot'] = df[['symbol','adj_close','prev_close_1','ema4','ema4_1','prev_close_2','ema4_2','prev_close_3','ema4_3']].groupby('symbol').apply(lambda g:(g['adj_close'] > g['ema4']) & (g['prev_close_1'] < g['ema4_1']) & \
                                (g['prev_close_2'] < g['ema4_2']) & (g['prev_close_3'] < g['ema4_3'])).reset_index().set_index('id').iloc[:,1]
    
    df['gap_am'] = df[['symbol','close','open_after']].groupby('symbol').apply(lambda g: ((g['open_after'] - g['close']) / g['close'])*100).reset_index().set_index('id').iloc[:,1]
    df['gap_pre'] = df[['symbol','open','close_prev']].groupby('symbol').apply(lambda g: ((g['open'] - g['close_prev']) / g['close_prev'])*100).reset_index().set_index('id').iloc[:,1]

    df['1m_perf'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: ((g['adj_close'] / g['adj_close'].shift(21)))).reset_index().set_index('id').iloc[:,1]
    df['3m_perf'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: ((g['adj_close'] / g['adj_close'].shift(63)))).reset_index().set_index('id').iloc[:,1]
    df['6m_perf'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: ((g['adj_close'] / g['adj_close'].shift(125)))).reset_index().set_index('id').iloc[:,1]
    df['12m_perf'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: ((g['adj_close'] / g['adj_close'].shift(252)))).reset_index().set_index('id').iloc[:,1]
    df['5d_perf'] = df[['symbol','adj_close']].groupby('symbol').apply(lambda g: ((g['adj_close'] / g['adj_close'].shift(5)))).reset_index().set_index('id').iloc[:,1]

    #df['perf_3m'] = df[['symbol','adj_close']].groupby('symbol').pct_change(65)
    #df['perf_6m'] = df[['symbol','adj_close']].groupby('symbol').pct_change(125)
    #df['perf_12m'] = df[['symbol','adj_close']].groupby('symbol').pct_change(252)

    df['change'] = df[['symbol','adj_close']].groupby('symbol').pct_change()
    df['change_1'] = df[['symbol','adj_close']].groupby('symbol').pct_change(1)

    df['gain_open'] = df[['symbol','adj_close','open']].groupby('symbol').apply(lambda g: g['adj_close']/g['open']).reset_index().set_index('id').iloc[:,1] 

    print("--- %s seconds ---" % (time.time() - start_time))
    #df.to_parquet('yahoo_nasdaq_2022_2020_prep_mas.parquet')
    return df

def get_weekly(symbols=None, data=None, iThreshold = 1):
    import aiohttp
    import asyncio
    import time
    import pandas as pd
    import requests
    import time
    import datetime

    yf.pdr_override() # <== that's all it takes :-)

    """[summary]
    from yahoo_fin import stock_info as si
    stocklist = si.tickers_nasdaq()

    
    highestHigh = max(high, high[1], high[2])

    //---------------------------------------------------------
    // Calculate % change between closes
    //----------------------------------------------------------
    percentChange1 = ((close - close[1]) / close[1]) * 100
    percentChange2 = ((close[1] - close[2]) / close[2]) * 100

    //---------------------------------------------------------
    // 3 weeks tight if closes meet threshold
    //----------------------------------------------------------
    threeWeeksTight = (abs(percentChange1) <= iThreshold) and (abs(percentChange2) <= iThreshold)


    Args:
        symbols ([type]): [description]

    Returns:
        [type]: [description]


    """

    ##https://stackoverflow.com/questions/63107594/how-to-deal-with-multi-level-column-names-downloaded-with-yfinance/63107801#63107801

    if (symbols is None) & (data is None):
        print('symbols or data needs to be provided')

    if data is None:
        data = pdr.get_data_yahoo(tickers=symbols, interval='1wk', period='6mo', progress=True, group_by='ticker')
        data = data.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
        data = yahoo_prep(data)

    data['close_high'] = data.groupby('symbol')['close'].apply(lambda g: g.rolling(window=40,min_periods=1).max())
    data['close_high_max'] = data.groupby('symbol')['close'].apply(lambda g: g.rolling(window=1000,min_periods=1).max())

    max_close = data.groupby('symbol')['close'].max()
    max_close.name = 'close_ath_high'

    data = pd.merge(left=data, right=max_close, left_on='symbol', right_index=True , how='left')

    data['percent_change_1'] = data[['symbol','adj_close']].groupby('symbol').apply(lambda g: (g['adj_close'] - g['adj_close'].shift(1))/g['adj_close'].shift(1) * 100).reset_index().set_index('id').iloc[:,1] 
    data['percent_change_2'] = data[['symbol','adj_close']].groupby('symbol').apply(lambda g: (g['adj_close'].shift(1) - g['adj_close'].shift(2))/g['adj_close'].shift(2) * 100).reset_index().set_index('id').iloc[:,1] 

    data['prev_open'] = data.groupby('symbol')['open'].shift(1)
    data['prev_open_2'] = data.groupby('symbol')['open'].shift(2)
    data['prev_open_3'] = data.groupby('symbol')['open'].shift(3)

    data['prev_close'] = data.groupby('symbol')['close'].shift(1)
    data['prev_close_2'] = data.groupby('symbol')['close'].shift(2)
    data['prev_close_3'] = data.groupby('symbol')['close'].shift(3)

    data['prev_high'] = data.groupby('symbol')['high'].shift(1)
    data['prev_high_2'] = data.groupby('symbol')['high'].shift(2)
    data['prev_high_3'] = data.groupby('symbol')['high'].shift(2)

    data['prev_low'] = data.groupby('symbol')['high'].shift(1)
    data['prev_low_2'] = data.groupby('symbol')['high'].shift(2)
    data['prev_low_2'] = data.groupby('symbol')['high'].shift(3)

    #data['3w_tight'] = data.groupby('symbol').apply(lambda g: (g['prev_open_2'] <= g['prev_open']) & (g['prev_close_2']*iThreshold >= g['prev_close']) & \
    #                    (g['prev_open_2'] <= g['open']) & (g['prev_close_2']*iThreshold >= g['close'])).reset_index().set_index('id').iloc[:,1]
    #data['2w_tight'] = data.groupby('symbol').apply(lambda g: (g['prev_open_2'] <= g['open']) & (g['prev_close_2']*iThreshold >= g['close'])).reset_index().set_index('id').iloc[:,1]
    data['three_weeks_tight'] = data[['symbol','adj_close','percent_change_1','percent_change_2']].groupby('symbol').apply(lambda g: (np.abs(g['percent_change_1'])<= iThreshold) & (np.abs(g['percent_change_2'])<= iThreshold)).reset_index().set_index('id').iloc[:,1] 

    #data['prev_open_3'] <= data['prev_open_2'] & data['prev_close_3'] >= data['prev_close_2'] &
        # data['prev_open_2'] <= data['prev_open_1'] & data['prev_close_2'] >= data['prev_close_1'] &

    data['high_low'] = data[['symbol','high','low']].groupby('symbol').apply(lambda g: (g['high'] - g['low'])).reset_index().set_index('id').iloc[:,1]
    data['high_close'] = data[['symbol','high','close']].groupby('symbol').apply(lambda g: np.abs(g['high'] - g['close'].shift())).reset_index().set_index('id').iloc[:,1]
    data['low_close'] = data[['symbol','low','close']].groupby('symbol').apply(lambda g:  np.abs(g['low'] - g['close'].shift())).reset_index().set_index('id').iloc[:,1]

    data['true_range'] = data[['symbol','high_low','high_close','low_close']].groupby('symbol').apply(lambda g: np.max(g[['high_low','high_close','low_close']], axis=1)).reset_index().set_index('id').iloc[:,1]
    data['atr'] = data[['symbol','true_range']].groupby('symbol').apply(lambda g: g['true_range'].rolling(14).sum()/14).reset_index().set_index('id').iloc[:,1]

    data['range'] = data[['symbol','close','low','high']].groupby('symbol').apply(lambda g:(g['close'] - g['low']) / (g['high'] - g['low'])*100).reset_index().set_index('id').iloc[:,1]
    data['range_perc'] = data[['symbol','close','low','high','open','adj_close']].groupby('symbol').apply(lambda g:abs(g['adj_close']-g['open'])/g['range']).reset_index().set_index('id').iloc[:,1]

    data['adr'] = data[['symbol','high','low']].groupby('symbol').apply(lambda g: np.abs(g['high'] / g['low'])).reset_index().set_index('id').iloc[:,1]
    data['adr'] = data['adr']-1

    data['adr_ma'] = data[['symbol','adr']].groupby('symbol').apply(lambda g: g['adr'].rolling(window=14).mean()).reset_index().set_index('id').iloc[:,1]
    data['adr_ma'] = data['adr_ma'] * 100
    data['adr_ma'] = data['adr_ma'].round(2)

    data['volume_sma10'] = data[['symbol','volume']].groupby('symbol').apply(lambda g: g['volume'].rolling(window=10).mean()).reset_index().set_index('id').iloc[:,1]
    data['volume_sma40'] = data[['symbol','volume']].groupby('symbol').apply(lambda g: g['volume'].rolling(window=40).mean()).reset_index().set_index('id').iloc[:,1]

    data['volume_10_rel'] = data['volume'] / data['volume_sma10']
    data['volume_40_rel'] = data['volume'] / data['volume_sma40']

    data['volume_10w_max'] = data[['symbol','volume']].groupby('symbol').apply(lambda g: g['volume'].rolling(window=10).max()).reset_index().set_index('id').iloc[:,1]
    data['volume_20w_max'] = data[['symbol','volume']].groupby('symbol').apply(lambda g: g['volume'].rolling(window=20).max()).reset_index().set_index('id').iloc[:,1]
    data['volume_max'] = data[['symbol','volume']].groupby('symbol').apply(lambda g: g['volume'].rolling(window=1000).max()).reset_index().set_index('id').iloc[:,1]
    data['volume_max_f'] = data[['symbol','volume','volume_max']].groupby('symbol').apply(lambda g: g['volume'] == g['volume_max']).reset_index().set_index('id').iloc[:,1]

    return data


def finviz_get_charts(symbols, output_dir=None):
    for s in symbols:
        try:
            if os.path.exists(os.path.join(output_dir, s + '.jpg')):
                modified_date = os.path.getmtime(os.path.join(output_dir, s + '.jpg'))
                modified_date = datetime.fromtimestamp(modified_date)
                if datetime.today().date() != modified_date.date():
                    #print('skipped: ', s)
                    stock = finvizfinance(s)
                    stock.ticker_charts(out_dir=output_dir)
                else:
                    continue
            else:
                #print('getting: ', s)
                stock = finvizfinance(s)
                stock.ticker_charts(out_dir=output_dir)
        except:
            print(s)


def daily_scans(df, scans, load_premarket=True):
    import os
    import runpy

    d = {}

    if load_premarket:
        runpy.run_path(path_name='asyncio_yahoo_premarket.py')

    df_pre = pd.read_parquet('./data/yahoo_premarket.parquet')
    df_last = df[df.columns].groupby('symbol').last().reset_index()
    df_last = pd.merge(df_last, df_pre, on='symbol', how='left')
    df_last['marketCap'] = pd.to_numeric(df_last.marketCap, errors='ignore')

    #df_last[(df_last.close>4) & (df_last.volume_sma50>250000) & ((df_last.close*df_last.volume)>2000000) & (df_last.adr_ma>4)]
    if ('gainer_1m' in scans) | ('all' in scans):
        print('1M Performance')
        d['gainer_1m'] = df_last[(df_last.close>4) & (df_last.volume_sma50>250000) & ((df_last.close*df_last.volume)>2000000) & (df_last.adr_ma>4)].sort_values('1m_perf', ascending=False)
    if ('gainer_3m' in scans) | ('all' in scans):
        print('3M Performance')
        d['gainer_3m'] = df_last[(df_last.close>4) & (df_last.volume_sma50>250000) & ((df_last.close*df_last.volume)>2000000) & (df_last.adr_ma>4)].sort_values('3m_perf', ascending=False)
    if ('gainer_6m' in scans) | ('all' in scans):
        print('6M Performance')
        d['gainer_6m'] = df_last[(df_last.close>4) & (df_last.volume_sma50>250000) & ((df_last.close*df_last.volume)>2000000) & (df_last.adr_ma>4)].sort_values('6m_perf', ascending=False)
    if ('gainer_12m' in scans) | ('all' in scans):
        print('12M Performance')
        d['gainer_12m'] = df_last[(df_last.close>4) & (df_last.volume_sma50>250000) & ((df_last.close*df_last.volume)>2000000) & (df_last.adr_ma>4)].sort_values('12m_perf')
    if ('gainer_5day' in scans) | ('all' in scans):
        d['gainer_5day'] = df_last[(df_last.close>4) & (df_last.volume_sma50>250000) & ((df_last.close*df_last.volume)>2000000) & (df_last.close/df_last.prev_close_5)>1.2]
    if ('gainer_top' in scans) | ('all' in scans):
        d['gainer_top'] = df_last[(df_last.close>4) & (df_last.volume_sma50>200000) & ((df_last.close*df_last.volume)>2000000) & (df_last.close / df_last.prev_close_1>1.02)].sort_values('pct_change')
    if ('gainer_open' in scans) | ('all' in scans):
        d['gainer_open'] = df_last[(df_last.close>4) & (df_last.volume_sma50>250000) & ((df_last.close*df_last.volume)>2000000) & (df_last.close / df_last.prev_close_1>1.02)].sort_values('gain_open')
    if ('gainer_volume' in scans) | ('all' in scans):
        d['gainer_volume'] = df_last[(df_last.close>4) & (df_last.volume_sma50>200000) & ((df_last.close*df_last.volume_sma50)>2000000) & (df_last.close/df_last.prev_close_1)>1 &(df_last.volume_50_rel >2)].sort_values('volume_50_rel')
    if ('high_52w' in scans) | ('all' in scans):
        d['high_52w'] = df_last[(df_last.close>4) & (df_last.volume_sma50>200000) & ((df_last.close*df_last.volume_sma50)>2000000) & (df_last.close>=df_last.max_252) &(df_last.volume_50_rel >1)].sort_values('volume_50_rel')

    #df_last[(df_last.marketCap>500000000) & (df_last.marketCap<100000000000) & (df_last.close>4) & (df_last.sma20>df_last.sma50) & (df_last.close > df_last.sma50) & (df_last.volume_sma50>200000)  & ((df_last.volume_sma50*df_last.close)>2000000)& ((df_last.open/df_last.prev_close_1)>1.03)].sort_values('gain_open')

    if ('rsnhbp_12m' in scans) | ('all' in scans):
        d['rsnhbp_12m'] = df_last[(df_last.rs12 >=df_last.rs_max_252) & (df_last.close>df_last['max_252']) & (df_last.volume_sma50>200000) & ((df_last.volume_sma50*df_last.close)>200000)].sort_values('12m_perf')
    if ('rsnhbp_1m' in scans) | ('all' in scans):
        d['rsnhbp_1m'] = df_last[(df_last.rs12 >=df_last.rs_max_21) & (df_last.close>df_last['max_21']) & (df_last.volume_sma50>200000) & ((df_last.volume_sma50*df_last.close)>200000)].sort_values('1m_perf')
    if ('rsnhbp_3m' in scans ) | ('all' in scans):
        d['rsnhbp_3m'] = df_last[(df_last.rs12 >=df_last.rs_max_63) & (df_last.close>df_last['max_63']) & (df_last.volume_sma50>200000) & ((df_last.volume_sma50*df_last.close)>200000)].sort_values('3m_perf')
    # @TODO: 6m berechnen!
    #d['rsnhbp_6m'] =df_last[(df_last.rs12 >=df_last.rs_max_63) & (df_last.close>df_last['max_63']) & (df_last.volume_sma50>200000) & ((df_last.volume_sma50*df_last.close)>200000)].sort_values('3m_perf')
    if ('rsnhbp_5d' in scans) | ('all' in scans):
        d['rsnhbp_5d'] = df_last[(df_last.rs12 >=df_last.rs_max_5) & (df_last.close>df_last['max_5']) & (df_last.volume_sma50>200000) & ((df_last.volume_sma50*df_last.close)>200000)].sort_values('5d_perf')
    if ('ep' in scans) | ('all' in scans):
        #EP
        d['ep'] = df_last[(df_last.marketCap>500000000) & (df_last.marketCap<100000000000) & (df_last.close>4) & (df_last.sma20 > df_last.sma50) & (df_last.close>df_last.sma50) & (df_last.volume_sma50 > 2000000 ) & ((df_last.volume_sma50 *df_last.close) > 2000000 )  & ((df_last.close/df_last.prev_close_1)>1.03)].sort_values('volume_50_rel')

    ## EP With growth
    # Query
    """close>4 AND
    sma(20)>sma(50) AND 
    close>sma(50) AND 
    sma(50,volume)>200000 AND 
    sma(50,volume)*close>2000000 
    AND open/close(1)>1.03 AND 
    market_cap>500000000 AND 
    market_cap<100000000000 AND 
    revenue_growth>0.3 AND 
    earnings_estimate_p1y_growth>0.15

    # Sort by
    volume/sma(50,volume)
    """
    if ('pocket_pivot' in scans) | ('all' in scans):
        # Bradass Pocket Pivot EOD
        d['pocket_pivot'] = df_last[(df_last.close>3) & (df_last.volume_sma50 > 200000 ) & ((df_last.volume_sma50 *df_last.close) > 2000000 ) & (df_last.close/df_last.close_prev>=1.05) & (df_last.high - df_last.close<.05 * df_last.close-0.5*df_last.low)].sort_values('volume_50_rel')
    if ('stockbee_4_gainers' in scans) | ('all' in scans):
        # stockbee 4% gainers
        d['stockbee_4_gainers'] = df_last[(df_last.close>4) & (df_last.volume_sma50 > 200000 ) & ((df_last.close / df_last.close_prev) >.04) & ((df_last.close_prev -df_last.prev_day_open)<(df_last.close-df_last.open)) & \
        ((df_last.close_prev - df_last.prev_close_2) <1.02) & (df_last.volume > df_last.volume_prev) & ((df_last.close-df_last.low)*0.7)>((df_last.high-df_last.low)*0.7)].sort_values('volume_50_rel')
    """
    if ('stockbee_combo' in scans) | ('all' in scans):
        # stockbee combo
        d['stockbee_combo'] = df_last[((df_last.close-df_last.open)>0.9) & (df_last.volume>20000) & ((df_last.close_prev - df_last.prev_day_open) < (df_last.close-df_last.open)) & \
            ((df_last.close_prev/df_last.prev_close_2)<1.02) & ((df_last.close-df_last.low)*0.7)>((df_last.high-df_last.low)*0.7) &\
                (df_last.close>4) & ((df_last.close/df_last.close_prev)>1.04) & ((df_last.close_prev/df_last.prev_close_2)<=1.02)]
                
    # stockbee ants
    if ('stockbee_ants' in scans) | ('all' in scans):
        d['stockbee_ants'] = df_last[(df_last.close>3) & (df_last.volume>100000) & (df_last.adj_close / df_last.prev_close_1 >= -1.01) & (df_last.adj_close / df_last.prev_close_1<=1.01) & (df_last.ti>1.04) ].sort_values('1m_perf') #

    # stockbee breakout 3m base
    if ('stockbee_3m_base' in scans) | ('all' in scans):
        d['stockbee_3m_base'] = df_last[(df_last['3m_perf']<=1.1) & (df_last['3m_perf']>=0.9) & (df_last['pct_change'] > 0.04) & (df_last.volume>200000) & ((df_last.volume_sma50 * df_last.close)>2000000) &((df_last.close-df_last.low)*0.7)>((df_last.high-df_last.low)*0.7)].sort_values('1m_perf') #

    # stockbee breakout 1m base
    if ('stockbee_1m_base' in scans) | ('all' in scans):
        d['stockbee_1m_base'] = df_last[(df_last['1m_perf']<=1.1) & (df_last['1m_perf']>=0.9) & (df_last['pct_change'] > 0.04) & (df_last.volume>200000) & ((df_last.volume_sma50 * df_last.close)>2000000) &((df_last.close-df_last.low)*0.7)>((df_last.high-df_last.low)*0.7)].sort_values('1m_perf') #
    """
    # darvas scan

    # darvas growth

    # insiders

    # high trend intensit
    if ('high_trend_intensity' in scans) | ('all' in scans):
        d['high_trend_intensity']  = df_last[(df_last.adj_close > 4) & (df_last.adj_close < 20) & (df_last.volume>200000) & ((df_last.volume_sma50 * df_last.close)>2000000) & (df_last.ti >1.08) & (df_last.adr>.04)].sort_values('3m_perf')
    # htf

    return d, df_last


def yahoo_prep_async(df):
    df = df.rename({'ticker':'symbol'}, axis=1)
    df = df.rename({'Unnamed: 0':'date'}, axis=1)
    df = df.rename({'adjclose':'adj_close'}, axis=1)

    df['date'] = pd.to_datetime(df.date)
    df['symbol'] = df['symbol'].str.replace('-','')
    df = df.sort_values('date',ascending=True)

    df['id']  = df['symbol'] + df['date'].astype(str).str.replace('-','')
    df.index=df['id']
    df = df.drop('id', axis=1)
    return df 


def yahoo_prep(df):
    df.columns = [c.lower() for c in df.columns]
    df = df.rename({'ticker':'symbol'}, axis=1)
    df['date'] = df.index
    df['id']  = df['symbol'] + df['date'].astype(str).str.replace('-','')
    df.columns = [c.replace(' ','_') for c in df.columns]
    df['date'] = pd.to_datetime(df.date).dt.date
    df.index = df.id 
    df = df[['id','date','symbol', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
    df = df.drop('id',axis=1)
    df = df.sort_values('id', ascending=True)
    return df


def ts_to_dt(ts):
    from datetime import date
    return date.fromtimestamp(ts)


def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    import platform
    import os
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime

def tradingview_wl_rename(path = '/Users/heiko/Downloads/TradingView'):
    import os
    import glob
    import shutil
    import datetime
    import pandas as pd
    for item in os.scandir(path):
        if (item.name[:4] != str(datetime.date.today().year)):
            if len(item.name)>=10:
                if (item.name[10]!='_'):
                    os.rename(os.path.join(item.path), os.path.join(path, str(ts_to_dt(item.stat().st_atime))[:10].replace('-','_') + '_' + item.name))
                    #print(item.name, item.path, item.stat().st_size, ts_to_dt(item.stat().st_atime))

def tradingview_symbols(path = '/Users/heiko/Downloads/TradingView', filter_data=True, write_csv=True):
    dl = []
    for item in os.scandir(path):
        #print(item.name)
        if item.name.endswith('.txt')==False:
            continue
        wl = pd.read_csv(item.path, sep=",")
        wl = list(wl.columns)
        l = []
        for s in range(0, len(wl)):
            if ':' in wl[s]:
                wl[s] = wl[s][wl[s].find(':')+1:] # remove Market
            if wl[s].startswith('###'): #detect sections with ###
                l.append(wl[s])
        for i in l:
            wl.pop(wl.index(i)) # remove sections "###"

        symbols = wl
        
        d = pd.DataFrame(symbols)
        d.columns = ['Symbol']
        d['file'] = item.name
        dl.append(d)

    res = pd.concat(dl)

    res = res[res.file.str.endswith('.txt')==True]
    res['date'] = [s[:10] for s in res['file'].tolist()]
    res['watchlist'] = [s[11:] for s in res['file'].tolist()]
    res['watchlist'] = [s.replace('.txt','') for s in res['watchlist'].tolist()]

    res['date'] = [s.replace('_','-') for s in res['date'].tolist()]
    res['date'] = pd.to_datetime(res['date'])
    res = res.sort_values(['watchlist','date'], ascending=[True,True])
    res = res.groupby(['date','watchlist','Symbol']).last()
    res = res.reset_index()

    if filter_data:
        res = res[res.groupby('watchlist').date.transform('max') == res['date']]

    if write_csv:
        res.to_csv('/Users/heiko/Dropbox/Stocks/tradingview.csv')

    return res 

def calc_adr(df):
    df['adr'] = df.groupby('symbol').apply(lambda g: np.abs(g['high'] / g['low'])).reset_index().set_index('id').iloc[:,1]
    df['adr'] = df['adr']-1
    df['adr_ma'] = df.groupby('symbol').apply(lambda g: g['adr'].rolling(window=14).mean()).reset_index().set_index('id').iloc[:,1]
    df['adr_ma'] = df['adr_ma'] * 100
    df['adr_ma'] = df['adr_ma'].round(2)
    return df


def get_atr(symbol,data=None):
    """
    data = get_atr('APPS',data=data)
    data = data.drop(columns=['atr'])

    Args:
        symbol ([type]): [description]
        data ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    import numpy as np
    import pandas_datareader as pdr
    import datetime
    import pandas as pd

    start = datetime.datetime(2020, 1, 1)
    
    if data is None:
        data = pdr.get_data_yahoo(symbol, start)
        data.columns = [c.lower() for c in data.columns]
        data['symbol'] = symbol
        data_flag = None

    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)

    atr = true_range.rolling(14).sum()/14

    data['atr'] = atr

    return data


def yahoo_prep_async(df):
    df = df.rename({'ticker':'symbol'}, axis=1)
    df = df.rename({'Unnamed: 0':'date'}, axis=1)
    df = df.rename({'adjclose':'adj_close'}, axis=1)

    df['date'] = pd.to_datetime(df.date)
    df['symbol'] = df['symbol'].str.replace('-','')
    df = df.sort_values('date',ascending=True)

    df['id']  = df['symbol'] + df['date'].astype(str).str.replace('-','')
    df.index=df['id']
    df = df.drop('id', axis=1)
    return df 


def tipranks_pre_afterhours(symbol):
    """_summary_

        ## ASYNCIO BEREITS ERSTELLT ##

        Gets the price and _volume_ for pre and aftermarket price.
        Gets also basic info as pe, marketcap etc.

        symbol='AAPL'
        
        df = tipranks_pre_afterhours(symbol)
    Args:
        symbol (_type_): _description_

    Returns:
        _type_: _description_
    """
    r = requests.get('https://market.tipranks.com/api/details/GetRealTimeQuotes?tickers='+symbol)
    r = r.json()
    preMarket = pd.DataFrame(r[0].pop('preMarket'),index=[symbol])
    afterHours =  pd.DataFrame(r[0].pop('afterHours'),index=[symbol])
    afterHours.columns = ['afterHours' +'_' + c for c in afterHours.columns]
    preMarket.columns = ['preMarket' +'_' + c for c in preMarket.columns]
    df = pd.DataFrame(r, index=[symbol])
    df = pd.concat([df,preMarket, afterHours], axis=1)
    return df 


def finviz_get_market():
    #dbname='test.db'
    from finvizfinance.screener.custom import Custom
    #from finviz.screener import Screener
    import pandabase 
    import time
    import pytz

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.ext.declarative import declarative_base

    #engine = create_engine('sqlite:///'+dbname)#, echo = True

    start = datetime.now()
    start_time = time.time()
    filters = pickle.load(open("./utils/finviz_filter.pickle","rb"))

    overview = Custom()
    overview.set_filter(filters_dict=filters['Stocks']) #gapper
    print('types')
    df = overview.screener_view(columns=range(0,100))
    #df.to_csv('finviz_df_market.csv')
    #df = pd.read_csv('finviz_df_market.csv')
    df = finviz_prep(df, True)
    data = df 

    data = data.reset_index().drop_duplicates(subset='id', keep='first').set_index('id')
    data = data.drop('index', axis=1)

    #data_screener = data_screener.reset_index().drop_duplicates(subset='id', keep='first').set_index('id')
    #data_screener = data_screener.drop('index', axis=1)

    data['loadtime'] = data['loadtime'].dt.tz_localize('Europe/Berlin').dt.tz_convert(pytz.utc) #data['loadtime'].dt.tz_convert(pytz.utc)
 
    market = pd.DataFrame({'date':start.date(),
                'change_4_perc_up':sum(df.change >= 0.04),
                'change_4_perc_down':sum(df.change <= -0.04),
                'quart_25_perc_up':sum(df['perf quart'] >= 0.25),
                'quart_25_perc_down': sum(df['perf quart'] <= -0.25),
                'month_25_perc_up': sum(df['perf month'] >= 0.25),
                'month_25_perc_down': sum(df['perf month'] <= -0.25),
                'month_5_perc_down':sum(df['perf month'] <= -.5),
                'month_5_perc_up':sum(df['perf month'] >= -.5),
                'month_13_perc_up':sum(df['perf month'] >= .13),
                'month_13_perc_down':sum(df['perf month'] <= -.13),
                'sma50_above':sum(df['sma50'] <= 0),
                'sma50_below':sum(df['sma50'] >= 0),
                'sma200_up':sum(df['sma200'] >= 0),
                'sma200_up':sum(df['sma200'] <= 0),
                'sma20_up':sum(df['sma20'] >= 0),
                'sma20_down':sum(df['sma20'] <= 0),
                'below_52w_high_10_perc':sum(df['high 52w']<=-.90),
                'below_52w_high_20_perc':sum(df['high 52w']<=-.80),
                'below_52w_high_50_perc':sum(df['high 52w']<=-.50),
                'below_50d_high_10_perc':sum(df['high 50d']<=-.90),
                'below_50d_high_20_perc':sum(df['high 50d']<=-.80),
                'below_50d_high_30_perc':sum(df['high 50d']<=-.70)
            },index=[0])

    print("--- %s seconds ---" % (time.time() - start_time))

    return data, market



def finviz_prep(df: pd.DataFrame, datatypes=True) -> pd.DataFrame:
    #print(df)
    if df is None:
        return None
    df.drop(['No.'], axis=1,errors='ignore', inplace=True)
    df.insert(2, 'Date', str(datetime.now().date()))
    df.insert(0, 'ID', df['Ticker'] +df['Date'].str.replace('-',''))
    df.rename({'Ticker':'Symbol'},axis=1, inplace=True)
    df.insert(4, 'LoadTime', datetime.utcnow())
    df.columns = df.columns.str.lower()
    #if (df['market cap'].dtype != int) | (df['market cap'].dtype=='float64'):
    #    df['market cap'] = df['market cap'].apply(convert_marketcap)

    df.columns = [i.replace(' %%',' perc') for i in df.columns]
    df.columns = [i.replace('%',' perc') for i in df.columns]
    df.columns = [i.replace('/','_') for i in df.columns]

    df = df.rename({'50d high':'high 50d', '50d low' : 'low 50d', '52w high':'high 52w', '52w low' : 'low 52w'},axis=1)

    df['ah close'] = df['ah close'].replace('-','')
    df['ah change'] = df['ah change'].replace('-','')
        
    if datatypes:
        col_str = {'symbol':str,
                    'company':str,
                    'sector':str,
                    'industry':str,
                    'country':str,
                    'earnings':str,
                    'ipo date': str
                    }
    for c in col_str:
        df[c] = df[c].astype('str')

    col_floats = [c for c in df.columns if (c not in col_str.keys()) & (c not in ['date','loadtime'])]
    col_float = dict()

    #TODO: AH Close and AH CHange is str and not float.. conversion could be better..
    print(col_floats)
    for c in col_floats: 
        print(c)
        df[c] = pd.to_numeric(df[c], errors='ignore')

    df = df.astype(col_str)
    df['ipo date'] = df['ipo date'].str.replace('-','01/01/1900')
    #df['ipo date'] = df['ipo date'].str.replace(' ','01/01/1900')

    df['ipo date'] = pd.to_datetime(df['ipo date'], format='%m/%d/%Y')
    df['earnings_time'] = [e.split('/')[1] if len(e.split('/'))>1 else '' for e in df['earnings']]
    df['earnings'] = [e.split('/')[0] for e in df.earnings]
    #data['earnings'] = data['earnings'].str.replace('-','')
    df['earnings'] = pd.to_datetime(df['earnings'], format='%b %d', errors='ignore')

    return df


def get_finviz_news(symbols, telegram=False, exportfile='news_latest.csv'):
    import time
    from datetime import timedelta
    news_list = []
    for s in list(symbols):
        print(s)
        try:
            stock = finvizfinance(s)
            df = stock.ticker_news()
            df['Symbol'] = s
            df['Datetime'] = df['Date'].copy()  
            df['Date'] = df.Date.dt.date
            df.index = df.Symbol + df.Date.astype(str)
            news_list.append(df)
        except:
            continue

    news = pd.concat(news_list)
    news.index = range(0,news.shape[0])

    df = pdr.get_data_yahoo(tickers=symbols, interval='1d', period='6mo', progress=True, group_by='ticker')
    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    #print(df)
    df = yahoo_prep(df)
    df['change'] = df['adj_close'].pct_change() * 100
    
    from datetime import timedelta
    today = datetime.now().date()
    news['today'] = today

    news = pd.merge(news, df, left_on=['Date','Symbol'], right_on=['date','symbol'], how='left')
    if exportfile is not None:
        news.to_csv(exportfile)

    news_latest = news[(news.Date - news.today)>timedelta(days=-7)]
    news_latest = news_latest.loc[news_latest['change'] >=3]
    #news_latest = news_latest.loc[(news['Gain'] !='') | (news['Loss'] !='')]

    if telegram:
        msg = ''
        for i, r in news_latest.iterrows():
            msg += r['Symbol'] + ' ' + str(r['Date']) + ' ' + '<b>'+str(r['change'])+ '</b> \n' + '<a href="'+r['Link']+'">' + r['Title'] + '</a>'+ '\n'
            if i%5==0:
                try:
                    telegram_send.send(messages=[msg], parse_mode='html')
                    msg = ''
                except:
                    print('News failed')
                time.sleep(2)
    return news, news_latest



def seekingalpha_estimates_eps(tickerid='146'):
    headers={'User-Agent': 'Mozilla/5.0'}    
    #headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    resp = requests.get(f'https://seekingalpha.com/api/v3/symbol_data/estimates?estimates_data_items=eps_normalized_actual%2Ceps_normalized_consensus_mean%2Crevenue_actual%2Crevenue_consensus_mean&period_type=quarterly&relative_periods=0%2C-1%2C-2%2C-3%2C-4%2C-5%2C-6%2C-7%2C-8%2C-9%2C-10%2C-11%2C-12%2C-13%2C-14%2C-15%2C-16%2C-17%2C-18%2C-19%2C-20%2C-21%2C-22%2C-23&ticker_ids={tickerid}', headers={'User-Agent': 'Mozilla/5.0'})
    resp = resp.json()
    keys = resp['estimates'][tickerid].keys()
    d = {}
    for k in resp['estimates'][tickerid]:
        d[k] = []
        for i in resp['estimates'][tickerid][k].keys():
            d[k].append(pd.DataFrame(resp['estimates'][tickerid][k][i][0]))
        d[k] = pd.concat(d[k])

    return d


def seekingalpha_estimates_revenue_q(tickerid='146'):
    headers={'User-Agent': 'Mozilla/5.0'}
    #headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    r = requests.get('https://seekingalpha.com/api/v3/symbol_data/estimates?estimates_data_items=revenue_actual%2Crevenue_consensus_low%2Crevenue_consensus_mean%2Crevenue_consensus_high%2Crevenue_num_of_estimates&period_type=quarterly&relative_periods=-3%2C-2%2C-1%2C0%2C1%2C2%2C3%2C4%2C5%2C6%2C7%2C8%2C9%2C10%2C11&ticker_ids='+tickerid, headers=headers)
    resp = r.json()
    #r = requests.get('https://seekingalpha.com/api/v3/symbol_data/estimates?estimates_data_items=revenue_consensus_mean%2Crevenue_num_of_estimates%2Crevenue_consensus_low%2Crevenue_consensus_high&period_type=annual&relative_periods=1%2C2&ticker_ids=146', headers={'User-Agent': 'Mozilla/5.0'})

    di = {}
    for k in resp['estimates'][tickerid].keys():
        item = resp['estimates'][tickerid][k]
        di[k] = []
        for i in item.keys():
            p = item[i][0]['period']
            d = pd.DataFrame({'effectivedate': item[i][0]['effectivedate'],
                'fiscal': str(p['fiscalyear']) + '-Q' + str(p['fiscalquarter']),
                'calendar': str(p['calendaryear']) + '-Q' + str(p['calendarquarter']),
                'periodenddate': p['periodenddate'],
                'advancedate': p['advancedate'],
                'val' : item[i][0]['dataitemvalue']}, index=[0])
            di[k].append(d)
        di[k] = pd.concat(di[k])

    return di

def seekingalpha_estimates_eps_rev_q(tickerid='146'):
    r = requests.get('https://seekingalpha.com/api/v3/symbol_data/estimates?estimates_data_items=eps_gaap_actual%2Ceps_gaap_consensus_mean%2Ceps_normalized_actual%2Ceps_normalized_consensus_mean%2Crevenue_actual%2Crevenue_consensus_mean&period_type=quarterly&relative_periods=0%2C1&revisions_data_items=eps_normalized_actual%2Ceps_normalized_consensus_mean%2Crevenue_consensus_mean&ticker_ids='+tickerid, headers={'User-Agent': 'Mozilla/5.0'})
    resp = r.json()
    di = {}
    dix = {}

    for e in resp.keys():
        print(e)
        for k in resp[e][tickerid].keys():
            if e == 'estimates':
                item = resp['estimates'][tickerid][k]
                di[k] = []
                for i in item.keys():
                    p = item[i][0]['period']
                    d = pd.DataFrame({'effectivedate': item[i][0]['effectivedate'],
                        'fiscal': str(p['fiscalyear']) + '-Q' + str(p['fiscalquarter']),
                        'calendar': str(p['calendaryear']) + '-Q' + str(p['calendarquarter']),
                        'periodenddate': p['periodenddate'],
                        'advancedate': p['advancedate'],
                        'val' : item[i][0]['dataitemvalue']}, index=[0])
                    di[k].append(d)
                di[k] = pd.concat(di[k])
            elif e == 'revisions':  
                item = resp['revisions'][tickerid][k]
                print(k)
                dix[k] = []
                for i in item.keys():
                    x = item[i][0]
                    p = item[i][0]['period']
                    d = pd.DataFrame({'asofdate': x['asofdate'],
                        'numanalysts': x['numanalysts'],
                        'numanalystsup': x['numanalystsup'],
                        'numanalystsnochange': x['numanalystsnochange'],
                        'numanalystsdown': x['numanalystsdown'],
                        'fiscal': str(p['fiscalyear']) + '-Q' + str(p['fiscalquarter']),
                        'calendar': str(p['calendaryear']) + '-Q' + str(p['calendarquarter']),
                        'periodenddate': p['periodenddate'],
                        'advancedate': p['advancedate']}, index=[0])
                    dix[k].append(d)
                dix[k] = pd.concat(dix[k])

    return di, dix

def seekingalpha_rankings(symbol):
    #headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    headers={'User-Agent': 'Mozilla/5.0'}

    r = requests.get('https://seekingalpha.com/api/v3/symbols/aapl/relative_rankings', headers=headers)
    r = r.json()
    df = pd.DataFrame(r['data']['attributes'], index=[0])
    df.insert(0, 'symbol',symbol)

    return df

def seekingalpha_growth(symbol):
    """
    # https://seekingalpha.com/symbol/AAPL/growth 
    """
    #headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    headers={'User-Agent': 'Mozilla/5.0'}

    r = requests.get('https://seekingalpha.com/api/v3/symbol_data?fields[]=revenue_growth&fields[]=revenue_growth3&fields[]=revenue_growth5&fields[]=revenueGrowth10&fields[]=ebitdaYoy&fields[]=ebitda_3y&fields[]=ebitda_5y&fields[]=ebitda_10y&fields[]=operatingIncomeEbitYoy&fields[]=operatingIncomeEbit3y&fields[]=operatingIncomeEbit5y&fields[]=operatingIncomeEbit10y&fields[]=netIncomeYoy&fields[]=netIncome3y&fields[]=netIncome5y&fields[]=netIncome10y&fields[]=normalizedNetIncomeYoy&fields[]=normalizedNetIncome3y&fields[]=normalizedNetIncome5y&fields[]=normalizedNetIncome10y&fields[]=earningsGrowth&fields[]=earningsGrowth3&fields[]=earningsGrowth5y&fields[]=earningsGrowth10y&fields[]=dilutedEpsGrowth&fields[]=dilutedEps3y&fields[]=dilutedEps5y&fields[]=dilutedEps10y&fields[]=tangibleBookValueYoy&fields[]=tangibleBookValue3y&fields[]=tangibleBookValue5y&fields[]=tangibleBookValue10y&fields[]=totalAssetsYoy&fields[]=totalAssets3y&fields[]=totalAssets5y&fields[]=totalAssets10y&fields[]=leveredFreeCashFlowYoy&fields[]=leveredFreeCashFlow3y&fields[]=leveredFreeCashFlow5y&fields[]=leveredFreeCashFlow10y&fields[]=net_interest_income_yoy&fields[]=net_interest_income_3y&fields[]=net_interest_income_5y&fields[]=net_interest_income_10y&fields[]=gross_loans_yoy&fields[]=gross_loans_3y&fields[]=gross_loans_5y&fields[]=gross_loans_10y&fields[]=common_equity_yoy&fields[]=common_equity_3y&fields[]=common_equity_5y&fields[]=common_equity_10y&slugs='+symbol,headers=headers)
    r = r.json()
    d = pd.DataFrame(r['data'][0]['attributes'], index=[0])
    d.insert(0,'symbol', r['data'][0]['id'])

    return d

def seekingalpha_ratings(symbol):
    #headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    headers={'User-Agent': 'Mozilla/5.0'}

    r = requests.get(f'https://seekingalpha.com/api/v3/symbols/{symbol}/rating/periods?filter[periods][]=0&filter[periods][]=3&filter[periods][]=6', headers=headers)
    r = r.json()
    x = pd.DataFrame(r['data'][0]['attributes'],index=[0])
    y = pd.DataFrame(r['data'][0]['attributes']['ratings'],index=[0])
    y['asDate'] = x['asDate']
    y['tickerid'] = x['tickerId']
    y.insert(0,'symbol', symbol)
    return y 

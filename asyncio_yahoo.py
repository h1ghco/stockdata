import aiohttp
import asyncio
import time
import sys
import pandas as pd
import requests
from yahoo_fin import stock_info as si
from utils.util import * 
from utils.util_clean import * 
from datetime import datetime
from datetime import timedelta
from pangres import upsert
from sqlalchemy import create_engine
import argparse
from asyncio_fetch_data import *
import dbconfig as db 

parser=argparse.ArgumentParser()
parser.add_argument("days_history", type=int, help="How many days to pull history from")
parser.add_argument("interval", type=str, help="Interval of data (1m, 30m, 1h, 1d, 1w)")
parser.add_argument("stocks", type=str, help="Options: (all, sp500, nasdaq)")
parser.add_argument("-t", "--tblname", type=str, help="Database tablename")
parser.add_argument("-e","--exportpath", type=str, help="Path to export result as parquet")
parser.add_argument("-p","--prep", type=str, help="Prepare Daily data")

args=parser.parse_args()

startdays = args.days_history
interval = args.interval
tblname = args.tblname
tblname = tblname.replace(' ','')

print(tblname)
print(args.exportpath)

if args.exportpath is None:
    exportpath=None
else: 
    exportpath = args.exportpath
stocks = args.stocks
prep = args.prep

if args.prep is None:
    prep = False

print(args)

startdays = 300
interval = '1d'
tblname = 'yahoo_daily'
exportpath = None
stocks = 'all'
prep = None
# export_path = 'yahoo_daily_1900_2000.parquet'
# datetime.today() - datetime(2005,12,31)

END_DATE_ = str(datetime.today() + timedelta(days=2))
START_DATE = str(datetime.today() + timedelta(days=-startdays))

# END_DATE_ = '1999-12-31'
# START_DATE = '1900-12-31'

engine = create_engine(f'postgresql+psycopg2://{db.user}:{db.password}@{db.raspberry}')

#if market_open():
start_time = time.time()
print('start loading..')
df = asyncio.run(main_yahoo(get_stocklist(stocks), INTERVAL_=interval, START_DATE_=START_DATE, END_DATE_=END_DATE_,
                            export_path=exportpath, tblname=tblname, prep=prep, filter_hour=True))
print("--- %s seconds ---" % (time.time() - start_time))    
    
"""
df = asyncio.run(main_yahoo(['APPS','MGNI'], INTERVAL_=interval, START_DATE_=START_DATE,END_DATE_=END_DATE_,
                                        export_path=None, tblname=None, prep=None, filter_hour=False))

df = asyncio.run(main_yahoo(['APPS','MGNI'], INTERVAL_=interval, START_DATE_=START_DATE,END_DATE_=END_DATE_,
                                        export_path=None, tblname=None, prep=False, filter_hour=False))
df['change'] = df.groupby('symbol')['adj_close'].transform(lambda x: x.pct_change())                                      
last = df.groupby('symbol').last()
last.to_sql('yahoo_latest', con=engine, if_exists='append',chunksize=1000, method='multi', index=True)
"""
"""
pandabase.to_sql(df.head(200), table_name='yahoo_hourly_test22', con=engine, how='upsert', auto_index=False, add_new_columns=True)

df = pd.read_parquet('./data/yahoo_2020_2022_1h_2.parquet')

aapl = si.get_data("aapl",interval='1m', start_date='2022-04-19', end_date='2022-04-22')
asyncio.run(main_yahoo(get_stocklist('sp500'), INTERVAL_='1m', START_DATE_=START, END_DATE_= END_DATE_, export_path='./data/yahoo_2020_2022_1m.parquet'))


df.tail(100).to_sql('yahoo_hourly_2', con=engine)


, DocsExampleTable

df.index = range(0,df.shape[0])
df.index.name = 'test'
 # default

df = pd.read_parquet('./data/yahoo_2020_2022_1m.parquet')
len(df.ticker.unique())
start_time = time.time()
engine = create_engine('postgresql+psycopg2://tradekit:yourpassword@127.0.0.1')
df.head(1).to_sql(name='yahoo_hourly_test2',con=engine,if_exists='append',method='multi',index=False)


df = get_daily_fast('./data/yahoo_2020_2022_1m.parquet',type_='long')
df.to_csv('./data/yahoo_2020_2022_1d.csv')
print("--- %s seconds ---" % (time.time() - start_time))     
from talib import *
import talib
from talib_pattern import candlestick_patterns

for pattern in candlestick_patterns:
    pattern_function = getattr(talib, pattern)
    df[pattern.lower()] = pattern_function(df['open'], df['high'], df['low'], df['close'])
    
"""
"""
yahoo_fin = ()
import yfinance as yf
aapl = yf.Ticker('AAPL')
test = aapl.history(period='1y', interval='1h')
df2 = pd.read_parquet('./data/yahoo_2020_2022_1h.parquet')
df2['date'] = pd.to_datetime(df2.index.date)
df2['time'] = df2.index.time
first_hour = df2.groupby(['ticker',df2.date.dt.date]).first(1).reset_index()
first_hour.ticker
first_hour[first_hour.ticker=='ZY']['volume'].describe()
symbols = df2.ticker.unique()

stocklist = get_stocklist('all')
stocklist = set(symbols) ^ set(stocklist)

l = []
for s in stocklist:
    l.append(pdr.get_data_yahoo(ticker=s, interavl='1h', period='max'))


zyne = df[df.symbol.isin(['ZYNE','ZY'])]

df['signal'] = df[['symbol','low','high']].groupby('symbol').apply(lambda g: (g['low'] > g['low'].shift(1)) & (g['high'] < g['high'].shift(1))).reset_index().set_index('id').iloc[:,1]
df['signal1'] = df[['symbol','low','high','signal']].groupby('symbol').apply(lambda g: (g['signal'].shift(1)==True) & \
                                    (g['low'] > g['low'].shift(2)) & (g['high'] < g['high'].shift(2))).reset_index().set_index('id').iloc[:,1]
df['signal2'] = df[['symbol','low','high','signal','signal1']].groupby('symbol').apply(lambda g: (g['signal'].shift(2)==True) & (g['signal1'].shift(1)==True) &
                                    (g['low'] > g['low'].shift(3)) & (g['high'] < g['high'].shift(3))).reset_index().set_index('id').iloc[:,1]
df['signal3'] = df[['symbol','low','high','signal','signal1','signal2']].groupby('symbol').apply(lambda g: (g['signal'].shift(3)==True) & (g['signal1'].shift(2)==True) & (g['signal2'].shift(1)==True) &
                                    (g['low'] > g['low'].shift(4)) & (g['high'] < g['high'].shift(4))).reset_index().set_index('id').iloc[:,1]
df['signal4'] = df[['symbol','low','high','signal','signal1','signal2','signal3']].groupby('symbol').apply(lambda g: (g['signal'].shift(4)==True) & (g['signal1'].shift(3)==True) & (g['signal2'].shift(2)==True) & (g['signal3'].shift(1)==True) &
                                    (g['low'] > g['low'].shift(5)) & (g['high'] < g['high'].shift(5))).reset_index().set_index('id').iloc[:,1]
df[(df.signal4==True) & (df.close>df['high'].shift(6))]
"""
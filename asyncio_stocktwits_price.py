import aiohttp
import asyncio
import time
import pandas as pd
import requests
from yahoo_fin import stock_info as si
import time
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from utils.util_clean import * 
from utils.util import * 
from asyncio_fetch_data import *
start_time = time.time()
today = str(datetime.now().date())
today = today.replace('-','_')
pqwriter =None

#df = asyncio.run(main_stocktwits(['APPS','MGNI'], tblname='stocktwits_hourly')) #get_stocklist('sp500')

if trading_day():
    while True:
        now = datetime.now()
        if now.hour >23:
            break
        if (now.hour) >9 & (now.hour <=15):
            df = asyncio.run(main_stocktwits(get_stocklist('tradingview_1'),tblname='stocktwits_hourly'))
            time.sleep(60*60)

"""

df.to_sql(name='stocktwits',con=engine, if_exists='append',method='multi', index=True)

z.to_sql(name='test', con=engine, if_exists='append',chunksize=25000, method='multi', index=True)
df.index = df.symbol
df = df.drop(columns=['symbol'], axis=1)
df['timestamp'] = df['timestamp'].dt.tz_localize('Europe/Berlin').dt.tz_convert(pytz.utc)

#df['timestamp'].dt.tz_localize('Europe/Berlin').dt.tz_convert(pytz.utc)
pandabase.to_sql(df, table_name='finviz_screen', con=engine, how='upsert', auto_index=False,add_new_columns=True)

print("--- %s seconds ---" % (time.time() - start_time))
"""
""" 
        file_path = '/stocktwits/stocktwits_' + str(datetime.today())
        file_path = file_path.replace(':','-')
        file_path = file_path.replace('.','-')
        file_path = file_path.replace(' ','-')
        file_path = './data' + file_path + '.parquet'
"""
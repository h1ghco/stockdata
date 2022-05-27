from re import X
import aiohttp
import asyncio
import time
import pandas as pd
import requests
from yahoo_fin import stock_info as si
import time
from utils.util_clean import * 
from sqlalchemy import create_engine

#stocklist = si.tickers_sp500() #+ si.tickers_nasdaq()
#stocklist = ['MGNI','APPS','AAPL','GOOGL']  
start_time = time.time()
today = str(datetime.now().date())
today = today.replace('-','_')

"""
df = asyncio.run(main_tip_pre(stocklist=get_stocklist('all'),export_path='./data/tipranks_premarket.parquet'))
engine = create_engine('postgresql+psycopg2://tradekit:yourpassword@127.0.0.1')
df['timestamp'] = datetime.now()
df.to_sql(name='tipranks_premarket',con=engine,if_exists='append',method='multi', index=True)
#print("--- %s seconds ---" % (time.time() - start_time))
#df = pd.read_parquet(today+'_tipranks.parquet')




"""
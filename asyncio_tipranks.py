import pandas as pd
import requests
import urllib
import bs4 
from datetime import datetime
import time 
import random
from yahoo_fin import stock_info as si
from datetime import datetime
import os
import sys
from utils.util_clean import *

import bs4
import aiohttp
import asyncio
import time
import pandas as pd
import requests
import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from yahoo_fin import stock_info as si
import numpy as np
from utils.util_clean import *
import datetime
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

start_time = time.time()
err = []
tdy = str(datetime.datetime.now().date())
async def get_tipranks(session, site):
    print(site)
    async with session.get(site, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}) as resp:
        text = await resp.read()
        #print(resp.content)
        #html = r.text
        #print(html)
        #await asyncio.sleep(0.5)
        symbol = site.split('/')
        symbol = symbol[-2]
        #print(resp)
        try:
            soup = bs4.BeautifulSoup(text.decode('utf-8'), 'html5lib')
            x = soup.find('div',class_="bt1_solid borderColorwhite-8 w12 bgwhite-5 px4 py3 laptop_py3 laptop_h_pxauto flexrsc displayflex h_pxsmall60").find('table')
            d = pd.read_html(str(x))[0]
            d.columns = ['high','avg','low']
        except:
            print('Erorr: ', site)
        try:
            d['high'] = d['high'].str.replace('Highest Price Target$','', regex=False)
            d['avg'] = d['avg'].str.replace('Average Price Target$','', regex=False)
            d['low'] = d['low'].str.replace('Lowest Price Target$','', regex=False)
            d['low'] = float(d['low'].replace(',',''))
            d['high'] = float(d['high'].replace(',',''))
            d['avg'] = float(d['avg'].replace(',',''))
            #d['price'] = soup.find('div', class_="flexccc mt3 displayflex colorpale shrink0 lineHeight2 fontSize2 ml2 ipad_fontSize3").contents[0].text 
            #d['price'] =  float(d['price'].str.replace('$',''))
            p = soup.find('div', class_="flexccc mt4 displayflex shrink0 fontWeightsemibold mobile_w12").text # !!!!!!
            buy = p[p.find('gs')+2:p.find('Buy')]
            hold = p[p.find('Buy')+3:p.find('Hold')]
            sell = p[p.find('Hold')+4:p.find('Sell')]
            d['buy'] = buy
            d['hold'] = hold
            d['sell'] = sell
            d.insert(0,'id', tdy.replace('-','')+symbol)
            d.insert(1,'symbol', symbol)
            d.insert(2,'date', tdy)
            d['text'] = soup.find('div', class_="flexcb_ bgwhite h12 w12 px0 displayflex positionrelative py3").text
        except:
            d = None

        return d

async def crawl_tipranks(stocklist, export_path='./tipranks_analysts.parquet'):
    async with aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}) as session:
        tasks = []
        symbols = get_stocklist('sp500')
    
        for symbol in stocklist:#:
            site = 'https://www.tipranks.com/stocks/'+symbol+'/forecast'
            print(site)
            tasks.append(asyncio.ensure_future(get_tipranks(session, site)))
        original_pokemon = await asyncio.gather(*tasks)
        df = pd.concat(original_pokemon)
        df.to_parquet(export_path)
        #df.to_csv('nasdaq_yahoo_premarket2.csv')
        #df = pa.table(df)
        #pq.write_table(df, "./data/tipranks_test.parquet")
        print('Done')
        return df

df = asyncio.run(crawl_tipranks(get_stocklist('sp500'), export_path='./tipranks_analysts.parquet'))
print("--- %s seconds ---" % (time.time() - start_time))
#df = pd.read_parquet("./data/tipranks_test.parquet")

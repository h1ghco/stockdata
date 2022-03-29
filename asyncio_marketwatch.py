import pandas as pd
import requests
import urllib
import bs4 
import time
from utils.util_clean import * 
#headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
#symbol='AAPL'
l = []
start_time = time.time()


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
headers = {'User-Agent': 'Mozilla/5.0'}

start_time = time.time()
err = []
tdy = str(datetime.datetime.now().date())
async def get_marketwatch(session, site):
    print(site)
    async with session.get(site, headers={'User-Agent': 'Mozilla/5.0'}) as resp:
        text = await resp.read()
        #print(resp.content)
        #html = r.text
        #print(html)
        #await asyncio.sleep(0.5)
        symbol = site.split('/')[-1]
        symbol = symbol[:symbol.find('?')]
        #print(resp)
        #try:
        soup = bs4.BeautifulSoup(text.decode('utf-8'), 'html5lib')
        #soup = bs4.BeautifulSoup(text.text, 'html5lib')
        try:
            columns = soup.find('div',class_="intraday__close").find('table').find_all('th')
            input = soup.find('div',class_="intraday__close").find('table').find_all('td')

            dailyvol = soup.find('mw-rangebar', class_='element element--range range--volume').attrs
            dailyvol = pd.DataFrame(dailyvol).head(1)
            dailyvol = dailyvol.drop(columns=['class','quote-channel'])
            dailyvol.columns = [c.replace('-','') + '_volume' for c in dailyvol.columns]
            dailyvol.insert(0,'symbol',symbol)

            dailyrange = soup.find('mw-rangebar', class_='element element--range range--daily').attrs
            dailyrange = pd.DataFrame(dailyrange).head(1)
            dailyrange = dailyrange.drop(columns=['precision','quote-channel','class'])
            dailyrange.insert(0,'symbol',symbol)

            s = {}
            for i in range(0,len(columns)):
                s[columns[i]] = input[i].text
                
            dx = pd.DataFrame(s, index=[0])

            dx.columns = [str(c.text).lower() for c in dx.columns]
            dx.columns = [c.replace('chg %','change_perc') for c in dx.columns]
            dx.columns = [c.replace('chg','change') for c in dx.columns]
            dx.insert(0,'symbol',symbol)
            dx['volume'] = soup.find("div", class_="intraday__volume").find('span',class_='volume__value').text

            if 'M' in dx['volume'].values[0]:
                dx['volume'] = float(dx['volume'].str.replace('M','')) * 1000000
            if dx['volume'].dtype=='O':
                if 'K' in dx['volume'].values[0]:
                    dx['volume'] = float(dx['volume'].str.replace('K','')) * 100000

            dx['hours'] = soup.find("div", class_="intraday__volume").find('span',class_='volume__label').text
            dx['time'] = soup.find("div", class_="intraday__timestamp").find('span',class_='timestamp__time').text
            dx['price'] = soup.find("h2", class_="intraday__price").contents[-2].text
            dx['price'] = pd.to_numeric(dx['price'].str.replace(',',''))
            dx['close'] = dx['close'].str.replace('$','', regex=False)
            dx['close'] = dx['close'].str.replace(',','')
            dx['close'] = pd.to_numeric(dx['close'])
            dx['afterpre_change_perc'] = soup.find('span',class_='change--percent--q').text
            dx['afterpre_change_perc'] = pd.to_numeric(dx['afterpre_change_perc'].str.replace('%',''))
            dx['change_perc'] = pd.to_numeric(dx['change_perc'].str.replace('%',''))
            dx['change'] = pd.to_numeric(dx['change'])

            dx = pd.merge(dx, dailyrange, left_on='symbol', right_on='symbol')
            dx = pd.merge(dx, dailyvol, left_on='symbol', right_on='symbol')
        except:
            print('Symbol: ', symbol)
            dx = None
        await asyncio.sleep(0.30)
        return dx

async def crawl_marketwatch(symbols, export_path='./data/marketwatch_test.parquet'):
    connector = aiohttp.TCPConnector(limit=25)
    async with aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0'},connector=connector) as session:
        tasks = []    
        for symbol in symbols:#:
            site = 'https://www.marketwatch.com/investing/stock/'+symbol+'?mod=over_search'
            tasks.append(asyncio.ensure_future(get_marketwatch(session, site)))
        original_pokemon = await asyncio.gather(*tasks)
        df = pd.concat(original_pokemon)
        df.to_csv(export_path+'.csv')
        print("--- %s seconds ---" % (time.time() - start_time))

        df.to_parquet(export_path)
        #df.to_csv('nasdaq_yahoo_premarket2.csv')
        #df = pa.table(df)
        #pq.write_table(df, export_path)
        print('Done')
        
asyncio.run(crawl_marketwatch(get_stocklist('sp500'), export_path='./data/marketwatch_test.parquet'))
print("--- %s seconds ---" % (time.time() - start_time))
#df = pd.read_parquet("./data/marketwatch_test.parquet")
#df = pd.read_csv('./data/marketwatch_test.parquet.csv')

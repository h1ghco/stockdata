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

start_time = time.time()
END_DATE_ = str(datetime.today() + timedelta(days=2))
#datetime.timestamp(end_date)
#datetime.utcnow().timestamp()
def build_url(ticker, start_date = None, end_date = None, interval = "1d"):
    base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"

    if interval not in ("1d", "1wk", "1mo", "1m",'60m','1h'):
        raise AssertionError("interval must be of of '1d', '1wk', '1mo', or '1m'")

    if end_date is None:  
        end_seconds = int(pd.Timestamp("now").timestamp())
        
    else:
        end_seconds = int(pd.Timestamp(end_date).timestamp())
        
    if start_date is None:
        start_seconds = 7223400    
        
    else:
        start_seconds = int(pd.Timestamp(start_date).timestamp())
    
    site = base_url + ticker
    
    params = {"period1": start_seconds, "period2": end_seconds,
              "interval": interval.lower(), "events": "div,splits"}
    
    return site, params


async def get_yahoo(session, site, interval, index_as_date):
    #print(site)
    async with session.get(site) as resp:
        await asyncio.sleep(0.5)
        data = await resp.json()
        #print(site)
        global l_err
        l_err = []
        try:     
            # get open / high / low / close data
            frame = pd.DataFrame(data["chart"]["result"][0]["indicators"]["quote"][0])
            #print(frame)
            # get the date info
            temp_time = data["chart"]["result"][0]["timestamp"]

            if interval not in ["1m","60m","1h"]:
                # add in adjclose
                frame["adjclose"] = data["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"]   
                frame.index = pd.to_datetime(temp_time, unit = "s", utc=False)
                frame.index = frame.index.map(lambda dt: dt.floor("d"))
                frame = frame[["open", "high", "low", "close", "adjclose", "volume"]]
                    
            else:
                frame.index = pd.to_datetime(temp_time, unit = "s", utc=False)
                frame = frame[["open", "high", "low", "close", "volume"]]
                #print(frame)
            ticker = site.split('/')[6:7]
            ticker = ticker[0].split('?')[0]
            
            frame['ticker'] = ticker
            if not index_as_date:  
                frame = frame.reset_index()
                frame.rename(columns = {"index": "date"}, inplace = True)
        except:
            frame = None
            l_err.append(site)
        return frame


async def main_yahoo(stocklist, export_path = './data/yahoo_test.parquet', INTERVAL_='1d', START_DATE_='2021-01-01', END_DATE_=END_DATE_, index_as_date=True):


    start_time = time.time()
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    async with aiohttp.ClientSession() as session:
    
        tasks = []

        print('No of symbols: ' + str(len(stocklist)))

        for symbol in stocklist:#:
            site, params = build_url(ticker=symbol, start_date=START_DATE_, end_date=END_DATE_, interval=INTERVAL_)
            site += '?'
            for k, v in params.items():
                site += k +'='+str(v) +'&'
            tasks.append(asyncio.ensure_future(get_yahoo(session, site, INTERVAL_, index_as_date)))

        original_pokemon = await asyncio.gather(*tasks)

        d = pd.concat(original_pokemon)
        d.to_parquet(export_path)

        """
        if len(sys.argv)>1:
            df = pd.concat([df,d])
            df.to_parquet(sys.argv[1])
            print(len(df.ticker.unique()))
        else:
            print(len(d.ticker.unique()))
            d.to_parquet(sys.argv[0])
        """  

asyncio.run(main_yahoo(get_stocklist('all'), export_path='./data/yahoo_test_2021.parquet'))
print("--- %s seconds ---" % (time.time() - start_time))



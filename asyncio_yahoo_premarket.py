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
start_time = time.time()

async def get_yahoo_pre(session, site):
    print(site)
    async with session.get(site) as resp:
        #await asyncio.sleep(0.5)
        #if not resp.ok:
        #    df = None
        #    return df
        json_result = await resp.json()
        #print(json_result)
        #df = pd.DataFrame(columns=['premarket','symbol','loadtime'])
        
        try:     
            info = json_result["quoteResponse"]["result"]
            info = info[0]
            symbol = site.split('=')[1]

            if "postMarketPrice" in info:
                postmarket = info["postMarketPrice"]
                postMarketChangePercent = info['postMarketChangePercent']
                postMarketTime = info['postMarketTime']
                postMarketPrice = info['postMarketPrice']
                postMarketChange = info['postMarketChange']
            else:
                postmarket = None
                postMarketChangePercent = None
                postMarketTime = None
                postMarketPrice = None
                postMarketChange  = None

            if "preMarketPrice" in info:
                premarket = info["preMarketPrice"]
                premarket = info["preMarketPrice"]
                preMarketChangePercent = info['preMarketChangePercent']
                preMarketTime = info['preMarketTime']
                preMarketPrice = info['preMarketPrice']
                preMarketChange = info['preMarketChange']
            else:
                premarket = None
                preMarketChangePercent = None
                preMarketTime = None
                preMarketPrice = None
                preMarketChange  = None
            
            # get open / high / low / close data
            df = pd.DataFrame({
                'symbol':symbol,
                'postmarket':postmarket,
                'postMarketChangePercent': postMarketChangePercent,
                'postMarketTime': postMarketTime,
                'postMarketPrice':postMarketPrice,
                'postMarketChange':postMarketChange,
                'premarket': premarket,
                'preMarketChangePercent': preMarketChangePercent,
                'preMarketTime': preMarketTime,
                'preMarketPrice':preMarketPrice,
                'preMarketChange':preMarketChange,
                'marketState' :info['marketState'],
                'fiftyDayAverage' :info['fiftyDayAverage'], 
                'fiftyDayAverageChange' :info['fiftyDayAverageChange'], 
                'fiftyDayAverageChangePercent' :info['fiftyDayAverageChangePercent'], 
                'twoHundredDayAverage' :info['twoHundredDayAverage'], 
                'twoHundredDayAverageChange' :info['twoHundredDayAverageChange'], 
                'twoHundredDayAverageChangePercent' :info['twoHundredDayAverageChangePercent'],
                'fiftyTwoWeekLowChangePercent' :info['fiftyTwoWeekLowChangePercent'], 
                'fiftyTwoWeekRange' :info['fiftyTwoWeekRange'], 
                'fiftyTwoWeekHighChange': info['fiftyTwoWeekHighChange'], 
                'fiftyTwoWeekHighChangePercent': info['fiftyTwoWeekHighChangePercent'],
                'marketCap' :info['marketCap'],
                'shortName' :info['shortName'],
                'forwardPE' :info['forwardPE'],
                'trailingPE' :info['trailingPE'],
                'priceToBook' :info['priceToBook'],
                'bookValue' :info['bookValue'],
                'epsCurrentYear' :info['epsCurrentYear'],
                'epsTrailingTwelveMonths' :info['epsTrailingTwelveMonths'],
                'epsForward' :info['epsForward'],
                'priceEpsCurrentYear' :info['priceEpsCurrentYear'],
                'sharesOutstanding' :info['sharesOutstanding'],
                'exchangeTimezoneName' :info['exchangeTimezoneName'],
                'exchangeTimezoneShortName' :info['exchangeTimezoneShortName'],
                'gmtOffSetMilliseconds' :info['gmtOffSetMilliseconds'],
                'regularMarketPrice' :info['regularMarketPrice'],
                'regularMarketTime' :info['regularMarketTime'],
                'regularMarketChange' :info['regularMarketChange'],
                'regularMarketPreviousClose' :info['regularMarketPreviousClose'],
                'regularMarketOpen' :info['regularMarketOpen'],
                'regularMarketDayHigh' :info['regularMarketDayHigh'],
                'regularMarketDayLow' :info['regularMarketDayLow'],
                'regularMarketVolume' :info['regularMarketVolume'],
                'averageDailyVolume3Month':info['averageDailyVolume3Month'],
                'averageDailyVolume10Day':info['averageDailyVolume10Day'],
                'loadtime':datetime.datetime.now(),
                }, index=[0])
            #print(df)
            await asyncio.sleep(.3)
        except:
            df = None
        return df

async def craw_yahoo_pre(symbols, export_path="./data/yahoo_premarket.parquet"):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for symbol in symbols:#:
            site = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=" + symbol
            print(site)
            tasks.append(asyncio.ensure_future(get_yahoo_pre(session, site)))
        original_pokemon = await asyncio.gather(*tasks)
        df = pd.concat(original_pokemon)
        #df.to_csv('nasdaq_yahoo_premarket2.csv')
        df = pa.table(df)
        pq.write_table(df, export_path)
        print('Done')
        
asyncio.run(craw_yahoo_pre(get_stocklist('tradingview_1'), "./data/yahoo_premarket.parquet"))
print("--- %s seconds ---" % (time.time() - start_time))
#df = pd.read_parquet("./data/yahoo_premarket.parquet")

"""
tdy = str(datetime.datetime.today().date())
earnings = pd.DataFrame(si.get_earnings_for_date(tdy))
symbols = earnings.ticker.unique()

df = df[(df.marketCap>500000000) & (df.averageDailyVolume3Month>700000)]
df[(df.fiftyDayAverageChangePercent>0) & (df.twoHundredDayAverageChangePercent>0)]

df[df.postMarketChangePercent>4].sort_values('postMarketChangePercent')
df[df.preMarketChangePercent>4].sort_values('preMarketChangePercent')
"""
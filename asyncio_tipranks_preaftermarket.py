from re import X
import aiohttp
import asyncio
import time
import pandas as pd
import requests
from yahoo_fin import stock_info as si
import time
from utils.util_clean import * 
#stocklist = si.tickers_sp500() #+ si.tickers_nasdaq()
#stocklist = ['MGNI','APPS','AAPL','GOOGL']  
start_time = time.time()
today = str(datetime.now().date())
today = today.replace('-','_')

def mill_to_float(x):
    try:
        if type(x) == str:
            if 'M' in x:
                x = x.replace('M','')
                x=float(x)
                x=x*1000000
            elif 'K' in x:
                x = x.replace('K','')
                x=float(x)
                x=x*1000
            elif 'B' in x:
                x = x.replace('B','')
                x = x.replace(',','')
                x=float(x)
                x=x*100000000
            elif x == '':
                return np.nan
    except:
        print(x, type(x))
    return x


async def get_tipranks_pre(session, url):
    async with session.get(url) as resp:
        r = await resp.json()
        try:
            symbol = url.split('=')[1]
            preMarket = pd.DataFrame(r[0].pop('preMarket'),index=[symbol])
            afterHours =  pd.DataFrame(r[0].pop('afterHours'),index=[symbol])
            afterHours.columns = ['afterHours' +'_' + c for c in afterHours.columns]
            preMarket.columns = ['preMarket' +'_' + c for c in preMarket.columns]
            df = pd.DataFrame(r, index=[symbol])
            df = pd.concat([df,preMarket, afterHours], axis=1)
        except:
            df = None
        await asyncio.sleep(.3)
        return df


async def main_tip_pre(stocklist, export_path='./data/tipranks_premarket.parquet'):
    async with aiohttp.ClientSession() as session:

        tasks = []
        print(len(stocklist))
        for symbol in stocklist:#['APPS','MSFT','MGNI','GOOGL']:
            url = f'https://market.tipranks.com/api/details/GetRealTimeQuotes?tickers='+symbol
            tasks.append(asyncio.ensure_future(get_tipranks_pre(session, url)))

        original_pokemon = await asyncio.gather(*tasks)
        df = pd.concat(original_pokemon)
        df['afterHours_volume'] = pd.to_numeric(df['afterHours_volume'].apply(mill_to_float), errors='coerce').round()
        df['preMarket_volume'] = pd.to_numeric(df['preMarket_volume'].apply(mill_to_float), errors='coerce').round()
        df['volume'] = pd.to_numeric(df['volume'].apply(mill_to_float), errors='coerce').round()
        df['marketCap'] = pd.to_numeric(df['marketCap'].apply(mill_to_float), errors='coerce').round()
        df.to_parquet(export_path)
        print('Done')

        return df

df = asyncio.run(main_tip_pre(stocklist=['APPS','MGNI'],export_path='./data/tipranks_premarket.parquet'))
print("--- %s seconds ---" % (time.time() - start_time))
#df = pd.read_parquet(today+'_tipranks.parquet')





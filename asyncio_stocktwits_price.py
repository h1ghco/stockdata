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
    
start_time = time.time()
today = str(datetime.now().date())
today = today.replace('-','_')
pqwriter =None

async def get_stocktwits(session, url):
    async with session.get(url) as resp:
        pokemon = await resp.json()
        try:
            pokemon['symbol'].pop('price_data')
            pokemon['symbol'].pop('aliases') 
            pokemon = pd.DataFrame(pd.DataFrame(pokemon['symbol'], index=[0]), index=[0])
        except:
            pokemon = None
        return pokemon


async def main_stocktwits(stocklist, filepath):

    async with aiohttp.ClientSession() as session:

        tasks = []
        for symbol in stocklist:#['APPS','MSFT','MGNI','GOOGL']:
            url = f'https://api.stocktwits.com/api/2/symbols/with_price/{symbol}.json?extended=true'
            #url = f'https://ql.stocktwits.com/pricedata?fundamentals=true&symbol={symbol}'
            tasks.append(asyncio.ensure_future(get_stocktwits(session, url)))

        original_pokemon = await asyncio.gather(*tasks)
        df = pd.concat(original_pokemon)
        df['loadtime'] = int(time.time())

        #df.to_csv(today+'_stocktwits.csv')
        table = pa.Table.from_pandas(df)
        pqwriter = pq.ParquetWriter(filepath, table.schema)
        pqwriter.write_table(table)
        pqwriter.close()
        print('Done')
        return df

df = asyncio.run(main_stocktwits(get_stocklist('sp500'),'./data/stocktwits_price_20220325.parquet'))
print("--- %s seconds ---" % (time.time() - start_time))

""" 
        file_path = '/stocktwits/stocktwits_' + str(datetime.today())
        file_path = file_path.replace(':','-')
        file_path = file_path.replace('.','-')
        file_path = file_path.replace(' ','-')
        file_path = './data' + file_path + '.parquet'
"""
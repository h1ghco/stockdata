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
            pokemon = pokemon['symbol']
            pokemon.pop('aliases')
            data = pd.DataFrame(pokemon.pop('price_data'), index=[0])
            #print(data)
            data2 = pd.DataFrame(pokemon, index=[0])
            #print(data2)
            pokemon = pd.concat([data, data2], axis=1)
            #print(pokemon)
            pokemon.columns = [c.lower() for c in pokemon.columns]
        except:
            pokemon = None
        return pokemon


async def main_stocktwits(stocklist, export_path):

    async with aiohttp.ClientSession() as session:

        tasks = []
        for symbol in stocklist:#['APPS','MSFT','MGNI','GOOGL']:
            url = f'https://api.stocktwits.com/api/2/symbols/with_price/{symbol}.json?extended=true'
            #url = f'https://ql.stocktwits.com/pricedata?fundamentals=true&symbol={symbol}'
            tasks.append(asyncio.ensure_future(get_stocktwits(session, url)))

        data = await asyncio.gather(*tasks)
        df = pd.concat(data)
        df['loadtime'] = int(time.time())

        #df.to_csv(today+'_stocktwits.csv')
        table = pa.Table.from_pandas(df)
        pqwriter = pq.ParquetWriter(export_path, table.schema)
        pqwriter.write_table(table)
        pqwriter.close()
        print('Done')
        return df

df = asyncio.run(main_stocktwits(get_stocklist('sp500'),export_path='./data/stocktwits_20220325.parquet'))
print("--- %s seconds ---" % (time.time() - start_time))

""" 
        file_path = '/stocktwits/stocktwits_' + str(datetime.today())
        file_path = file_path.replace(':','-')
        file_path = file_path.replace('.','-')
        file_path = file_path.replace(' ','-')
        file_path = './data' + file_path + '.parquet'
"""
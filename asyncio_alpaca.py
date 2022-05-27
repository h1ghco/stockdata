from enum import Enum
import time
import alpaca_trade_api as tradeapi
import asyncio
import asyncio
import aiohttp
from aiohttp.client import ClientTimeout
import os
import pandas as pd
import sys
from alpaca_trade_api.rest import TimeFrame, URL, TimeFrameUnit
from alpaca_trade_api.rest_async import gather_with_concurrency, AsyncRest
from utils.util_clean import *
from utils.util import *
import numpy as np
from sqlalchemy import create_engine

NY = 'America/New_York'

engine = create_engine('postgresql+psycopg2://tradekit:yourpassword@127.0.0.1')
start_date = pd.Timestamp('2022-04-10', tz=NY).date().isoformat()
end_date = pd.Timestamp('2022-04-22', tz=NY).date().isoformat()
symbols = set(get_stocklist('sp500'))

endpoint='https://paper-api.alpaca.markets'
api_key='PK88Q6YTX1K4J783EPHR'
secret='2oTh6GvGX1J8YLzRooOtjhPR3ZYJfqV2RUIaxtLb'

class DataType(str, Enum):
    Bars = "Bars"
    Trades = "Trades"
    Quotes = "Quotes"

def get_data_method(data_type: DataType):
    if data_type == DataType.Bars:
        return rest.get_bars_async
    elif data_type == DataType.Trades:
        return rest.get_trades_async
    elif data_type == DataType.Quotes:
        return rest.get_quotes_async
    else:
        raise Exception(f"Unsupoported data type: {data_type}")


async def get_historic_data_base(symbols, data_type: DataType, start, end,
                                 timeframe: TimeFrame = None):
    """
    base function to use with all
    :param symbols:
    :param start:
    :param end:
    :param timeframe:
    :return:
    """
    major = sys.version_info.major
    minor = sys.version_info.minor
    if major < 3 or minor < 6:
        raise Exception('asyncio is not support in your python version')
    msg = f"Getting {data_type} data for {len(symbols)} symbols"
    msg += f", timeframe: {timeframe}" if timeframe else ""
    msg += f" between dates: start={start}, end={end}"
    print(msg)
    step_size = 1000
    results = []
    for i in range(0, len(symbols), step_size):
        tasks = []
        for symbol in symbols[i:i+step_size]:
            args = [symbol, start, end, timeframe.value] if timeframe else \
                [symbol, start, end]
            tasks.append(get_data_method(data_type)(*args))

        if minor >= 8:
            results.extend(await asyncio.gather(*tasks, return_exceptions=True))
        else:
            results.extend(await gather_with_concurrency(500, *tasks))

    bad_requests = 0
    for response in results:
        if isinstance(response, Exception):
            print(f"Got an error: {response}")
        elif not len(response[1]):
            bad_requests += 1

    print(f"Total of {len(results)} {data_type}, and {bad_requests} "
          f"empty responses.")

    return results


async def get_historic_bars(symbols, start, end, timeframe: TimeFrame):
    df = await get_historic_data_base(symbols, DataType.Bars, start, end, timeframe)
    return df

async def get_historic_trades(symbols, start, end, timeframe: TimeFrame):
    await get_historic_data_base(symbols, DataType.Trades, start, end)


async def get_historic_quotes(symbols, start, end, timeframe: TimeFrame):
    await get_historic_data_base(symbols, DataType.Quotes, start, end)


async def main(symbols, start_date, end_date, interval=TimeFrame(1, TimeFrameUnit.Minute)):
    #timeframe: TimeFrame = TimeFrame.Minute
    df = await get_historic_bars(symbols, start_date, end_date, interval)
    #await get_historic_trades(symbols, start, end, timeframe)
    #await get_historic_quotes(symbols, start, end, timeframe)
    return df

rest = AsyncRest(key_id=api_key,
                    secret_key=secret)

start_time = time.time()
#symbols = [el.symbol for el in api.list_assets(status='active')]
#symbols = symbols[:100]
#df = asyncio.run(main(symbols[:100]))

def run_main(symbols=list(symbols)[:101], no = 100, start_date = start_date, end_date = end_date, write_db=True):
    l = []
    cnt= 0
    engine = create_engine('postgresql+psycopg2://tradekit:yourpassword@127.0.0.1')

    for x in range(0,int(np.ceil(len(symbols)/no))):
        if x == 0:
            print(x,x+no)
            df = asyncio.run(main(symbols[x:x+no], start_date, end_date))
            for r in df:
                try:
                    df = r[1]
                    df['symbol'] = r[0]
                    l.append(df)
                except:
                    continue
            time.sleep(60)
            print(df.head())
        else:
            print(cnt,cnt+no)
            df = asyncio.run(main(symbols[x*no:(x*no)+no], start_date, end_date))
            for r in df:
                try:
                    df = r[1]
                    df['symbol'] = r[0]
                    l.append(df)
                except:
                    continue
            time.sleep(60)
            print(df.head())
        cnt = cnt+no
    print(f"took {time.time() - start_time} sec")
    d = pd.concat(l)
    if write_db:
        d.to_sql(name='alpaca_minutes',con=engine,if_exists='append',chunksize=1000,method='multi', index=True)
        
    return d 


df = run_main()

'''

y = pd.read_sql_query("""SELECT * FROM yahoo_daily WHERE date >= '2022-04-22'""", con=engine)

import psycopg2
connection = psycopg2.connect(host='127.0.0.1', database='tradekit', user='tradekit', password='yourpassword')
cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
cursor.execute("""DELETE FROM alpaca_minutes""")
connection.commit()
cursor.execute("""SELECT * FROM yahoo_daily WHERE date >= '2022-04-22'""")

l = []
for r in df:
    try:
        df = r[1]
        df['symbol'] = r[0]
        l.append(df)
    except:
        continue

x = pd.concat(l)
s = x.symbol.unique()

'''
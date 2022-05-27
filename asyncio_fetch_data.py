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
from pangres import upsert, DocsExampleTable
import pandabase 
from sqlalchemy import create_engine
from datetime import timedelta

engine = create_engine('postgresql+psycopg2://tradekit:yourpassword@192.168.0.174')

start_time = time.time()
today = str(datetime.now().date())
today = today.replace('-','_')
pqwriter =None

async def get_stocktwits(session, url):
    async with session.get(url) as resp:
        pokemon = await resp.json()
        try:
            #print(pokemon)
            pokemon = pokemon['symbol']
            pokemon.pop('aliases')
            data = pd.DataFrame(pokemon.pop('price_data'), index=[0])
            #print(data)
            data2 = pd.DataFrame(pokemon, index=[0])
            data2 = data2.drop('symbol', axis=1)
            #print(data2)
            pokemon = pd.concat([data, data2], axis=1)
            #print(pokemon)
            pokemon.columns = [c.lower() for c in pokemon.columns]
        except:
            pokemon = None
        return pokemon

async def main_stocktwits(stocklist, filepath=None, tblname=None):

    async with aiohttp.ClientSession() as session:

        tasks = []
        for symbol in stocklist:#['APPS','MSFT','MGNI','GOOGL']:
            url = f'https://api.stocktwits.com/api/2/symbols/with_price/{symbol}.json?extended=true'
            #url = f'https://ql.stocktwits.com/pricedata?fundamentals=true&symbol={symbol}'
            tasks.append(asyncio.ensure_future(get_stocktwits(session, url)))

        df = await asyncio.gather(*tasks)
        df = pd.concat(df)
        df['loadtime'] = datetime.now()
        df.index = df.symbol
        df.index.name = 'sym'
        if tblname is not None:
            df.to_sql(tblname,con=engine, if_exists='append',chunksize=1000, method='multi', index=False)
            #upsert(con=engine, df=df, table_name=tblname, if_row_exists='update', chunksize=1000, create_table=True) 

        if filepath is not None:
            table = pa.Table.from_pandas(df)
            pqwriter = pq.ParquetWriter(filepath, table.schema)
            pqwriter.write_table(table)
            pqwriter.close()
        #print('Done')
        return df

async def get_stocktwits_price(session, url):
    async with session.get(url) as resp:
        pokemon = await resp.json()
        try:
            pokemon['symbol'].pop('price_data')
            pokemon['symbol'].pop('aliases') 
            pokemon = pd.DataFrame(pd.DataFrame(pokemon['symbol'], index=[0]), index=[0])
        except:
            pokemon = None
        return pokemon

async def main_stocktwits_price(stocklist, filepath=None, tblname=None):

    async with aiohttp.ClientSession() as session:

        tasks = []
        for symbol in stocklist:#['APPS','MSFT','MGNI','GOOGL']:
            url = f'https://api.stocktwits.com/api/2/symbols/with_price/{symbol}.json?extended=true'
            #url = f'https://ql.stocktwits.com/pricedata?fundamentals=true&symbol={symbol}'
            tasks.append(asyncio.ensure_future(get_stocktwits(session, url)))

        original_pokemon = await asyncio.gather(*tasks)
        df = pd.concat(original_pokemon)
        df['loadtime'] = datetime.now()
        df.index = df.symbol
        df.index.name = 'id'
        if tblname is not None:
            df.to_sql(tblname,con=engine, if_exists='append',chunksize=1000, method='multi', index=True)
            #upsert(con=engine, df=df, table_name=tblname, if_row_exists='update', chunksize=1000, create_table=True) 

        if filepath is not None:
            table = pa.Table.from_pandas(df)
            pqwriter = pq.ParquetWriter(filepath, table.schema)
            pqwriter.write_table(table)
            pqwriter.close()
        return df

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
        #print(len(stocklist))
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
        #print('Done')

        return df

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
            #print(d)
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
            #print(d)
        except:
            d = None
        print(d)
        return d

async def crawl_tipranks(stocklist, export_path='./tipranks_analysts.parquet'):
    async with aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}) as session:
        tasks = []
        #symbols = get_stocklist('sp500')
        print(stocklist)
        for symbol in stocklist:#:
            site = 'https://www.tipranks.com/stocks/'+symbol+'/forecast'
            tasks.append(asyncio.ensure_future(get_tipranks(session, site)))
        original_pokemon = await asyncio.gather(*tasks)
        print(original_pokemon)
        df = pd.concat(original_pokemon)
        print(df.shape)
        #df.to_parquet(export_path)
        #df.to_csv('nasdaq_yahoo_premarket2.csv')
        #df = pa.table(df)
        #pq.write_table(df, "./data/tipranks_test.parquet")
        #print('Done')
        return df

def build_yahoo_url(ticker, start_date = None, end_date = None, interval = "1d"):
    base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"

    if interval not in ("1d", "1wk", "1mo", "1m",'60m','1h','5m'):
        raise AssertionError("interval must be of of '1d', '1wk', '1mo', '1m', '5m")

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

            if interval not in ["60m","1h"]:
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
            
        await asyncio.sleep(1)

        return frame

async def main_yahoo(stocklist, export_path = None, INTERVAL_='1d', START_DATE_='2021-01-01',END_DATE_=None, tblname='yahoo_data', index_as_date=True, prep = False, filter_hour=True):

    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

    async with aiohttp.ClientSession() as session:
    
        tasks = []
        print('No of symbols: ' + str(len(stocklist)))

        for symbol in stocklist:#:
            site, params = build_yahoo_url(ticker=symbol, start_date=START_DATE_, end_date=END_DATE_, interval=INTERVAL_)
            site += '?'
            for k, v in params.items():
                site += k +'='+str(v) +'&'
            tasks.append(asyncio.ensure_future(get_yahoo(session, site, INTERVAL_, index_as_date)))

        original_pokemon = await asyncio.gather(*tasks)

        d = pd.concat(original_pokemon)
        start_time = time.time()
        print(d.columns)
        if ('h' in INTERVAL_) | ('m' in INTERVAL_):
            d = yahoo_prep(d, how='hourly')
            if filter_hour:
                d = d[(d.date.dt.second==0) & (d.date.dt.minute==30)]
        else:
            print('daily')
            d = yahoo_prep(d)
        
        print(d.shape)
        print(d.head())
        if ((prep is not None) | (prep != False)) & ('d' in INTERVAL_):
            if prep=='short':
                d = get_daily_fast(df = d, type_='short')
            if prep=='mid':
                d = get_daily_fast(df = d, type_='mid')
            if prep=='long':
                d = get_daily_fast(df = d, type_='long')
            print('data prepared')

        print(engine)
        print('-------------------')
        print(tblname)
        if export_path is not None:
            d.to_parquet(export_path)
        if ((tblname is not None) & (prep=='long')):
            start_time = time.time()
            print('upserting prepared data')
            #upsert(con=engine, df=d, table_name=tblname, if_row_exists='ignore', chunksize=1000, create_table=True) 
            try:
                upsert(con=engine, df=d[d.date>=str(END_DATE_ + timedelta(days=-10))], table_name=tblname, if_row_exists='update', chunksize=1000, create_table=True)
            except:
                print('failed') 
        elif ((tblname is None) | (tblname==''))==False:
            print('upserting')
            try:
                upsert(con=engine, df=d, table_name=tblname, if_row_exists='update', chunksize=1000, create_table=True) 
            except:
                print('failed 2')
        print(f'This took {time.time()-start_time}')
        return d
        """
        if len(sys.argv)>1:
            df = pd.concat([df,d])
            df.to_parquet(sys.argv[1])
            print(len(df.ticker.unique()))
        else:
            print(len(d.ticker.unique()))
            d.to_parquet(sys.argv[0])
        """  

from lxml import html
from utils.util import *
import pandas as pd 
from user_agent import generate_user_agent

async def get_stocktwits(session, url):
    async with session.get(url=url, headers=({'User-Agent':generate_user_agent()})) as resp:
        data = await resp.text()
        
        try:
            page_parsed = html.fromstring(data)
            news_table = page_parsed.cssselect('table[id="news-table"]')

            if len(news_table) == 0:
                return []

            rows = news_table[0].xpath("./tr[not(@id)]")

            results = []
            date = None
            for row in rows:
                raw_timestamp = row.xpath("./td")[0].xpath("text()")[0][0:-2]

                if len(raw_timestamp) > 8:
                    parsed_timestamp = datetime.strptime(raw_timestamp, "%b-%d-%y %I:%M%p")
                    date = parsed_timestamp.date()
                else:
                    parsed_timestamp = datetime.strptime(raw_timestamp, "%I:%M%p").replace(
                        year=date.year, month=date.month, day=date.day)

                results.append((
                    parsed_timestamp.strftime("%Y-%m-%d %H:%M"),
                    row.xpath("./td")[1].cssselect('a[class="tab-link-news"]')[0].xpath("text()")[0],
                    row.xpath("./td")[1].cssselect('a[class="tab-link-news"]')[0].get("href"),
                    row.xpath("./td")[1].cssselect('div[class="news-link-right"] span')[0].xpath("text()")[0][1:]
                ))
            results = pd.DataFrame(results)
            results.columns = ['time','title','link','source']
            symbol = url.split('=')
            symbol = symbol[1].split('&')[0]
            results.insert(1, 'symbol', symbol)
            results.insert(0, 'date',[r[:10] for r in results['time']])
            results.insert(0, 'id',results['date'].str.replace('-','') + results['symbol'])
            results['date'] = pd.to_datetime(results.date)
            results['time'] = pd.to_datetime(results.time)
        except:
            results = None
        return results

async def main_finviz_news(stocklist, filepath=None, tblname=None):
    connector = aiohttp.TCPConnector(limit=25)
    async with aiohttp.ClientSession(headers=({'User-Agent':generate_user_agent()}), connector=connector) as session:
        tasks = []
        for symbol in stocklist:#['APPS','MSFT','MGNI','GOOGL']:
            url = f'https://finviz.com/quote.ashx?t={symbol}&ty=c&p=d&b=1'
            #url = f'https://ql.stocktwits.com/pricedata?fundamentals=true&symbol={symbol}'
            tasks.append(asyncio.ensure_future(get_stocktwits(session, url)))

        df = await asyncio.gather(*tasks)
        df = pd.concat(df)
        return df

#df = asyncio.run(main_finviz_news(['AAPL','APPS']))
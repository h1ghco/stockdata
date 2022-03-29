#!/usr/bin/env python3
import asyncio
import logging
from contextlib import closing
import aiohttp # $ pip install aiohttp
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from debugpy import log_to
import pandas as pd
import posixpath
import time
from yahoo_fin import stock_info as si
from utils.util_clean import * 
try:
    from urlparse import urlsplit
    from urllib import unquote
except ImportError: # Python 3
    from urllib.parse import urlsplit, unquote


##############################
symbols = get_stocklist('sp500')
#symbols = pd.read_csv('stockrow_companies.csv')
#symbols = list(symbols['ticker'].unique())

#df = pd.read_csv('IWM.csv', skiprows=9, nrows=2033)
#symbols = df.Ticker.unique()
##############################


u = {
'SYMBOL_Income_A.xlsx':'https://stockrow.com/api/companies/' + 'SYMBOL' + '/financials.xlsx?dimension=A&section=Income%20Statement&sort=desc',
'SYMBOL_Income_Q.xlsx':'https://stockrow.com/api/companies/' + 'SYMBOL'  +'/financials.xlsx?dimension=Q&section=Income%20Statement&sort=desc',
'SYMBOL_Income_T.xlsx':'https://stockrow.com/api/companies/' + 'SYMBOL'  + '/financials.xlsx?dimension=T&section=Income%20Statement&sort=desc',
'SYMBOL_Balance_A.xlsx':'https://stockrow.com/api/companies/' + 'SYMBOL'  + '/financials.xlsx?dimension=A&section=Balance%20Sheet&sort=desc',
'SYMBOL_Balance_Q.xlsx':'https://stockrow.com/api/companies/' + 'SYMBOL'  + '/financials.xlsx?dimension=Q&section=Balance%20Sheet&sort=desc',
'SYMBOL_Cashflow_Q.xlsx':'https://stockrow.com/api/companies/' + 'SYMBOL'  + '/financials.xlsx?dimension=Q&section=Cash%20Flow&sort=desc',
'SYMBOL_Cashflow_A.xlsx':'https://stockrow.com/api/companies/' + 'SYMBOL'  + '/financials.xlsx?dimension=A&section=Cash%20Flow&sort=desc',
'SYMBOL_Cashflow_T.xlsx':'https://stockrow.com/api/companies/' + 'SYMBOL'  + '/financials.xlsx?dimension=T&section=Cash%20Flow&sort=desc',
'SYMBOL_Metric_Q.xlsx':'https://stockrow.com/api/companies/' + 'SYMBOL'  + '/financials.xlsx?dimension=Q&section=Metrics&sort=desc',
'SYMBOL_Metric_A.xlsx':'https://stockrow.com/api/companies/' + 'SYMBOL'  + '/financials.xlsx?dimension=A&section=Metrics&sort=desc',
'SYMBOL_Metric_T.xlsx':'https://stockrow.com/api/companies/' + 'SYMBOL'  + '/financials.xlsx?dimension=T&section=Metrics&sort=desc',
'SYMBOL_Growth_Q.xlsx': 'https://stockrow.com/api/companies/' + 'SYMBOL'  + '/financials.xlsx?dimension=Q&section=Growth&sort=desc',
'SYMBOL_Growth_A.xlsx':'https://stockrow.com/api/companies/' + 'SYMBOL'  + '/financials.xlsx?dimension=A&section=Growth&sort=desc',
}

urls ={}
for s in symbols:
   for k, v in u.items():
        urls[k.replace('SYMBOL',s)] = v.replace('SYMBOL',s)


def url2filename(url):
    """Return basename corresponding to url.
    >>> print(url2filename('http://example.com/path/to/file%C3%80?opt=1'))
    fileÀ
    >>> print(url2filename('http://example.com/slash%2fname')) # '/' in name
    Traceback (most recent call last):
    ...
    ValueError
    """
    urlpath = urlsplit(url).path
    basename = posixpath.basename(unquote(urlpath))
    if (os.path.basename(basename) != basename or
        unquote(posixpath.basename(urlpath)) != basename):
        raise ValueError  # reject '%2f' or 'dir%5Cbasename.ext' on Windows
    return basename

@asyncio.coroutine
def download(url, name, session, semaphore, chunk_size=1<<15):
    with (yield from semaphore): # limit number of concurrent downloads
        if os.path.exists(os.path.join('stockrow',name))==False:
            filename = url2filename(url)
            logging.info('downloading %s', filename)
            response = yield from session.get(url)
            with closing(response), open(os.path.join('stockrow',name), 'wb') as file:
                while True: # save file
                    chunk = yield from response.content.read(chunk_size)
                    if not chunk:
                        break
                    file.write(chunk)
            logging.info('done %s', filename)
            time.sleep(1)
            return filename, (response.status, tuple(response.headers.items()))
        else:
            return None

urls.values()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
with closing(asyncio.get_event_loop()) as loop, \
     closing(aiohttp.ClientSession()) as session:
    semaphore = asyncio.Semaphore(4)
    download_tasks = (download(url, name, session, semaphore) for  name, url in urls.items())
    result = loop.run_until_complete(asyncio.gather(*download_tasks))


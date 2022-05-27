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
from asyncio_fetch_data import *

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

start_time = time.time()
err = []
tdy = str(datetime.datetime.now().date())


df = asyncio.run(crawl_tipranks(get_stocklist('sp500'), export_path='./tipranks_analysts_all2.parquet'))
print("--- %s seconds ---" % (time.time() - start_time))
#df = pd.read_parquet("./data/tipranks_test.parquet")

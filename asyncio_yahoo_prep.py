
from utils.util import *
from utils.util_clean import *
import sys
import time

start_time = time.time()
print(sys.argv)
#df = pd.read_parquet('./data/yahoo_nasdaq_2020_2022_full_prep.parquet')
#df = pd.read_parquet('yahoo_nasdaq_2020_today_full_1d.parquet')#'./data/yahoo_nasdaq_2020_2022_full_prep.parquet')#sys.argv[0])
df = pd.read_parquet('./data/yahoo_test_2021.parquet')
#stocklist=si.tickers_sp500()
#df = df[df.symbol.isin(stocklist)]

#df = df[df.symbol.isin(get_stocklist('sp500'))]
df = yahoo_prep_async(df)
#df = df[df.symbol.isin(['APPS','MGNI'])]
df = get_daily(['APPS'], df)

df.to_parquet('./data/yahoo_test_2021_prep.parquet')
print("--- %s seconds ---" % (time.time() - start_time))

#df.to_parquet(sys.argv[1])
#d, d_last = daily_scans(df,['rsnhbp_5d'],load_premarket=False)
"""
#er = pd.DataFrame(si.get_earnings_history('APPS'))
earnings = si.get_earnings_in_date_range('2021-02-19','2021-02-28')
earnings = pd.DataFrame(earnings)

earningsdf = si.get_earnings('APPS')
get_active = si.get_day_most_active()
get_gainers = si.get_day_gainers()
live_price = si.get_live_price('APPS')
si.get_stats('APPS')
start_time = time.time()
"""
"""
x = pd.read_parquet("./data/yahoo_premarket.parquet")
x[x.marketCap > 100000000].shape
x[x.averageDailyVolume3Month > 2000000].shape
x[(x.regularMarketPrice * x.averageDailyVolume3Month) > 5000000].shape
x[['symbol','fiftyDayAverageChange','postMarketChangePercent']]

volmask = (x.regularMarketPrice * x.averageDailyVolume3Month) > 5000000

",".join(x.loc[(x.marketCap > 100000000) & volmask, ['symbol','fiftyDayAverageChange','postMarketChangePercent','marketCap']].sort_values('postMarketChangePercent', ascending=False).head(20).symbol.unique())
"""
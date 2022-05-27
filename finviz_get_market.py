import time
import dbconfig as db
from utils.util_clean import *
from sqlalchemy import create_engine
from finviz.screener import Screener

start_time = time.time()

engine = create_engine(f'postgresql+psycopg2://{db.user}:{db.password}@{db.raspberry}')

data, market = finviz_get_market()
print('finviz data loaded')
data.to_sql('finviz_data', con=engine, if_exists='append',chunksize=1000, method='multi', index=True)
market.to_sql('finviz_market', con=engine, if_exists='append',chunksize=1000, method='multi', index=True)
print('finviz data written to db')

#data.to_parquet('finviz_market_data.parquet')
#market.to_parquet('finviz_market.parquet')

print("--- %s seconds ---" % (time.time() - start_time))

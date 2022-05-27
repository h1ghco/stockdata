
import glob
import os
import pandas as pd
from sqlalchemy import create_engine
from pangres import upsert
engine = create_engine('postgresql+psycopg2://tradekit:yourpassword@192.168.0.174')

def prep_stockrow(df):
    df.columns = [c.replace('(','') for c in df.columns]
    df.columns = [c.replace(')','') for c in df.columns]
    df.columns = [c.replace(' ','_') for c in df.columns]
    df.columns = [c.replace('/','_') for c in df.columns]
    df.columns = [c.lower() for c in df.columns]
    df.index.name = 'id'
    return df

"""
os.chdir('stockrow')
growth = glob.glob('*Growth_*')
growth_a = glob.glob('*Growth_A*')
growth_q = glob.glob('*Growth_Q*')

metrics_a = glob.glob('*Metric_A*')
metrics_q = glob.glob('*Metric_Q*')
metrics_t = glob.glob('*Metric_T*')

cashflow_a = glob.glob('*Cashflow_A*')
cashflow_q = glob.glob('*Cashflow_Q*')
cashflow_t = glob.glob('*Cashflow_T*')

balance_a = glob.glob('*Balance_A*')
balance_q = glob.glob('*Balance_Q*')

income_a = glob.glob('*Income_A*')
income_q = glob.glob('*Income_Q*')
income_t = glob.glob('*Income_T*')
l = list()
err = list()
for i in income_t:
    s = i.split('_')[0]
    try:
        print(i)
        d = pd.read_excel(i,engine='openpyxl')
        d.index=d['Unnamed: 0']
        d = d.drop(columns=['Unnamed: 0'])
        d = d.T
        d.insert(0,'symbol',s)
        d.insert(1, 'date', pd.to_datetime(d.index))
        d.index = d['symbol'] + d['date'].astype('str').str.replace('-','')
        d.index.name = 'date'
        l.append(d)
    except:
        err.append(i)
os.chdir('..')
gr = pd.concat(l)
gr.to_parquet('stockrow_income_t.parquet')
"""
import shutil
cashflow_a = pd.read_parquet('./data/stockrow_cashflow_a.parquet')
cashflow_q = pd.read_parquet('./data/stockrow_cashflow_q.parquet')
cashflow_t = pd.read_parquet('./data/stockrow_cashflow_t.parquet')

income_a = pd.read_parquet('./data/stockrow_income_a.parquet')
income_q = pd.read_parquet('./data/stockrow_income_q.parquet')

balance_a = pd.read_parquet('./data/stockrow_balance_a.parquet')
balance_q = pd.read_parquet('./data/stockrow_balance_q.parquet')

metrics_a = pd.read_parquet('./data/stockrow_metrics_a.parquet')
metrics_q = pd.read_parquet('./data/stockrow_metrics_q.parquet')
metrics_t = pd.read_parquet('./data/stockrow_metrics_t.parquet')
metrics_t = prep_stockrow(metrics_t)
upsert(con=engine, df=metrics_t, table_name='stockrow_metrics_t', if_row_exists='update', chunksize=1000, create_table=True)  # default

growth_a = pd.read_parquet('./data/stockrow_growth_a.parquet')

growth_a = prep_stockrow(growth_a)
upsert(con=engine, df=growth_a, table_name='stockrow_growth_a', if_row_exists='update', chunksize=1000, create_table=True)  # default
growth_q = pd.read_parquet('stockrow_growth_q.parquet')

'./data/stockrow_cashflow_a.parquet','./data/stockrow_cashflow_q.parquet','./data/stockrow_cashflow_t.parquet',
        './data/stockrow_income_a.parquet','./data/stockrow_income_q.parquet',
files = [        './data/stockrow_balance_a.parquet','./data/stockrow_balance_q.parquet',
        './data/stockrow_metrics_a.parquet','./data/stockrow_metrics_q.parquet','./data/stockrow_metrics_t.parquet'
        ]
for file in files:
    df = pd.read_parquet(file)
    df = prep_stockrow(df)
    filename = os.path.split(file)[1].replace('.parquet','')
    upsert(con=engine, df=df, table_name=filename, if_row_exists='update', chunksize=1000, create_table=True)  # default
    print(f'{filename} done')
#l_ = growth + metrics + cashflow + balance + income
#l_ = [s.split('_')[0] for s in l_]
#l_ = set(l_)
#len(l_)

import pandas as pd

cashflow = pd.read_parquet('stockrow_cashflow.parquet')
balance = pd.read_parquet('stockrow_balance.parquet')
income = pd.read_parquet('stockrow_income.parquet')
metrics = pd.read_parquet('stockrow_metrics.parquet')
growth = pd.read_parquet('stockrow_growth.parquet')
cashflow = pd.read_parquet('stockrow_cashflow.parquet')
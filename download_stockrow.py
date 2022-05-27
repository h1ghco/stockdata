from utils.util_clean import *
import time
symbols = get_stocklist('sp500')[:50]
#symbols=['AAPL']

start_time = time.time()
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


from parfive import Downloader
dl = Downloader(max_conn=10)
for n,u in urls.items():
    dl.enqueue_file(url=u, path="./stockrowdata", filename=n)
files = dl.download()
print("--- %s seconds ---" % (time.time() - start_time))

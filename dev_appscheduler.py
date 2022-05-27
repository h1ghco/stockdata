
from datetime import datetime
import time
import os
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from runpy import run_path
import sys


def tick():
    print('Tick! The time is: %s' % datetime.now())


def yahoo_hourly():
    #time.sleep(20)
    print(datetime.now())
    sys.argv = ['', '1', '1h', 'all','-t yahoo_hourly_test']
    run_path('asyncio_yahoo.py', run_name='__main__')

def yahoo_daily():
    #time.sleep(20)
    print(datetime.now())
    sys.argv = ['', '2', '1d', 'sp500','-t yahoo_daily_test']
    run_path('asyncio_yahoo.py', run_name='__main__')

def finviz_data():
    #time.sleep(20)
    print(datetime.now())
    #sys.argv = ['', '5', '1d', 'all','-t yahoo_hourly']
    run_path('finviz_get_market.py', run_name='__main__')


def job_function2():
    print(datetime.now())

def main():
    sched = BackgroundScheduler(timezone="Europe/Berlin")
    # Schedule job_function to be called every two hours
    #sched.add_job(job_function, 'cron', day_of_week='mon-fri', hour='13', minute=37, second=0)
    #sched.add_job(job_function2, 'cron', day_of_week='mon-fri', hour='13', minute=37, second=0)
    #sched.add_job(job_function, 'interval', seconds=30)
    # sched.add_job(yahoo_daily, 'cron', day_of_week='mon-fri', hour='15', minute=30, second=10)
    # sched.add_job(yahoo_daily, 'cron', day_of_week='mon-fri', hour='16', minute=30, second=10)
    # sched.add_job(yahoo_daily, 'cron', day_of_week='mon-fri', hour='17', minute=30, second=10)
    # sched.add_job(yahoo_daily, 'cron', day_of_week='mon-fri', hour='18', minute=30, second=10)
    # sched.add_job(yahoo_daily, 'cron', day_of_week='mon-fri', hour='19', minute=30, second=10)
    # sched.add_job(yahoo_daily, 'cron', day_of_week='mon-fri', hour='20', minute=30, second=10)
    # sched.add_job(yahoo_daily, 'cron', day_of_week='mon-fri', hour='21', minute=30, second=10)
    sched.add_job(yahoo_daily, 'cron', day_of_week='mon-sat', hour='15', minute=3, second=10)
    #sched.add_job(finviz_data, 'cron', day_of_week='mon-sat', hour='14', minute=45, second=10)

    sched.start()

    while True:
        time.sleep(5)

if __name__ == '__main__':
    main()


    """
    #sched.add_job(job_function, 'interval', minutes=60, start_date= str(datetime.now().date()) + ' 12:53:30', end_date=str(datetime.now().date()) + ' 21:30:30')
    #sched.add_job(job_function, 'interval', minutes=1, start_date= str(datetime.now().date()) + ' 12:53:30', end_date=str(datetime.now().date()) + ' 21:30:30')


    

    scheduler = BackgroundScheduler()
    scheduler.add_job(tick, 'interval', seconds=10)
    scheduler.start()
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))

    try:
        # This is here to simulate application activity (which keeps the main thread alive).
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        # Not strictly necessary if daemonic mode is enabled but should be done if possible
        scheduler.shutdown()
    """
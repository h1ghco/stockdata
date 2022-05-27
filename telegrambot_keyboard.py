#!/usr/bin/env python
# pylint: disable=C0116,W0613
# This program is dedicated to the public domain under the CC0 license.

"""
Basic example for a bot that uses inline keyboards. For an in-depth explanation, check out
 https://git.io/JOmFw.
"""
import logging

from telegram import * #InlineKeyboardButton, InlineKeyboardMarkup, Update, ReplyKeyboardMarkup,KeyboardButton, MessageHandler
from telegram.ext import *#Updater, CommandHandler, CallbackQueryHandler, CallbackContext

import aiohttp
import asyncio
import time
import pandas as pd
import requests
from yahoo_fin import stock_info as si
import time
from datetime import datetime
from datetime import timedelta

#import pyarrow as pa
#import pyarrow.parquet as pq
import yfinance
from utils.util_clean import * 
from asyncio_tipranks_preaftermarket import *
from yahoo_fin.stock_info import get_data, tickers_sp500, tickers_nasdaq, tickers_other, get_quote_table
import yahoo_fin.stock_info as si
from asyncio_fetch_data import *
from sqlalchemy import create_engine
import dbconfig as db
#import datetime, pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler

global df
global earnings_dates



df = None
start_time = time.time()
today = str(datetime.now().date())
today = today.replace('-','_')
pqwriter = None
engine = create_engine(f'postgresql+psycopg2://{db.user}:{db.password}@{db.raspberry}')

#allowedUsers = ['h1ghco']

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)


def start(update: Update, context: CallbackContext) -> None:
    """Sends a message with three inline buttons attached."""

    #if update.effective_chat.username not in allowedUsers:
    #    context.bot.send_message(chat_id=update.effective_chat.id, text='you are not allowed here')
    #    return

    keyboard = [
        [
            KeyboardButton("Load Stocktwits"),
            KeyboardButton("Get Stocktwits Trend"),
        ],
        [KeyboardButton("Get Stocktwits Sentiment")],
    ]

    reply_markup = ReplyKeyboardMarkup(keyboard)
    user_id = update.message.from_user.id
    user_name = update.message.from_user.name

    user = pd.DataFrame({'user_id':user_id,
                        'user_name':user_name}, index=[0])
    user.to_sql('telegram_user',con=engine, if_exists='ignore', index=False)
    print(user)
    tdy = datetime.now()
    context.job_queue.run_repeating(hourly_update, 60, context=update.message.chat_id,
            first=tdy.replace(hour=15, minute=31), 
            last= tdy.replace(hour=22, minute=35))
    
    update.message.reply_text('Use the keyboard or \n' +
                              'Use /premarket to get pre and aftermarket changes', reply_markup=reply_markup)


def button(update: Update, context: CallbackContext) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    query.answer()

    query.edit_message_text(text=f"Selected option: {query.data}")

def message_handler(update: Update, context: CallbackContext):
    global df
    if ('Load Stocktwits' in update.message.text):
        df = asyncio.run(main_stocktwits(get_stocklist('sp500'),tblname='stocktwits_hourly')) #
    elif 'Get Stocktwits Trend' in update.message.text:
        if df is None:
            print('load date')
            df = pd.read_sql_query("""
            SELECT *
            FROM (SELECT *, 
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY loadtime DESC) seq 
                FROM stocktwits_hourly) t 
            WHERE (seq = 1) 
            ORDER BY symbol;
            """, con=engine)
        trend = df[df.trending_score>3].sort_values('trending_score')
        if trend.shape[0]>0:
            resp = ''
            for i, r in trend.iterrows():
                resp += r['symbol'] + ' ' + str(round(r['trending_score'])) + ' ' + str(round(r['percentchange'])) + ' '+ str(round(r['extendedhourspercentchange'])) + '\n'
            resp += 'https://finviz.com/screener.ashx?v=351&t=' + ",".join(trend.head().symbol.unique()) + '&o=-change'
            print(resp)
            context.bot.send_message(chat_id= update.effective_chat.id, text=resp)

    elif ('Get Stocktwits Sentiment' in update.message.text):
        if df is None:
            df = pd.read_sql_query("""
            SELECT *
            FROM (SELECT *, 
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY loadtime DESC) seq 
                FROM stocktwits_hourly) t 
            WHERE (seq = 1) 
            ORDER BY symbol;
            """, con=engine)        
        sentiment = df[df.sentiment_change>10].sort_values('sentiment_change')
        resp = ''
        if sentiment.shape[0]>0:
            for i, r in sentiment.iterrows():
                resp += r['symbol'] + ' ' + str(round(r['sentiment_change'])) + ' ' + str(round(r['percentchange'])) + ' '+ str(round(r['extendedhourspercentchange'])) + '\n'
            resp += 'https://finviz.com/screener.ashx?v=351&t=' + ",".join(sentiment.head().symbol.unique()) + '&o=-change'
            print(resp)
            context.bot.send_message(chat_id= update.effective_chat.id, text=resp)

def symbol_price(update: Update, context: CallbackContext) -> None: 
        message = update.message.text.replace("/price ", "")
        chat_id = update.message.chat_id
        print(message)
        if message.strip().split("@")[0] == "/price":
            update.message.reply_text(
                "This command searches for symbols supported by the bot.\nExample: /search Tesla Motors or /search $tsla"
            )
            return

        context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        price = yf.Ticker(message)
        price = price.history(period='2d', interval='1d')
        price.columns = [c.lower() for c in price.columns]
        price['change_perc'] = price.close.pct_change() * 100
        reply = round(price['change_perc'].tail(1).values[0], 2)
        reply = message + ' ' + str(reply) + ' %'
        update.message.reply_text(
                text=reply,
                disable_notification=True
        )

def earnings(update: Update, context: CallbackContext) -> None: 
        #message = update.message.text.replace("/earnings ", "")
        #chat_id = update.message.chat_id
        today = datetime.today().date()
        earningsdate = today+timedelta(days=-1)
        global earnings_dates
        earnings_dates = pd.DataFrame(si.get_earnings_for_date(str(earningsdate))) 
        """
        if message.strip().split("@")[0] == "/earnings":
            update.message.reply_text(
                "This command searches for symbols supported by the bot.\nExample: /search Tesla Motors or /search $tsla"
            )
            return
        """
        #nyc_datetime = datetime.datetime.now(pytz.timezone('US/Eastern'))
        #nyc_datetime.time()
        df = pd.read_sql_query('SELECT * FROM finviz_data WHERE date = (SELECT MAX(date) FROM finviz_data)', con = engine)
        df['earnings'] = df['earnings'].str.replace('-','')
        df = df[df['earnings']!='']
        df['earnings'] = pd.to_datetime(df['earnings'], format='%b %d', errors='ignore')
        df['earnings'] = df.earnings + pd.offsets.DateOffset(year=2022)
        df = df[df['market cap']/(10**6) > 300]
        df = df[df['avg volume']/(10**6) > 0.3]
    
        earnings_ytdy = df[(df['earnings'].dt.date > datetime.today().date() + timedelta(days=-1)) & (df.earnings_time == 'a')]
        earnings_tdy = df[(df['earnings'].dt.date > datetime.today().date()) & (df.earnings_time == 'b')] 

        pre = asyncio.run(main_tip_pre(list(earnings_tdy.symbol.unique())+list(earnings_ytdy.symbol.unique())))
        earnings_ytdy = pd.merge(pre, earnings_ytdy, left_on='ticker', right_on='symbol', how='inner')
        earnings_ytdy = earnings_ytdy[earnings_ytdy.preMarket_changePercent>5]

        earnings_tdy = pd.merge(pre, earnings_tdy, left_on='ticker', right_on='symbol', how='inner', suffixes=['_finviz','_pre'])
        earnings_tdy = earnings_tdy[earnings_tdy.preMarket_changePercent>5]
        
        
        fmsg = ''
        if earnings_ytdy.shape[0]>0:
            for i, r in earnings_ytdy.iterrows():
                fmsg += f"<a href='https://stocktwits.com/symbol/{r['ticker']}'> {r['ticker']}</a> pre: {r['preMarket_changePercent']}% + Vol: {r['preMarket_volume']} Marketcap: {r['marketCap']/10**6} \n"
            fmsg += '\n'
        if earnings_tdy.shape[0]>0:
            for i, r in earnings_tdy.iterrows():
                fmsg += f"<a href='https://stocktwits.com/symbol/{r['ticker']}'> {r['ticker']}</a> after: {r['afterHours_changePercent']}% +  Vol: {r['afterHours_volume']} Marketcap: {r['marketCap']/10**6} \n"
            
        tickers = list(earnings_ytdy.symbol.unique()) + list(earnings_tdy.symbol.unique())
        fmsg += '<a href="https://finviz.com/screener.ashx?v=351&t=' + ",".join(tickers) + '&o=-change">Finviz</a>'
        
        if earnings_dates.shape[0]>0:
            df2 = asyncio.run(main_tip_pre(earnings_dates.ticker.unique()))
            #df2 = asyncio.run(main_stocktwits(earnings_dates.ticker.unique(),export_path='./data/stocktwits_20220325.parquet'))
            l = []
            # for symbol in df2.ticker.unique():
            #     ticker = yf.Ticker(symbol)
            #     ticker = ticker.calendar
            #     l.append()
            #df2['afterHours_price'] > 2
            df2['preMarket_changePercent'] = df2['preMarket_changePercent'].round(2)
            df2['afterHours_changePercent'] = df2['afterHours_changePercent'].round(2)
            df2 = pd.merge(df2, earnings_dates, left_on='ticker', right_on='ticker', how='inner')

            df2 = df2[(df2.marketCap/10**6)>500]
            preGainer = df2[df2['preMarket_changePercent'] > 5]
            afterGainer = df2[df2['afterHours_changePercent'] > 5]
            msg = ''
            for i, r in preGainer.iterrows():
                msg += f"<a href='https://stocktwits.com/symbol/{r['ticker']}'> {r['ticker']}</a> pre: {r['preMarket_changePercent']}% + Vol: {r['preMarket_volume']} EPSSuprise: {r['epssurprisepct']} Marketcap: {r['marketCap']/10**6} \n"
            msg += '\n'
            for i, r in afterGainer.iterrows():
                msg += f"<a href='https://stocktwits.com/symbol/{r['ticker']}'> {r['ticker']}</a> after: {r['afterHours_changePercent']}% +  Vol: {r['afterHours_volume']}  EPSSuprise: {r['epssurprisepct']} Marketcap: {r['marketCap']/10**6} \n"
            
            tickers = list(preGainer.ticker.unique()) + list(afterGainer.ticker.unique())
            #context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            msg += '<a href="https://finviz.com/screener.ashx?v=351&t=' + ",".join(tickers) + '&o=-change">Finviz</a>'
        else:
            msg = 'no earnings today'
        update.message.reply_text(
                text=msg + '\n' + '\n' + fmsg,
                disable_notification=True,
                parse_mode=ParseMode.HTML
        )


def help_command(update: Update, context: CallbackContext) -> None:
    """Displays info on how to use the bot."""
    update.message.reply_text("Use /start to test this bot. \n" +
                              "Use /premarket to get pre and aftermarket changes")

def watchlist_price(update: Update, context: CallbackContext) -> None:
    df = asyncio.run(main_stocktwits(get_stocklist('tradingview_1'), tblname='stocktwits_hourly'))
    df = df[df.percentchange>2]
    if df.shape[0]>0:
        for i, r in df.iterrows():
            msg =  + f"{r['symbol']} change: {str(round(r['percentchange']))} % volume: {str(r['volume'])}"
        update.message.reply_text(msg)

def premarket_watchlist(update: Update, context: CallbackContext) -> None:
    df = asyncio.run(main_tip_pre(get_stocklist('tradingview_1')))
    for i, r in df.iterrows():
        if r['afterHours_changePercent']>2:#r['isAfterMarketTime']==True:
            msg = r['ticker'] + ' after hours change %:' + str(round(r['afterHours_changePercent'])) + '% volume: ' + str(r['afterHours_volume'])
        elif r['isPreMarketTime']==True:
            msg = r['ticker'] + ' pre market change:' + str(round(r['afterHours_changePercent'])) + '% volume: ' + str(r['preMarket_volume'])
        elif r['isMarketOpen']==True:
            msg = r['ticker'] + ' change :' + str(round(r['changePercent'])) + '% volume: ' + str(r['volume'])
    update.message.reply_text(msg)

def hourly_update(context) -> None:
    #df = asyncio.run(main_tip_pre(get_stocklist('tradingview_1')))
    #user = pd.read_sql('telegram_user', con=engine)
    #userid = user.user_id.values[0]
    df = pd.read_sql_query('SELECT * FROM yahoo_latest_v',con=engine)
    df = df.sort_values(['symbol','day'])
    df['change'] = df.groupby('symbol')['close'].pct_change()*100
    df = df.groupby('symbol').last()
    df = df[df.change > 2]
    msg = ''
    if df.shape[0] >0:
        for i, r in df.iterrows():
            msg += f"{r['symbol']} {r['change']}"
    context.bot.send_message(context.job.context, text=msg)
    #bot.send_message(chat_id=context.chat_id ,text='Hello World')
    #userid='hello'
    #print(userid)
    #msg = str(userid)
    #context.bot.send_message(chat_id=userid, text=msg)


def main() -> None:
    """Run the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater("1873071464:AAGNYevaE68Zsyclzb-FWOpI6IxWbWeWISI", use_context=True)

    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('prewatch', premarket_watchlist))
    updater.dispatcher.add_handler(CommandHandler('price', symbol_price))
    updater.dispatcher.add_handler(CommandHandler('earnings', earnings))
    updater.dispatcher.add_handler(CommandHandler('help', help_command))
    updater.dispatcher.add_handler(CallbackQueryHandler(button))
    updater.dispatcher.add_handler(MessageHandler(Filters.text, message_handler))

    # Start the Bot
    updater.start_polling()

    # Run the bot until the user presses Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT
    

    updater.idle()



if __name__ == '__main__':
    main()

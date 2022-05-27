import pandas as pd
import dbconfig as db
from sqlalchemy import create_engine
from datetime import timedelta 
from datetime import datetime
engine = create_engine(f'postgresql+psycopg2://{db.user}:{db.password}@{db.raspberry}')


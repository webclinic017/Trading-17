import importlib
import pandas as pd
from backtesting import Backtest


DATA= 'Backdata\gemini_BTCUSD_2021_1min.csv'
data= pd.read_csv(DATA, index_col=1, parse_dates=True)
data= data.head(10000)
data= data.drop(['Unix Timestamp', 'Symbol'], axis=1)
data['Open']= data['Open'].div(1_000_000)
data['Close']= data['Close'].div(1_000_000)
data['High']= data['High'].div(1_000_000)
data['Low']= data['Low'].div(1_000_000)

STRATERGY= 'teststratergy'
strategy = importlib.import_module(STRATERGY)

bt = Backtest(data, strategy.SmaCross, cash=10_000_000, commission=.00)
stats = bt.run()
print(stats)
bt.plot()
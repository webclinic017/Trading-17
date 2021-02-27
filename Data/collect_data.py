
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
import time
ts = TimeSeries(key='UXGHT48VBN37M55C', output_format='pandas', indexing_type='date')

Symbols = ['VOO', 'IBM', 'BA', 'BABA', 'TSLA', 'AAPL', 'JPM', 'GS', 'O', 'AMD', 'NVDA', 'GOOGL', 'FB', 'ROKU', 'NKLA', 'MCD']

for sym in Symbols:
    # Get json object with the intraday data and another with  the call's metadata
    data, meta_data = ts.get_intraday(symbol=sym,interval='5min', outputsize='full')
    data.columns = ['open', 'high', 'low', 'close', 'volume']

    # data.to_csv('{}_5Min'.format(sym))

    temp = pd.read_csv('{}_5Min'.format(sym), index_col=0, parse_dates=True)

    df = pd.concat([data, temp], axis=1)
    df.columns = ['open_x', 'high_x', 'low_x', 'close_x', 'volume_x', 'open_y', 'high_y', 'low_y', 'close_y', 'volume_y']

    df['open'] = np.where(df['open_y'].isna(), df['open_x'], df['open_y'])
    df = df.drop(columns=['open_x', 'open_y'])

    df['high'] = np.where(df['high_y'].isna(), df['high_x'], df['high_y'])
    df = df.drop(columns=['high_x', 'high_y'])

    df['low'] = np.where(df['low_y'].isna(), df['low_x'], df['low_y'])
    df = df.drop(columns=['low_x', 'low_y'])

    df['close'] = np.where(df['close_y'].isna(), df['close_x'], df['close_y'])
    df = df.drop(columns=['close_x', 'close_y'])

    df['volume'] = np.where(df['volume_y'].isna(), df['volume_x'], df['volume_y'])
    df = df.drop(columns=['volume_x', 'volume_y'])

    df.to_csv('{}_5Min'.format(sym))

    print('Finished {}'.format(sym))
    print('Waiting....')
    time.sleep(15)
    
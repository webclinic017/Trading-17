from backtesting.test import GOOG
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from backtesting import Strategy
from backtesting import Backtest

def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()

class AIBacktest(Strategy):

    n1 = 10

    check = 0

    def init(self):

        # For centerisation
        self.sma_open = self.I(SMA, self.data.Open, self.n1)
        self.sma_close = self.I(SMA, self.data.Close, self.n1)
        self.sma_low = self.I(SMA, self.data.Low, self.n1)
        self.sma_high = self.I(SMA, self.data.High, self.n1)
        self.sma_volumne = self.I(SMA, self.data.Volume, self.n1)

        # AI
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(26,)))
        self.model.add(tf.keras.layers.Dense(5, activation="relu"))
        self.model.add(tf.keras.layers.Dense(1))


        # Compile the model
        self.model.compile(
                        optimizer='rmsprop',
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                        )

    def next(self):

        self.check += 1

        if self.check > 1:
            self.check = 0

            input_data = np.array((self.data.Open[-5:], self.data.Close[-5:], self.data.High[-5:], self.data.Low[-5:], self.data.Volume[-5:]))
            central_data = input_data - np.array((self.sma_open[-5:], self.sma_close[-5:], self.sma_high[-5:], self.sma_low[-5:], self.sma_volumne[-5:]))
            central_data = np.append(central_data.reshape(1, -1), self.position.pl_pct)

            out = self.model.predict(central_data.reshape(1, -1))

            if out > 100:
                self.buy()
            elif out < 100:
                self.position.close()
            else:
                # Do nothing
                pass


def fitness(stats):
    pass
        


def main_AI(AI_type_file, Data_file):
    
    data = pd.read_csv(Data_file)

    bt = Backtest(GOOG, AIBacktest, cash=10_000, commission=.00)
    stats = bt.run()

    print(stats)
    print(stats['Sharpe Ratio'], stats['Return [%]'], stats['Win Rate [%]'])

    bt.plot()


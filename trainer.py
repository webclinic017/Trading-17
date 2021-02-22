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
    pos_open = False
    trailing_value = 0
    iteration = 0
    max_loss = 0.04
    check_every_n_iteration = 4

    current_trade_start = None

    def init(self):

        # For centerisation
        self.sma_open = self.I(SMA, self.data.Open, self.n1)
        self.sma_close = self.I(SMA, self.data.Close, self.n1)
        self.sma_low = self.I(SMA, self.data.Low, self.n1)
        self.sma_high = self.I(SMA, self.data.High, self.n1)
        self.sma_volumne = self.I(SMA, self.data.Volume, self.n1)

        # AI
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.Input(shape=(25,)))
        self.model.add(tf.keras.layers.Dense(5, activation="relu"))
        self.model.add(tf.keras.layers.Dense(1))


        # Compile the model
        self.model.compile(
                        optimizer='rmsprop',
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                        )

    def next(self):

        if not(self.pos_open):

            if self.iteration > self.check_every_n_iteration:
                self.iteration = 0

                input_data = np.array((self.data.Open[-5:], self.data.Close[-5:], self.data.High[-5:], self.data.Low[-5:], self.data.Volume[-5:]))
                central_data = input_data - np.array((self.sma_open[-5:], self.sma_close[-5:], self.sma_high[-5:], self.sma_low[-5:], self.sma_volumne[-5:]))

                out = self.model.predict(central_data.reshape(1, -1))

                if out > 100:
                    file = open('AI_Data/5Back_Fitness.csv', 'a')
                    file.write(','.join(['%.5f' % num for num in central_data.reshape(1, -1)[0]]))
                    file.close()
                    self.buy()
                    self.pos_open = True

            else:
                self.iteration += 1

        else:

            if self.data.Close[-1] > self.trailing_value:
                self.trailing_value = self.data.Close * (1-self.max_loss)

            elif self.data.Close[-1] < self.trailing_value:
                file = open('AI_Data/5Back_Fitness.csv', 'a')
                file.write('Comp\n')
                file.close()
                self.position.close()
                self.pos_open = False




def fitness(stats):
    pass
        


def main_AI(AI_type_file, Data_file):
    
    data = pd.read_csv(Data_file)

    bt = Backtest(GOOG, AIBacktest, cash=10_000, commission=.02)
    stats = bt.run()

    print(stats)
    print(stats['Sharpe Ratio'], stats['Return [%]'], stats['Win Rate [%]'])

    bt.plot()


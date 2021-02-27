from backtesting.test import GOOG
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from backtesting import Strategy
from backtesting import Backtest
import pygad.kerasga


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
    max_loss = 0.05
    check_every_n_iteration = 10
    model = '2'

    def init(self):

        # For centerisation
        self.sma_open = self.I(SMA, self.data.Open, self.n1)
        self.sma_close = self.I(SMA, self.data.Close, self.n1)
        self.sma_low = self.I(SMA, self.data.Low, self.n1)
        self.sma_high = self.I(SMA, self.data.High, self.n1)
        self.sma_volumne = self.I(SMA, self.data.Volume, self.n1)

    def next(self):
        if not(self.pos_open):

            if self.iteration <= self.check_every_n_iteration:
                self.iteration += 1
            else:
                self.iteration = 0

                input_data = np.array((self.data.Open[-16:], self.data.Close[-16:], self.data.High[-16:], self.data.Low[-16:], self.data.Volume[-16:]))
                central_data = input_data - np.array((self.sma_open[-16:], self.sma_close[-16:], self.sma_high[-16:], self.sma_low[-16:], self.sma_volumne[-16:]))

                out = self.model.predict(central_data.reshape(1, -1))

                if out > 100:
                    self.buy()
                    self.trailing_value = self.data.Close
                    self.pos_open = True

        else:

            if self.data.Close[-1] > self.trailing_value:
                self.trailing_value = self.data.Close

            elif self.data.Close[-1] < self.trailing_value * (1-self.max_loss):
                self.position.close()
                self.pos_open = False
        


def main_AI(AI_type_file, Data_file):
    
    # data = pd.read_csv(Data_file)
    data = GOOG

    # bt = Backtest(GOOG, AIBacktest, cash=10_000, commission=.02)
    # stats = bt.run(model='tim')

    # print(stats)

    # val = np.array((stats['Return [%]'], stats['# Trades'], stats['Win Rate [%]']))
    # weighting = np.array((20, 0.01, 1))

    # fitness = np.dot(val,weighting)

    # bt.plot()

    # AI
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(80,)))
    model.add(tf.keras.layers.Dense(20, activation="relu"))
    model.add(tf.keras.layers.Dense(1))


    keras_ga = pygad.kerasga.KerasGA(model=model,
                                     num_solutions=50)


    def fitness_func(solution, sol_index):
        model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                 weights_vector=solution)
        model.set_weights(weights=model_weights_matrix)

        # Run Model
        bt = Backtest(data, AIBacktest, cash=10_000, commission=.02)
        stats = bt.run(model=model)

        val = np.array((stats['Return [%]'], -stats['# Trades'], stats['Win Rate [%]']-50))
        weighting = np.array((1, 0.01, 1))

        fitness = np.dot(val,weighting)
        if np.isnan(fitness):
            fitness = -1000

        print('Sol={}'.format(fitness))

        return fitness

    def callback_generation(ga_instance):
        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        best_solution_weights = pygad.kerasga.model_weights_as_matrix(model=model,
                                                                weights_vector=solution)
        model.set_weights(best_solution_weights)
        model.save('AIs/Test1')


    def stop(ga_instance, fitnesses):
        # After run fetch best fitness
        ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))



    num_generations = 10
    num_parents_mating = 5
    initial_population = keras_ga.population_weights

    ga_instance = pygad.GA(num_generations=num_generations, 
                        num_parents_mating=num_parents_mating, 
                        initial_population=initial_population,
                        fitness_func=fitness_func,
                        on_generation=callback_generation,
                        on_stop=stop)


    try:
        ga_instance.run()
    except KeyboardInterrupt:
        ga_instance.plot_result(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
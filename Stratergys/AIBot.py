import tensorflow as tf
from tensorflow.keras.layers import Dense


class Bot:

    def __init__(self):
        self.network = tf.keras.Sequential()

        self.network.add(Dense(25, input_shape=(25,), activation='relu'))
        self.network.add(Dense(100, activation='relu'))
        self.network.add(Dense(50, activation='relu'))
        self.network.add(Dense(1, activation='relu'))

        self.network.compile(
                optimizer='rmsprop',
                loss='hinge',
                metrics=['accuracy']
                )

    def train(self, data):
        

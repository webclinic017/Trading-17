from backtesting.test import GOOG
import random
import numpy as np
import tensorflow as tf

def main_AI(AI_type_file, Data_file):

    print(GOOG)

    # Create data set
    test_size = 1
    train_size = 5

    testX = []
    testY = []
    trainX = []
    trainY = []

    for i in range(test_size):

        index = random.randint(5, GOOG.shape[0]-1)
        testX.append(GOOG.iloc[[index-5+j for j in range(5)]])
        testY.append(GOOG.iloc[[index]]['Close'])

    for i in range(train_size):

        index = random.randint(5, GOOG.shape[0]-1)
        # index = 5
        trainX.append(np.array(GOOG.iloc[[index-5+j for j in range(5)]]).flatten())
        trainY.append(np.array(GOOG.iloc[[index]]['Close']))

    trainX = np.array(trainX)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(25,)))
    model.add(tf.keras.layers.Dense(100, activation="relu"))
    model.add(tf.keras.layers.Dense(50, activation="relu"))
    model.add(tf.keras.layers.Dense(5))

    print(model.summary())

    print(trainX.shape)
    print(trainX[0].shape)

    print(model(trainX[0].reshape(1, -1)).numpy())



    # # Compile the model
    # network.compile(
    #                 optimizer='rmsprop',
    #                 loss='binary_crossentropy',
    #                 metrics=['accuracy']
    #                 )

    # Train the model
    # network.fit(trainX, trainY, epochs=100)



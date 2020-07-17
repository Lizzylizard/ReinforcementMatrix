#!/usr/bin/env python
import numpy as np
from numpy import random

# tensorflow
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop

from keras import backend as K

'''https://pylessons.com/CartPole-reinforcement-learning/'''

class Network():
  # constructor
  def __init__(self, mini_batch_size):
    # input shape
    input_shape = (mini_batch_size*50, )

    # network model instance
    self.model = keras.Sequential()

    # input layer
    layer_in = layers.Dense(2, activation="relu",
                            input_shape=input_shape,
                            name="layer_in")
    # layer 1
    layer_1 = layers.Dense(64,
      kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),
      bias_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),
      activation="relu", name="layer_1")

    # layer 2
    layer_2 = layers.Dense(256,
      kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),
      bias_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),
      activation="relu", name="layer_2")

    # layer 3
    layer_3 = layers.Dense(128,
      kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),
      bias_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),
      activation="relu", name="layer_3")

    # output layer
    size_out = 7
    layer_out = layers.Dense(size_out, activation="linear",
                             name="layer_out")

    # add layers to model
    self.model.add(layer_in)
    self.model.add(layer_1)
    self.model.add(layer_2)
    self.model.add(layer_3)
    self.model.add(layer_out)

    # update weights of model
    # optimizer
    opt = keras.optimizers.Adam(learning_rate=0.001)

    # loss compilation & back propagation
    self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                       optimizer=opt)

    # how does built model look
    self.model.summary()


  # update weights depending on the mini batch size
  def update_weights(self, state, targets, batch_size):
      # calculate q values one time with updated weights
      history = self.model.fit(state, targets, epochs=20,
                               batch_size=batch_size,
                     verbose=0)
      loss = history.history.get("loss")

      print("output = " + str(self.model.layers[4].output))

      # return q values
      return loss

  # use network to drive, do not update weights anymore
  # returns q-values
  def use_network(self, state):
    output = self.model.predict(state)
    return output

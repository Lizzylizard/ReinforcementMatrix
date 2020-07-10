#!/usr/bin/env python
import numpy as np
from numpy import random

# tensorflow
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop

'''https://pylessons.com/CartPole-reinforcement-learning/'''

class Network():
  # constructor
  def __init__(self, mini_batch_size):
    # input layer
    input_shape =(mini_batch_size*50, )
    input_layer = Input(shape=input_shape)

    # layer 1
    A = Dense(64, input_shape=input_shape, activation="relu",
              kernel_initializer='he_uniform')(input_layer)
    # layer 2
    B = Dense(256, activation="relu",
              kernel_initializer='he_uniform')(A)
    # layer 3
    C = Dense(128, activation="relu",
              kernel_initializer='he_uniform')(B)
    # output layer
    size_out = 7
    output_layer = Dense(size_out, activation="linear",
                         kernel_initializer='he_uniform')(C)
    # define whole model
    self.model = Model(inputs=input_layer, outputs=output_layer,
                  name="Three Pi Model")
    # update weights of model
    self.model.compile(loss="mse", optimizer=RMSprop(lr=0.00025,
                                                  rho=0.95,
                                                epsilon=0.01),
                  metrics=["accuracy"])

    # self.model.summary()


  # update weights depending on the mini batch size
  def update_weights(self, state, targets, batch_size):
      # calculate q values one time with updated weights
      history = self.model.fit(state, targets, batch_size=batch_size,
                     verbose=0)
      loss = history.history.get("loss")

      # return q values
      return loss

  # use network to drive, do not update weights anymore
  # returns q-values
  def use_network(self, state):
    output = self.model.predict(state)
    return output

  # copy all of the layers, weights and biases to the target network
  def copy(self, target_net):
    self.sess.run(tf.assign(target_net.W1, self.W1)) ;
    # W3
    self.sess.run(tf.assign(target_net.W3, self.W3)) ;
    # b1
    self.sess.run(tf.assign(target_net.b1, self.b1)) ;
    # b2
    self.sess.run(tf.assign(target_net.b3, self.b3)) ;

    return target_net

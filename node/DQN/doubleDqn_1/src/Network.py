#!/usr/bin/env python
import numpy as np
from numpy import random

# tensorflow
import numpy as np
import tensorflow as tf
from tensorflow import keras

class Network():
  # constructor
  def __init__(self, mini_batch_size, size_layer1, session):
    self.mini_batch_size = mini_batch_size
    self.sess = session

    # input layer
    # input is state
    # state is a scalar number (0 to 7)
    # self.input = tf.compat.v1.placeholder(tf.float64, [None,1])
    self.input = tf.compat.v1.placeholder(tf.float64, [1,
                                                       mini_batch_size*50])
    self.a0 = tf.reshape(self.input, (1, -1));

    # output
    # output are q_values for the possible actions
    out_0 = 1
    out_1 = 7

    # targets
    self.targets_p = tf.compat.v1.placeholder(tf.float64, [None,
                                                           out_1],
                                              name="targets")

    # layer 1
    self.W1 = tf.Variable(np.random.uniform(-0.01, 0.01,
                                            [mini_batch_size*50,
                                                    size_layer1]))
    self.b1 = tf.Variable(np.random.uniform(-0.01, 0.01,
                                            [size_layer1]),
                          name="b1")
    self.a1 = tf.compat.v1.nn.relu_layer(self.a0, self.W1, self.b1,
                                         "a1")

    # layer 2

    # layer 3
    self.W3 = tf.Variable(np.random.uniform(-0.01, 0.01, [size_layer1,
                                                      out_1]))
    self.b3 = tf.Variable(np.random.uniform(-0.01, 0.01, [out_0, out_1]),
                          name="b3")
    self.a3 = tf.compat.v1.matmul(self.a1, self.W3) + self.b3

    # AG modification!!
    # loss
    self.loss = tf.reduce_mean(
      tf.compat.v1.losses.mean_squared_error(self.targets_p, self.a3))

    # gradient descent & backpropagation
    self.sgdObj = tf.train.GradientDescentOptimizer(
      learning_rate=0.001)
    # var_list = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
    var_list = [self.W1, self.W3, self.b1, self.b3]
    self.updateOp = self.sgdObj.minimize(self.loss)

    ### until here!
    # initialize
    self.sess.run(tf.compat.v1.global_variables_initializer())


  # update weights depending on the mini batch size
  def update_weights(self, state, targets):
      # run session (generate ouput from input)
      #for i in range(epochs):
      '''
      batchIndex = 0
      stop = len(images) / self.mini_batch_size
    while(batchIndex < stop):
      # create smaller batch from input data
      batch = images \
        [batchIndex*self.mini_batch_size:(batchIndex+1)*self.mini_batch_size,
        0:len(images[0])]
      targetIndex = batchIndex
      my_targets = np.zeros(shape=[1, 8])
      my_targets[0] = targets[targetIndex]

      # loss
      self.loss = tf.reduce_mean(
        tf.compat.v1.losses.mean_squared_error(my_targets, self.a3))

      # gradient descent & backpropagation
      self.sgdObj = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)
      # var_list = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
      var_list = [self.W1, self.W3, self.b1, self.b3]
      self.updateOp = self.sgdObj.minimize(self.loss,
                                           var_list=var_list)


      #print("batched images shape = " + str(np.shape(batch)))

      # calculate loss and update weights for batch data
      _, loss2 = self.sess.run([self.updateOp, self.loss], feed_dict={
        self.input: batch, self.targets_p: my_targets})

      print("loss = " + str(loss2))

      batchIndex += 1
      # repeat until total data is processed
      '''

      # calculate q values one time with updated weights
      output, loss2, _ = self.sess.run([self.a3, self.loss,
                                        self.updateOp],
                             feed_dict={ self.input: state,
                                         self.targets_p: targets})

      #print("y = \n" + str(output))
      #print("loss = " + str(loss2))

      # return q values
      return output, loss2

  # use network to drive, do not update weights anymore
  # returns q-values
  def use_network(self, state):
    output = self.sess.run(self.a3, feed_dict={
      self.input: state})
    return output

  # copy all of the layers, weights and biases to the target network
  def copy(self, target_net):
    self.sess.run(tf.assign(target_net.W1, self.W1))
    # W3
    self.sess.run(tf.assign(target_net.W3, self.W3))
    # b1
    self.sess.run(tf.assign(target_net.b1, self.b1))
    # b2
    self.sess.run(tf.assign(target_net.b3, self.b3))


    return target_net

  # polyak averaging
  # delta' = tau*delta + (1-tau)*delta' where
  # delta is the value of policy network and
  # delta' is the value of target network
  '''https://towardsdatascience.com/double-deep-q-networks-905dd8325412'''
  def polyak(self, p, t, tau):
    dif = tf.math.subtract(tf.cast(1.0, tf.float64), tau)
    mult1 = tf.multiply(tau, p)
    mult2 = tf.multiply(dif, t)
    sum = tf.math.add(mult1, mult2)
    return sum

  # copy all of the layers, weights and biases to the target
  # network through Polyak averaging
  # since only 0.01 times the actual value is copied, the target
  # network can be updated at EVERY step
  '''https: // adgefficiency.com / dqn - tuning /'''
  def copy_softly(self, target_net):
    tau1 = 0.01
    tau = tf.cast(tau1, tf.float64)
    # W1
    sum = self.polyak(self.W1, target_net.W1, tau)
    self.sess.run(tf.assign(target_net.W1, sum))
    # W3
    sum2 = self.polyak(self.W3, target_net.W3, tau)
    self.sess.run(tf.assign(target_net.W3, sum2))
    # b1
    sum3 = self.polyak(self.b1, target_net.b1, tau)
    self.sess.run(tf.assign(target_net.b1, sum3))
    # b3
    sum4 = self.polyak(self.b3, target_net.b3, tau)
    self.sess.run(tf.assign(target_net.b3, sum4))

    return target_net

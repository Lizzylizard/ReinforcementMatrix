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

    '''----------------------model definition----------------------'''
    # input layer
    self.input = tf.compat.v1.placeholder(tf.float64, \
                                          [None,
                                           mini_batch_size*50],
                                          name="input")
    # self.a0 = tf.reshape(self.input, (1, -1))
    self.a0 = self.input

    # output
    # output are q_values for the possible actions
    out_1 = 7

    # layer 1
    self.W1 = tf.Variable \
      (np.random.uniform(-0.01,0.01,[mini_batch_size*50,
                                     size_layer1]), name="W1")
    self.b1 = tf.Variable \
      (np.random.uniform(-0.01, 0.01,[size_layer1]),name="b1")
    self.a1 = tf.compat.v1.nn.relu_layer \
      (self.a0, self.W1, self.b1, name="a1")

    # layer 2
    self.W2 = tf.Variable \
      (np.random.uniform(-0.01, 0.01,[size_layer1,256]), name="W2")
    self.b2 = tf.Variable \
      (np.random.uniform(-0.01, 0.01, [256]),name="b2")
    self.a2 = tf.compat.v1.matmul(self.a1, self.W2) + self.b2

    # layer 3
    self.W3 = tf.Variable \
      (np.random.uniform(-0.01, 0.01,[256, 125]), name="W3")
    self.b3 = tf.Variable \
      (np.random.uniform(-0.01, 0.01, [125]),name="b3")
    self.a3 = tf.compat.v1.matmul(self.a2, self.W3) + self.b3

    # layer end
    self.W_end = tf.Variable \
      (np.random.uniform(-0.01, 0.01,[125, out_1]), name="W_end")
    self.b_end = tf.Variable \
      (np.random.uniform(-0.01, 0.01, [out_1]),name="b_end")
    self.a_end = tf.compat.v1.matmul(self.a3, self.W_end) + self.b_end

    '''--------------------loss calculation--------------------'''
    # targets
    self.targets_p = tf.compat.v1.placeholder(tf.float64, [None,
                                                           out_1],
                                              name="targets")
    # AG modification!!
    # loss
    self.loss = tf.reduce_mean(
      tf.compat.v1.losses.mean_squared_error(self.targets_p,
                                             self.a_end))

    # gradient descent & backpropagation
    self.sgdObj = tf.train.GradientDescentOptimizer(
      learning_rate=0.001)
    self.updateOp = self.sgdObj.minimize(self.loss)

    ### until here!

    '''-----------------------initialization-----------------------'''
    # initialize
    self.sess.run(tf.compat.v1.global_variables_initializer())

  # old
  # update weights depending on the mini batch size
  def update_weights(self, state, targets):
      # calculate q values one time with updated weights
      loss2, _ = self.sess.run([self.loss, self.updateOp],
                             feed_dict={ self.input: state,
                                         self.targets_p: targets})
      # return q values
      return loss2

  # calculate q-values based on current weights and biases
  def use_network(self, state):
    output = self.sess.run(self.a_end, feed_dict={
      self.input: state})
    return output

  # copy all of the layers, weights and biases to the target network
  def copy(self, target_net):
    # W1
    self.sess.run(tf.assign(target_net.W1, self.W1))
    # W2
    self.sess.run(tf.assign(target_net.W2, self.W2))
    # W3
    self.sess.run(tf.assign(target_net.W3, self.W3))
    # W end
    self.sess.run(tf.assign(target_net.W_end, self.W_end))
    # b1
    self.sess.run(tf.assign(target_net.b1, self.b1))
    # b2
    self.sess.run(tf.assign(target_net.b2, self.b2))
    # b3
    self.sess.run(tf.assign(target_net.b3, self.b3))
    # b end
    self.sess.run(tf.assign(target_net.b_end, self.b_end))

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
    assign_w1 = tf.assign(target_net.W1, sum)
    #self.sess.run(tf.assign(target_net.W1, sum))
    # W2
    sum2 = self.polyak(self.W2, target_net.W2, tau)
    assign_w2 = tf.assign(target_net.W2, sum2)
    #self.sess.run(tf.assign(target_net.W2, sum2))
    # W3
    sum3 = self.polyak(self.W3, target_net.W3, tau)
    assign_w3 = tf.assign(target_net.W3, sum3)
    #self.sess.run(tf.assign(target_net.W3, sum3))
    # W end
    sum_W_end = self.polyak(self.W_end, target_net.W_end, tau)
    assign_w_end = tf.assign(target_net.W_end, sum_W_end)
    #self.sess.run(tf.assign(target_net.W_end, sum_W_end))
    # b1
    sum4 = self.polyak(self.b1, target_net.b1, tau)
    assign_b1 = tf.assign(target_net.b1, sum4)
    #self.sess.run(tf.assign(target_net.b1, sum3))
    # b2
    sum5 = self.polyak(self.b2, target_net.b2, tau)
    assign_b2 = tf.assign(target_net.b2, sum5)
    #self.sess.run(tf.assign(target_net.b2, sum5))
    # b3
    sum6 = self.polyak(self.b3, target_net.b3, tau)
    assign_b3 = tf.assign(target_net.b3, sum6)
    #self.sess.run(tf.assign(target_net.b3, sum6))
    # b end
    sum_b_end = self.polyak(self.b_end, target_net.b_end, tau)
    assign_b_end = tf.assign(target_net.b_end, sum_b_end)
    #self.sess.run(tf.assign(target_net.b_end, sum_b_end))

    self.sess.run([assign_w1, assign_w2, assign_w3, assign_w_end,
                   assign_b1, assign_b2, assign_b3, assign_b_end])

    return target_net

#!/usr/bin/env python3
import numpy as np
from numpy import random

# tensorflow
import numpy as np
import tensorflow as tf
from tensorflow import keras

class Network():
  # constructor
  def __init__(self, images, size_layer1, size_layer2, session):
    self.sess = session
    # shape input layer
    # print("Shape of images = " + str(np.shape(images)))
    dim0 = len(images)
    # print("dim0 = " + str(dim0))
    dim1 = len(images[0])
    # print("dim1 = " + str(dim1))
    dim2 = len(images[0][0])
    # print("dim2 = " + str(dim2))
    size_input = dim0 * dim1 * dim2
    # print("input size = " + str(size_input))

    # shape output layer
    # print("Shape targets = " + str(np.shape(targets)))
    out0 = 1
    # print("out0 = " + str(out0))
    out1 = 7
    # print("out1 = " + str(out1))

    # input
    self.input = tf.compat.v1.placeholder(tf.float64, [dim0, dim1,
                                                       dim2])
    self.a0 = tf.reshape(self.input, (-1, size_input))
    # print("Shape a0 = " + str(np.shape(a0)))

    # targets
    self.targets_p = tf.compat.v1.placeholder(tf.float64,
                                              [out0, out1])

    # layer 1
    self.W1 = tf.Variable(np.random.uniform(-1, 1, [size_input,
                                                    size_layer1]))
    self.b1 = tf.Variable(np.random.uniform(-1, 1,
                                            [size_layer1]),
                          name="b1")
    self.a1 = tf.compat.v1.nn.relu_layer(self.a0, self.W1, self.b1,
                                         "a1")

    # layer 2
    self.W2 = tf.Variable(np.random.uniform(-1, 1, [size_layer1,
                                                    size_layer2]))
    self.b2 = tf.Variable(np.random.uniform(-1, 1,
                                            [size_layer2]),
                          name="b2")
    self.a2 = tf.compat.v1.nn.relu_layer(self.a1, self.W2, self.b2,
                                         "a2")

    # output
    self.W3 = tf.Variable(np.random.uniform(-1, 1, [size_layer2,
                                                    out1]))
    self.b3 = tf.Variable(np.random.uniform(-1, 1, [out0,
                                                    out1]),
                          name="b3")
    self.a3 = tf.compat.v1.matmul(self.a2, self.W3) + self.b3
    # print("Shape a3 = " + str(np.shape(a3)))
    self.a4 = tf.compat.v1.nn.softmax(self.a3)

    # initialize
    self.sess.run(tf.compat.v1.global_variables_initializer())

    # list for easier copying
    self.list = [self.sess, self.input, self.targets_p, self.a0,
                 self.a1, self.a2, self.a3, self.a4, self.W1,
                 self.W2, self.W3, self.b1, self.b2, self.b3]


  def update_weights(self, images, epochs, targets, learning_rate):
    # loss
    self.loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(targets, self.a4))
    # loss = tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(
    # targets, self.a3))

    # gradient descent & backpropagation
    '''
    sgdObj = tf.compat.v1.train.AdamOptimizer (
      learning_rate=learning_rate, beta1=0.9, beta2=0.999, 
      epsilon=1e-08, use_locking=False, name='Adam')
    updateOp = sgdObj.minimize(loss)
    '''
    self.sgdObj = tf.train.GradientDescentOptimizer(
      learning_rate=learning_rate)
    self.updateOp = self.sgdObj.minimize(self.loss,
                                    var_list=[self.W1, self.W2,
                                              self.W3,
                                              self.b1, self.b2,
                                              self.b3])

    # run session (generate ouput from input)
    for i in range(epochs):
      # output, loss2, upOp = sess.run([self.a3, loss,
      #                                updateOp], feed_dict={
      #  self.input: images, self.targets_p: targets})
      output, loss2, upOp, a2S, a1S, a0S, w1S, b1S, w2S, \
      b2S, w3S, b3S, a4S = self.sess.run([self.a3, self.loss,
                                     self.updateOp,
                                     self.a2, self.a1, self.a0,
                                     self.W1, self.b1, self.W2,
                                     self.b2, self.W3, self.b3,
                                     self.a4],
                                    feed_dict={
                                      self.input: images,
                                      self.targets_p: targets})
      # print("a0: ", a0S)
      # print("w1: ", w1S)
      # print("b1: ", b1S)
      # print("a1: ", a1S)
      # print("w2: ", w2S)
      # print("b2: ", b2S)
      # print("a2: ", a2S)
      # print("w3: ", w3S)
      # print("b3: ", b3S)
      # print("a4: ", a4S)
      # print("targets: ", targets)

      # print("\n\nShape output = " +  str(np.shape(output)))
      print("y = " + str(output))
      print("Loss = " + str(loss2))
      # print("sgdObj = " + str(sgdObj))
      # print("Updated = " + str(upOp))
      # print("-"*60)
      # print(output)

    # print("Weights = " + str(updatedWeights))
    return output

  # use network to drive, do not update weights anymore
  # returns q-values
  def use_network(self, images):
    output = self.sess.run(self.a3, feed_dict={self.input: images})
    return output

  '''
  # 'one-hot' coding for target values
  def fill_targets_old(self, state):
    for i in range(len(self.targets[0])):
      self.targets[0, i] = 0
    if not (state == self.lost_line):
      self.targets[0, state] = 1
    # if robot lost the line -> no way of knowing what the best
    # possible next action might be, so choose randomly every time
    else:
      rand = np.random.randint(low=0, high=(len(self.targets)),
                               size=1)
      self.targets[0, rand] = 1
    return self.targets

    # 'one-hot' coding for target values

  def fill_targets(self, action, reward):
    for i in range(len(self.targets[0])):
      self.targets[0, i] = 0
    self.targets[0, action] = reward
    return self.targets
  '''

  def copy(self, target_net):
    target_net.sess = self.list[0]
    target_net.input = self.list[1]
    target_net.targets_p = self.list[2]
    target_net.a0 = self.list[3]
    target_net.a1 = self.list[4]
    target_net.a2 = self.list[5]
    target_net.a3 = self.list[6]
    target_net.a4 = self.list[7]
    target_net.W1 = self.list[8]
    target_net.W2 = self.list[9]
    target_net.W3 = self.list[10]
    target_net.b1 = self.list[11]
    target_net.b2 = self.list[12]
    target_net.b3 = self.list[13]
    return target_net
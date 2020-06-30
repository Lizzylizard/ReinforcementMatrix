#!/usr/bin/env python3
import numpy as np
from numpy import random

# tensorflow
import numpy as np
import tensorflow as tf
from tensorflow import keras

class Network():
  # constructor
  '''
  # size layer 1: integer
  #   size of the hidden layer
  # session: session object of the tensorflow graph
  # batch_size: integer
  #   number of samples that go together into the backpropagation
  #   and loss calculation
  '''
  def __init__(self, size_layer1, session, batch_size):
    self.batch_size = batch_size
    self.sess = session

    # shape input layer
    # we get 4 memory samples
    dim0 = 1
    print(dim0)
    # each memory sample has 200 entries (4 pictures with each 50
    # pixels)
    dim1 = 50
    # if we flatten the 2D matrix [dim0, dim1] this will be the
    # number of elements in the resulting 1D array
    size_input = dim0 * dim1

    # input layer
    self.input = tf.compat.v1.placeholder(tf.float64, [dim0, dim1])
    # reshape input layer to a 1D array
    self.a0 = tf.reshape(self.input, (-1, size_input))
    #print("Shape a0 = " + str(np.shape(self.a0)))

    # targets
    self.targets_p = tf.compat.v1.placeholder(tf.float64, [1, 7],
                                              name="targets")

    # layer 1
    self.W1 = tf.Variable(np.random.uniform(0.01, 1, [size_input,
                                                    size_layer1]))
    self.b1 = tf.Variable(np.random.uniform(0.01, 1,
                                            [size_layer1]),
                          name="b1")
    self.a1 = tf.compat.v1.nn.relu_layer(self.a0, self.W1, self.b1,
                                         "a1")

    '''
    # layer 2
    self.W2 = tf.Variable(np.random.uniform(0.01, 1, [size_layer1,
                                                    size_layer2]))
    self.b2 = tf.Variable(np.random.uniform(0.01, 1,
                                            [size_layer2]),
                          name="b2")
    self.a2 = tf.compat.v1.nn.relu_layer(self.a1, self.W2, self.b2,
                                         "a2")
    '''

    # output
    out_0 = 1
    out_1 = 7
    self.W3 = tf.Variable(np.random.uniform(0.01, 1, [size_layer1,
                                                      out_1]))
    self.b3 = tf.Variable(np.random.uniform(0.01, 1, [out_0, out_1]),
                          name="b3")
    self.a3 = tf.compat.v1.matmul(self.a1, self.W3) + self.b3
    # print("Shape a3 = " + str(np.shape(a3)))

    # initialize
    self.sess.run(tf.compat.v1.global_variables_initializer())

    # list for easier copying
    #self.list = [self.sess, self.input, self.targets_p, self.a0,
                 #self.a1, self.a2, self.a3, self.a4, self.W1,
                 #self.W2, self.W3, self.b1, self.b2, self.b3]
    self.list = [self.sess, self.input, self.targets_p, self.a0,
                 self.a1, self.a3, self.W1, self.W3,
                 self.b1, self.b3]


  # update weights depending on the batch size
  '''
  # images: 2D input data array
  #   rows = number of extracted memory samples
  #   cols = flattened consecutive images
  # epochs: integer
  #   number of learning epochs
  # targets: 2D array
  #   only one row
  #   in columns: reward at index current action, 0 else
  # learning rate: float
  '''
  def update_weights(self, images, epochs, targets, learning_rate):
    # loss
    self.loss = tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(
      targets, self.a3))

    # gradient descent & backpropagation
    self.sgdObj = tf.train.GradientDescentOptimizer(
      learning_rate=learning_rate)
    #var_list = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
    var_list = [self.W1, self.W3, self.b1, self.b3]
    self.updateOp = self.sgdObj.minimize(self.loss, var_list=var_list)

    # run session (generate ouput from input)
    for i in range(epochs):
      batchIndex = 0
      stop = len(images) / self.batch_size
      '''
      #print("len images: "+  str(len(images)))
      #print("bath_size: "+ str(self.batch_size))
      '''
      while(batchIndex < stop):
        '''
        #print("buffered images shape = " + str(np.shape(images)))
        #print("1: ", batchIndex*self.batch_size)
        #print("2: ", (batchIndex+1)*self.batch_size)
        '''
        # create smaller batch from input data
        batch = images \
          [batchIndex*self.batch_size:(batchIndex+1)*self.batch_size,
          0:len(images[0])]
        '''
        #print("batched images shape = " + str(np.shape(batch)))
        '''
        # calculate loss and update weights for batch data
        self.sess.run(self.updateOp, feed_dict={self.input: batch,
                                                self.targets_p: targets})
        batchIndex += 1
        # repeat until total data is processed

      # calculate q values one time with updated weights
      output, loss2 = self.sess.run([self.a3, self.loss], feed_dict={
        self.input: images,
                                                 self.targets_p: targets})

      print("y = \n" + str(output))
      print("loss = " + str(loss2))

    # return q values
    return output

  # use network to drive, do not update weights anymore
  # returns q-values
  def use_network(self, images, targets):
    output = self.sess.run(self.a3, feed_dict={self.input: images,
                                               self.targets_p: targets})
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

  # copy all of the layers, weights and biases to the target network
  def copy(self, target_net):
    target_net.sess = self.list[0]
    target_net.input = self.list[1]
    target_net.targets_p = self.list[2]
    target_net.a0 = self.list[3]
    target_net.a1 = self.list[4]
    #target_net.a2 = self.list[5]
    target_net.a3 = self.list[5]
    target_net.W1 = self.list[6]
    #target_net.W2 = self.list[9]
    target_net.W3 = self.list[7]
    target_net.b1 = self.list[8]
    #target_net.b2 = self.list[12]
    target_net.b3 = self.list[9]
    return target_net
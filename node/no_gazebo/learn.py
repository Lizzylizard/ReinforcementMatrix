#!/usr/bin/env python

import cv2 as cv
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import tensorflow as tf

image_no = ""
actions = {
  0:  "sharp left",
  1:  "left",
  2:  "slightly left",
  3:  "forward",
  4:  "slightly right",
  5:  "right",
  6:  "sharp right",
  7:  "stop"
}
states = {
  0:  "far left",
  1:  "left",
  2:  "slightly left",
  3:  "middle",
  4:  "slightly right",
  5:  "right",
  6:  "far right",
  7:  "lost"
}
a0, a1, a3, W1, W3, b1, b3, input, targets_p = None, None, None, \
                                               None, None, None, \
                                               None, None, None
sess = tf.compat.v1.Session()

def loadImage(random):
  path = '/home/elisabeth/Dokumente/GazeboImages/v3-learning/img'
  path += str(random)
  path += '.jpg'
  img = cv.imread(path)
  return img

def crop(img):
  height = (len(img))
  return img[(height/2):((height/2)+1)]

def segmentation(img):
  light_black = (0)
  dark_black = (10)
  mask = cv.inRange(img, light_black, dark_black)
  return mask

def preprocessImage(img):
  small_img = crop(img)
  seg_img = segmentation(small_img)
  return seg_img

def get_rand_img():
  global image_no
  random = np.random.randint(low=1, high=4365)
  image_no = str(random)
  img = loadImage(random)
  proc_img = preprocessImage(img)
  return proc_img

def show_img(img):
  plt.imshow(img)
  global image_no
  title = "Image " + str(image_no)
  plt.title(title)
  plt.show()


def count_pxl(img):
  result = 0
  # for-loop dummy
  for i in range(1):
    k = 0
    j = img[i, k]
    # print("J = " + str(j))
    while j < 251: # as long as current pixel is black (is background)
      result += 1
      k += 1
      if (k < len(img[i])):  # check if it's still in bounds
        j = img[i, k]  # jump to next pixel
      else:
        break

  return result

def get_line_state(img):
  # get left edge of line
  left = count_pxl(img)
  # flip image vertically (pixel on the right will be on the left,
  # pixel on top stays on top)
  reversed_img = np.flip(img, 1)
  # get right edge of line (start counting from the right)
  right = count_pxl(reversed_img)

  # get width of image (should be 50)
  width = np.size(img[0])

  # get right edge of line (start counting from the left)
  absolute_right = width - right
  # middle is between left and right edge
  middle = float(left + absolute_right) / 2.0

  if (left >= (width * (99.0 / 100.0)) or right >= (
    width * (99.0 / 100.0))):
    # line is lost
    # just define that if line is ALMOST lost, it is completely
    # lost, so terminal state gets reached
    state = 7
  elif (middle >= (width * (0.0 / 100.0)) and middle <= (
    width * (2.5 / 100.0))):
    # line is far left
    state = 0
  elif (middle > (width * (2.5 / 100.0)) and middle <= (
    width * (21.5 / 100.0))):
    # line is left
    state = 1
  elif (middle > (width * (21.5 / 100.0)) and middle <= (
    width * (40.5 / 100.0))):
    # line is slightly left
    state = 2
  elif (middle > (width * (40.5 / 100.0)) and middle <= (
    width * (59.5 / 100.0))):
    # line is in the middle
    state = 3
  elif (middle > (width * (59.5 / 100.0)) and middle <= (
    width * (78.5 / 100.0))):
    # line is slightly right
    state = 4
  elif (middle > (width * (78.5 / 100.0)) and middle <= (
    width * (97.5 / 100.0))):
    # line is right
    state = 5
  elif (middle * (97.5 / 100.0)) and middle <= (
    width * (100.0 / 100.0)):
    # line is far right
    state = 6
  else:
    # line is lost
    state = 7

  return state

def get_reward(state, action):
  reward = 0
  if(state == action):
    reward += 10

  if(state == 0):
    if(action == 1 or action == 2):
      reward += 5
    else:
      reward += -1000

  if(state == 1):
    if(action == 0 or action == 2):
      reward += 5
    else:
      reward += -1000

  if(state == 2):
    if(action == 0 or action == 1):
      reward += 5
    else:
      reward += -1000

  if(state == 3):
    if(action == 2 or action == 4):
      reward += 10
    else:
      reward += -1000

  if(state == 4):
    if(action == 5 or action == 6):
      reward += 5
    else:
      reward += -1000

  if(state == 5):
    if(action == 4 or action == 6):
      reward += 5
    else:
      reward += -1000

  if(state == 6):
    if(action == 4 or action == 6):
      reward += 5
    else:
      reward += -1000

  if(state == 7):
    if(action == 7):
      reward += 0
    else:
      reward += -1000

  return reward

def get_rand_action():
  action = np.random.randint(low=0, high=8)
  return action

def initialize_network(images):
    global a0, a1, a3, W1, W3, b1, b3, input, targets_p, sess
    size_layer1 = 5
    size_input = len(images) * (len(images[0]))
    out_0 = 1
    out_1 = 8

    # input layer
    input = tf.compat.v1.placeholder(tf.float64, [len(images),
                                                       len(images[
                                                             0])])

    # reshape input layer to a 1D array
    a0 = tf.reshape(input, (-1, size_input))
    #print("Shape a0 = " + str(np.shape(self.a0)))

    # targets
    targets_p = tf.compat.v1.placeholder(tf.float64, [out_0, out_1],
                                              name="targets")

    # layer 1
    W1 = tf.Variable(np.random.uniform(0.01, 1, [size_input,
                                                    size_layer1]))
    b1 = tf.Variable(np.random.uniform(0.01, 1,
                                            [size_layer1]),
                          name="b1")
    a1 = tf.compat.v1.nn.relu_layer(a0, W1, b1, "a1")

    # layer 2

    # layer 3
    W3 = tf.Variable(np.random.uniform(0.01, 1, [size_layer1, out_1]))
    b3 = tf.Variable(np.random.uniform(0.01, 1, [out_0, out_1]),
                          name="b3")
    a3 = tf.compat.v1.matmul(a1, W3) + b3

    # initialize
    sess.run(tf.compat.v1.global_variables_initializer())

def update_weights(images, epochs, targets, learning_rate):
  global a0, a1, a3, W1, W3, b1, b3, input, targets_p, sess

  # loss
  loss = tf.reduce_mean(
    tf.compat.v1.losses.mean_squared_error(targets, a3))

  # gradient descent & backpropagation
  sgdObj = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate)
  # var_list = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
  var_list = [W1, W3, b1, b3]
  updateOp = sgdObj.minimize(loss,var_list=var_list)

  for i in range(epochs):
    # calculate loss and update weights for batch data
    _, loss2, output = sess.run([updateOp, loss, a3], feed_dict={
      input: images, targets_p: targets})

  # return q values
  return output, loss2

def get_targets(action, reward):
  targets = np.zeros(shape=[1, 8])
  for i in range(len(targets[0])):
    if(i == action):
      targets[0, i] = reward
  return targets

def get_best_action(output):
  action = np.argmax(output)
  return action

######################################################################

def get_next_image(cnt):
  path = '/home/elisabeth/Dokumente/GazeboImages/v2-fullRound/img'
  cnt += 1
  path += str(cnt)
  path += '.jpg'
  img = cv.imread(path)
  proc_img = preprocessImage(img)
  return proc_img

def use_network(images, epochs):
  global a3, input, sess

  for i in range(epochs):
    # calculate loss and update weights for batch data
    output = sess.run(a3, feed_dict={input: images})

  # return q values
  return output

######################################################################

if __name__ == '__main__':
  # learn
  global actions
  global states
  output = np.zeros(8)
  img = get_rand_img()
  initialize_network(img)
  for i in range(500):
    img = get_rand_img()
    state = get_line_state(img)
    #best_action = get_best_action(output)
    best_action = get_rand_action()
    reward = get_reward(state, best_action)
    targets = get_targets(best_action, reward)
    output, loss = update_weights(img, 1, targets, 0.001)

    print("Learning, Episode = " + str(i))
    print("-"*70)
    print("State = " + str(states.get(state)))
    print("Action = " + str(actions.get(best_action)))
    print("Reward = " + str(reward))
    print("Targets = " + str(targets))
    print("Output = \n" + str(output))
    print("Loss = " + str(loss))
    print("-" * 70)

  # test
  for cnt in range(1140):
    img = get_next_image(cnt)
    state = get_line_state(img)
    output = use_network(img, 1)
    action = get_best_action(output)

    print("Testing, Image number = " + str(cnt))
    print("-"*70)
    print("State = " + str(states.get(state)))
    print("Action = " + str(actions.get(action)))
    print("Output = \n" + str(output))
    print("-" * 70)
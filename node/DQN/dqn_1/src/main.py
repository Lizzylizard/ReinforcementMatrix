#!/usr/bin/env python

# import own scripts
import matrix as bt
import image as mi

# import numpy
import numpy as np
from numpy import random

# tensorflow
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import OpenCV libraries and tools
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

# ROS
import rospy
import rospkg
from std_msgs.msg import String, Float32, Int32
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

# other
import math
import time
import random


class Node:
  # callback; copies the received image into a global numpy-array
  def cam_im_raw_callback(self, msg):
    # convert ROS image to cv image, copy it and save it as a global
    # numpy-array
    img = self.imgHelper.img_conversion(msg)
    self.my_img = np.copy(img)

    # count the received images
    self.img_cnt += 1

  # wait until a new image is received
  def get_image(self):
    nr_images = self.img_cnt
    while (self.img_cnt <= nr_images):
      pass
    return self.my_img

  # wait until image stack is full
  def get_stack_of_images(self):
    nr_images = self.img_cnt
    mod_old = -1
    while (self.img_cnt <= (nr_images + self.stack_of_images)):
      mod_new = self.img_cnt % self.stack_of_images
      if not (mod_new == mod_old):
        self.img_stack[mod_new][0] = self.my_img
      mod_old = mod_new
    return self.img_stack

  # constructor
  def __init__(self):
    # helper classes
    self.bot = bt.Bot()  # reward + q-matrix
    self.imgHelper = mi.MyImage()  # image processing

    # global variables
    # images
    self.my_img = np.zeros(50)  # current image
    # next couple of (four) consecutive images
    self.stack_of_images = 4
    self.img_stack = np.zeros(shape=[self.stack_of_images, 1,
                                   len(self.my_img)])
    self.img_cnt = 0  # number of images received

    self.vel_msg = Twist()  # message to post on topic /cmd_vel
    self.start = time.time()  # starting time
    self.explorationMode = False  # Bool: exploring (True) or exploiting (False)?

    # terminal states
    self.lost_line = 7
    self.stop_action = 7

    # current state and action (initially negative)
    self.curr_state = -1
    self.curr_action = -1

    # last state (initially negative)
    self.last_state = -1000

    # starting coordinates of the robot
    self.x_position = -0.9032014349
    self.y_position = -6.22487658223
    self.z_position = -0.0298790967155

    # inital values
    self.max_episodes = 2000
    self.speed = 15.0

    # deviation from speed to turn the robot to the left or to the
    # right
    self.sharp = self.speed * (
      1.0 / 7.0)  # sharp curve => big difference
    self.middle = self.speed * (
      1.0 / 8.5)  # middle curve => middle difference
    self.slightly = self.speed * (
      1.0 / 10.0)  # slight curve => slight difference

    '''
    Did work with 'old' reward function:
    #inital values
    self.speed = 20.0

    #deviation from speed so average speed stays the same
    self.sharp = self.speed * (1.0/7.0)         #sharp curve => big difference
    self.middle = self.speed * (1.0/8.5)        #middle curve => middle difference
    self.slightly = self.speed * (1.0/10.0)     #slight curve => slight difference
    '''

    # strings to display actions and states
    self.action_strings = {
      0: "sharp left",
      1: "left",
      2: "slightly left",
      3: "forward",
      4: "slightly right",
      5: "right",
      6: "sharp right",
      7: "stop"
    }

    self.state_strings = {
      0: "far left",
      1: "left",
      2: "slightly left",
      3: "middle",
      4: "slightly right",
      5: "right",
      6: "far right",
      7: "lost"
    }

    # neural network
    self.targets = np.zeros(shape=[1, len(self.action_strings)],
                            dtype='float64')
    self.q_values = None

    # publisher to publish on topic /cmd_vel
    self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist,
                                              queue_size=100)
    # initializing ROS-node
    rospy.init_node('reinf_matrix_driving', anonymous=True)
    # subscribe to topic '/camera/image_raw' using rospy.Subscriber
    # class
    self.sub = rospy.Subscriber('/camera/image_raw', Image,
                                self.cam_im_raw_callback)

  # sets fields of Twist variable so robot drives sharp left
  def sharp_left(self):
    vel_msg = Twist()
    vel_msg.linear.x = self.speed + self.sharp
    vel_msg.linear.y = self.speed - self.sharp
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    # print("SHARP LEFT")
    return vel_msg

  # sets fields of Twist variable so robot drives slightly left
  def slightly_left(self):
    vel_msg = Twist()
    vel_msg.linear.x = self.speed + self.slightly
    vel_msg.linear.y = self.speed - self.slightly
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    # print("SLIGHTLY LEFT")
    return vel_msg

  # sets fields of Twist variable so robot drives left
  def left(self):
    vel_msg = Twist()
    vel_msg.linear.x = self.speed + self.middle
    vel_msg.linear.y = self.speed - self.middle
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    # print("LEFT")
    return vel_msg

  # sets fields of Twist variable so robot drives forward
  def forward(self):
    vel_msg = Twist()
    vel_msg.linear.x = self.speed
    vel_msg.linear.y = self.speed
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    # print("FORWARD")
    return vel_msg

  # sets fields of Twist variable so robot drives slightly right
  def slightly_right(self):
    vel_msg = Twist()
    vel_msg.linear.x = self.speed - self.slightly
    vel_msg.linear.y = self.speed + self.slightly
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    # print("SLIGHTLY RIGHT")
    return vel_msg

  # sets fields of Twist variable so robot drives right
  def right(self):
    vel_msg = Twist()
    vel_msg.linear.x = self.speed - self.middle
    vel_msg.linear.y = self.speed + self.middle
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    # print("RIGHT")
    return vel_msg

  # sets fields of Twist variable so robot drives sharp right
  def sharp_right(self):
    vel_msg = Twist()
    vel_msg.linear.x = self.speed - self.sharp
    vel_msg.linear.y = self.speed + self.sharp
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    # print("SHARP RIGHT")
    return vel_msg

  # sets fields of Twist variable to stop robot and puts the robot
  # back to starting position
  def stop(self):
    print("Stop")
    vel_msg = Twist()
    vel_msg.linear.x = 0.0
    vel_msg.linear.y = 0.0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    return vel_msg

    # wait for next image (do not start with state 'lost')
    curr_number_img = self.img_cnt
    while (self.img_cnt <= curr_number_img + 2):
      i = 1

  # publishes stopping message
  def stopRobot(self):
    vel = self.stop()
    self.velocity_publisher.publish(vel)

  # send the ROS message
  def execute_action(self, action):
    # execute action
    vel = Twist()
    directions = {
      0: self.sharp_left,
      1: self.left,
      2: self.slightly_left,
      3: self.forward,
      4: self.slightly_right,
      5: self.right,
      6: self.sharp_right,
      7: self.stop
    }
    function = directions.get(action)
    vel = function()
    self.velocity_publisher.publish(vel)

  ######################################### QUELLE ##########################################
  '''
  https://answers.gazebosim.org//question/18372/getting-model-state-via-rospy/
  '''

  def set_position(self, x, y, z):
    state_msg = ModelState()
    state_msg.model_name = 'three_pi'
    state_msg.pose.position.x = x
    state_msg.pose.position.y = y
    state_msg.pose.position.z = z
    state_msg.pose.orientation.x = 0
    state_msg.pose.orientation.y = 0
    state_msg.pose.orientation.z = 0
    state_msg.pose.orientation.w = 0

    rospy.wait_for_service('/gazebo/set_model_state')
    try:
      set_state = rospy.ServiceProxy('/gazebo/set_model_state',
                                     SetModelState)
      resp = set_state(state_msg)

    except rospy.ServiceException as e:
      print("Service call failed: %s" % e)

  ###########################################################################################

  # At the moment: reset to same starting position
  # choose one of five given positions randomly
  def choose_random_starting_position(self):
    # choose random number between 0 and 1
    rand = random.uniform(0, 1)
    '''
    if(rand <= (1.0/5.0)):
        #initial starting position
        self.x_position = -3.4032014349
        self.y_position = -6.22487658223
        self.z_position = -0.0298790967155
    if (rand > (1.0/5.0) and rand <= (2.0 / 5.0)):
        # straight line (long) going into left curve
        self.x_position = -0.9032014349
        self.y_position = -6.22487658223
        self.z_position = -0.0298790967155
    elif (rand > (2.0 / 5.0) and rand <= (3.0 / 5.0)):
        # sharp left curve
        self.x_position = 0.930205421421
        self.y_position = -5.77364575559
        self.z_position = -0.0301045554742
    elif (rand > (5.0 / 5.0) and rand <= (4.0 / 5.0)):
        # sharp right curve
        self.x_position = 1.1291257432
        self.y_position = -3.37940826549
        self.z_position = -0.0298815752691
    else:
        # straight line going into right curve
        self.x_position = 0.4132014349
        self.y_position = -2.89940826549
        self.z_position = -0.0298790967155
    '''
    # straight line (long)
    self.x_position = -0.9032014349
    self.y_position = -6.22487658223
    self.z_position = -0.0298790967155

  # if user pressed ctrl+c --> stop the robot
  def shutdown(self):
    print("Stopping")
    # publish
    self.vel_msg = self.stop()
    self.velocity_publisher.publish(self.vel_msg)

    # print statistics
    end = time.time()
    total = end - self.start
    minutes = total / 60.0
    speed = self.speed
    distance = speed * total

    total_learning_time = self.learning_time - self.start
    minutes_learning = total_learning_time / 60.0
    print("Learning time = " + str(
      total_learning_time) + " seconds = " + str(minutes_learning)
          + " minutes")
    print("Number Episodes = " + str(self.max_episodes))
    print("Total time = " + str(total) + " seconds = " + str(
      minutes) + " minutes")
    print("Distance = " + str(distance) + " meters")
    print("Speed = " + str(speed) + " m/s)")

    # save q matrix and records for later
    self.bot.save_q_matrix(end, total_learning_time,
                           self.max_episodes, minutes_learning,
                           total, minutes, distance, speed)

  # puts robot back to starting position
  def reset_environment(self):
    self.choose_random_starting_position()
    self.set_position(self.x_position, self.y_position,
                      self.z_position)

  # decide whether to explore or to exploit
  def epsilon_greedy(self, e):
    # random number
    exploration_rate_threshold = random.uniform(0, 1)

    if (exploration_rate_threshold < e):
      # explore
      self.explorationMode = True
      return True
    else:
      # exploit
      self.explorationMode = False
      return False


  # build a network
  def build_network(self, images, size_layer, size_output,
                    learning_rate, sess, tgts):
    dim0 = len(images)
    dim1 = len(images[0])
    dim2 = len(images[0, 0])
    size_input = dim0*dim1*dim2

    # input
    input = tf.compat.v1.placeholder(tf.float64, [dim0, dim1, dim2])
    a0 = tf.reshape(input, (-1, size_input))

    #targets
    targets = tf.compat.v1.placeholder(tf.float64, [1, size_output])

    # layer 1
    W1 = tf.Variable(np.random.uniform(-0.01, 0.01, [size_input,
                                                     size_layer]))
    b1 = tf.Variable(np.random.uniform(-0.01, 0.01, [size_layer]),
                     name="b1")
    a1 = tf.compat.v1.nn.relu_layer(a0, W1, b1, "a1")

    # layer 2
    W2 = tf.Variable(np.random.uniform(-0.01, 0.01, [size_layer,
                                                     size_layer]))
    b2 = tf.Variable(np.random.uniform(-0.01, 0.01, [size_layer]),
                     name="b2")
    a2 = tf.compat.v1.nn.relu_layer(a1, W2, b2, "a2")

    # output
    W3 = tf.Variable(np.random.uniform(-0.01, 0.01, [size_layer,
                                                     size_output]))
    b3 = tf.Variable(np.random.uniform(-0.01, 0.01, [1, size_output]),
                     name="b3")
    a3 = tf.compat.v1.matmul(a2, W3) + b3

    # loss
    # loss = tf.reduce_mean(
      # tf.nn.softmax_cross_entropy_with_logits_v2(targets, a3))
    loss = tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(
      targets, a3))

    # update weights
    sgdObj = tf.compat.v1.train.AdamOptimizer(learning_rate)
    updateOp = sgdObj.minimize(loss)

    # run session (generate ouput from input)
    sess.run(tf.compat.v1.global_variables_initializer())
    output, loss, upOp = sess.run([a3, loss, updateOp], feed_dict={
      input: images, targets: tgts})
    print("\n\nShape output = " +  str(np.shape(output)))
    print("Loss = " + str(loss))
    print("Updated = " + str(upOp))

    # print("Weights = " + str(updatedWeights))
    return output

  # 'one-hot' coding for target values
  def fill_targes(self, action, reward):
    for i in range(len(self.targets)):
      if(i == action):
        self.targets[0, i] = reward
      else:
        self.targets[0, i] = 0
    return self.targets

  # main program
  def reinf_main(self):
    # tell program what to do on shutdown (user presses ctrl+c)
    rospy.on_shutdown(self.shutdown)

    # time statistics
    self.start = time.time()  # starting time
    self.learning_time = 0  # end of learning

    # tensorflow session object
    sess = tf.compat.v1.Session()

    # episodes
    self.explorationMode = False  # exploring or exploiting?
    episode_counter = 0  # number of done episodes
    gamma = 0.95  # learning rate
    alpha = 0.8  # learning rate

    # variables deciding whether to explore or to exploit
    exploration_prob = 0.99
    decay_rate = 0.001
    min_exploration_rate = 0.01
    max_exploration_rate = 1

    # set starting position of the robot
    # not random, even though the name says differently
    self.choose_random_starting_position()
    self.set_position(self.x_position, self.y_position,
                      self.z_position)

    # current state and action are negative before starting
    self.curr_state = -1
    self.curr_action = -1
    self.last_state = -1000

    # set to False if you wish to see the robot driving with a
    # pre-defined matrix
    learn = True

    # number consecutive images
    self.stack_of_images = 1

    # important steps of the algorithm
    try:
      rate = rospy.Rate(20)
      # main loop
      while not rospy.is_shutdown():
        # reinforcement learning
        if (learn):
          # q-learning
          if (episode_counter <= self.max_episodes):
            # wait for next image
            # img = self.get_image()
            img = self.get_stack_of_images()

            # save last state, get new state and calculate reward
            self.last_state = self.curr_state
            print("Last state: " + str(
              self.state_strings.get(self.last_state)))
            self.curr_state = self.bot.get_state(img[
                                                   self.stack_of_images-1])
            print("Current state: " + str(self.state_strings.get(
                self.curr_state)))
            reward = self.bot.calculate_reward(self.curr_state)
            print("Reward: " + str(reward))

            # get target values
            targets = self.fill_targes(self.curr_action, reward)

            # save reward for last state and current
            # action in q-matrix
            # self.bot.update_q_table(self.last_state,
                                  #  self.curr_action,
                                  #  alpha, reward, gamma,
                                  #  self.curr_state)

            # put input in network model
            output = self.build_network(img, 100, len(targets[0]),
                                        0.001, sess, targets)
            print("\n\nOutput of DNN = " + str(output))
            self.q_values = output
            print("Q Values = " + str(self.q_values))

            # begin a new episode if robot lost the line
            if (self.curr_state == self.lost_line):
              # stop robot
              self.stopRobot()
              # set robot back to starting position
              self.reset_environment()
              # episode is done => increase counter
              episode_counter += 1
              print("NEW EPISODE: ", episode_counter)
              # print current q-matrix to see what's
              # going on
              # self.bot.printMatrix(time.time())
              print("-" * 100)
              # skip the next steps and start a new loop
              continue

            # get the next action
            # if exploring: choose random action
            if (self.epsilon_greedy(exploration_prob)):
              print("Exploring")
              action = self.bot.explore(img[self.stack_of_images-1])
              print("Action: " + self.action_strings.get(action))
              self.execute_action(action)
              self.curr_action = action
            # if exploiting: choose best action
            else:
              print("Exploiting")
              # action = self.bot.exploit(img[
              # self.stack_of_images-1], self.curr_state)
              action = np.argmax(self.q_values)
              print("Action: " + self.action_strings.get(action))
              self.execute_action(action)
              self.curr_action = action

            # decay the probability of exploring
            exploration_prob = min_exploration_rate + (
              max_exploration_rate - min_exploration_rate) * \
                np.exp(-decay_rate * episode_counter)

            # end of iteration
            print("-" * 100)

            # start a new loop at the current position
            # (robot will only be set back to starting
            # position if line is lost) 

          # using learned values in matrix to drive
          else:
            # first time in else: save time that robot
            # spent learning
            if (episode_counter == self.max_episodes + 1):
              self.learning_time = time.time()

            print("Driving!")
            # wait for new image
            img = self.get_image()
            # choose best next action out of q-matrix
            action = self.bot.drive(img)
            print("Action: " + self.action_strings.get(
              action))
            # execute action
            self.execute_action(action)
            # stop if line is lost (hopefully never, if
            # robot learned properly)
            if (action == self.stop_action):
              self.reset_environment()

        # drive by a pre-defined matrix
        else:
          # wait for next image
          # img = self.get_image()
          images = self.get_stack_of_images()

          '''
          #wait until image stack is full
          img_arr = self.get_stack_of_images()
          print("Number of images in stack = " + str(len(img_arr)))
          print("Shape of image stack = " + str(np.shape(img_arr)))
          for i in range(len(img_arr)):
            print("Image " + str(i))
            print(img_arr[i])
          '''

          # get next action
          # img = np.zeros(shape=[1, len(images[3])])
          # img[0] = images[3]
          # print("Shape of last image = " + str(np.shape(img)))
          action = self.bot.own_q_matrix(images[3])
          # execute action
          self.execute_action(action)
          # stop if line is lost
          if (action == self.stop_action):
            self.reset_environment()

    except rospy.ROSInterruptException:
      pass


# start node
if __name__ == '__main__':
  node = Node()
  node.reinf_main()

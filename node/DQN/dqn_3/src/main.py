#!/usr/bin/env python

# import own scripts
import matrix as bt
import image as mi
import Memory
import Network

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
    # last image
    img = self.imgHelper.img_conversion(msg)
    self.my_img = np.copy(img)

    # last couple of images (image queue)
    mod = self.img_cnt % self.number_of_images
    self.my_img_queue[mod] = self.my_img

    # count the received images
    self.img_cnt += 1
    print("Image number " + str(self.img_cnt))

  # wait until a new image is received
  def get_image(self):
    nr_images = self.img_cnt
    while (self.img_cnt <= nr_images):
      pass
    return self.my_img

  # wait until image stack is full => not necessary anymore because
  # of image queue?
  def get_stack_of_images(self):
    nr_images = self.img_cnt
    mod_old = -1
    while (self.img_cnt <= (nr_images + self.number_of_images)):
      mod_new = self.img_cnt % self.number_of_images
      if not (mod_new == mod_old):
        self.img_stack[mod_new] = self.my_img
      mod_old = mod_new
    return self.img_stack

  # constructor
  def __init__(self):
    # helper classes
    self.bot = bt.Bot()  # reward + q-matrix
    self.imgHelper = mi.MyImage()  # image processing
    self.memory = Memory.Memory(100) # replay buffer

    # global variables
    # images
    # current image
    self.my_img = np.zeros(50)
    # next couple of (four) consecutive images
    self.number_of_images = 4
    self.img_stack = np.zeros(shape=[self.number_of_images,
                                     len(self.my_img)])
    # number of images received
    self.img_cnt = 0

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

    # initial values
    self.max_episodes = 200
    self.speed = 3.0

    # deviation from speed to turn the robot to the left or to the
    # right
    self.sharp = self.speed * (
      1.0 / 7.0)  # sharp curve => big difference
    self.middle = self.speed * (
      1.0 / 8.5)  # middle curve => middle difference
    self.slightly = self.speed * (
      1.0 / 10.0)  # slight curve => slight difference

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
    self.sess = tf.compat.v1.Session()  # tensorflow session object
    self.batch_size = 4   # batch size for replay buffer
    self.mini_batch_size = 2    # batch size for neural network
    # image queue for the last couple of images
    self.my_img_queue = np.zeros(shape=[self.batch_size,
                                        len(self.my_img)])
    # initialize networks
    # policy network to train on
    self.policy_net = Network.Network(size_layer1=5,
                                      session=self.sess,
                                      batch_size=self.mini_batch_size)
    # target network to calculate optimal q-values on
    self.target_net = Network.Network(size_layer1=5,
                                      session=self.sess,
                                      batch_size=self.mini_batch_size)

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
    # curr_number_img = self.img_cnt
    # while (self.img_cnt <= curr_number_img + 2):
      # i = 1

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
    if (rand > (0.5)):
        # sharp left curve
        self.x_position = 0.930205421421
        self.y_position = -5.77364575559
        self.z_position = -0.0301045554742
    else:
        # sharp right curve
        self.x_position = 1.1291257432
        self.y_position = -3.37940826549
        self.z_position = -0.0298815752691
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
    if (rand > (2.0 / 5.0) and rand <= (3.0 / 5.0)):
        # sharp left curve
        self.x_position = 0.930205421421
        self.y_position = -5.77364575559
        self.z_position = -0.0301045554742
    else (rand > (5.0 / 5.0) and rand <= (4.0 / 5.0)):
        # sharp right curve
        self.x_position = 1.1291257432
        self.y_position = -3.37940826549
        self.z_position = -0.0298815752691
    else:
        # straight line going into right curve
        self.x_position = 0.4132014349
        self.y_position = -2.89940826549
        self.z_position = -0.0298790967155
    # straight line (long)
    self.x_position = -0.9032014349
    self.y_position = -6.22487658223
    self.z_position = -0.0298790967155
    '''


  # if user pressed ctrl+c --> stop the robot
  def shutdown(self):
    print("Stopping")
    # publish
    self.vel_msg = self.stop()
    self.velocity_publisher.publish(self.vel_msg)

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

  # main program
  def reinf_main(self):
    # tell program what to do on shutdown (user presses ctrl+c)
    rospy.on_shutdown(self.shutdown)

    # time statistics
    self.start = time.time()  # starting time
    self.learning_time = 0  # end of learning

    # episodes
    self.explorationMode = False  # exploring or exploiting?
    episode_counter = 0  # number of done episodes
    step_counter = 0  # counts every while iteration
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

    # initialize starting parameters
    # current state and action are negative before starting
    self.curr_state = -1
    self.curr_action = -1
    self.last_state = -1000

    # wait for the first couple of images without doing anything
    # to make sure, that the first image received is a proper one
    used_images = self.get_stack_of_images()
    my_img = self.get_image()


    # important steps of the algorithm
    try:
      rate = rospy.Rate(20)
      # main loop
      while not rospy.is_shutdown():
        # reinforcement learning
        if (episode_counter <= self.max_episodes):
          print("Learning")
          # wait for next image
          # img = self.get_stack_of_images()
          last_img = np.copy(my_img)
          my_img = self.get_image()
          my_img_queue = np.copy(self.my_img_queue)
          # print("Shape of image = " + str(np.shape(img)))

          # save last state
          self.last_state = self.curr_state
          print("Last state: " + str(self.state_strings.get(
            self.last_state)))
          # get new / current state
          self.curr_state = self.bot.get_state(my_img)
          # stop robot immediatley if state is 'line lost', but still
          # do calculations
          if(self.curr_state == self.lost_line):
            self.stopRobot()
          print("Current state: " + str(self.state_strings.get(
             self.curr_state)))
          #calculate reward
          reward = self.bot.calculate_reward(self.curr_state)
          print("Reward: " + str(reward))

          # store experience
          my_img_queue = my_img_queue.flatten()
          self.memory.store_experience(my_img, last_img,
                                       self.curr_action, reward)

          # update target network
          # targets = self.fill_targets(self.last_state)
          # targets = self.fill_targets(self.curr_action, reward)
          if(step_counter % 100 == 1):
            self.target_net = self.policy_net.copy(self.target_net)

          '''
          # update NN if the last state was not 'lost'
          # otherwise it will only learn to stop,
          # because after a stop action, the robot will always
          # be in the middle -> highest reward
          if not (self.last_state == self.lost_line):
            # get random (batch of) experience(s)
            experience = []
            batch_sz = self.number_of_images
            if not (self.curr_action == -1):
              experience = self.memory.get_random_experience(
                batch_size=batch_sz)

            # put input in network model
            # use experiences by a chance of 50% if enough episodes
            # explored
            rand = random.uniform(0, 1)
            if (
              episode_counter >= self.max_episodes / 4 and rand >= 0.5):
              print("Using experience")
              # make a stack of experience images
              exp_img = np.zeros(
                shape=[self.number_of_images, 1, 50])
              for i in range(len(experience)):
                exp_img[i] = experience[i].get("last_state")

              # feed network
              output = self.update_weights(images=exp_img,
                                           learning_rate=0.01,
                                           sess=sess, epochs=1,
                                           tgts=targets)

            # make new experiences
            else:
              output = self.update_weights(images=img,
                                           learning_rate=0.01,
                                           sess=sess, epochs=1,
                                           tgts=targets)

            # print("\n\nOutput of DNN = " + str(output))
            self.q_values = output
            # print("Q Values = " + str(self.q_values))
          '''

          # later:
          # get optimal q-values for current state via the target
          # network
          # targets = self.target_net.use_network(images=last_img)
          # current: keep it simple
          # put reward at targets[1, current action] and 0 else
          targets = np.zeros(shape=[1, 7])
          for i in range(len(targets[0])):
            targets[0, i] = 0
          if not(self.curr_action == -1):
            targets[0, self.curr_action] = reward

          # get self.batch_size number of examples from buffer
          buffer_examples = self.memory.get_random_experience(1)
          # print("Buffer = " + str(buffer_examples))
          # we only have to feed the images representing the
          # starting state into the network, therefore extract
          # those from the buffer_examples dictionary
          buffer_images = buffer_examples[0].get("state")

          # update policy network and get optimal q-values as a result
          # every twentieth step use images from memory
          # else use current images
          if (step_counter % 20 == 1):
            # what should the targets be here??
            # how do I calculate the reward??
            print("Memory")
            output = self.policy_net.update_weights(
            images=buffer_images, epochs=1, targets=targets,
            learning_rate=0.01)
            used_images = buffer_images
          else:
            print("Real life")
            output = self.policy_net.update_weights(
            images=my_img, epochs=1, targets=targets,
            learning_rate=0.01)
            used_images = my_img_queue

          # print("\n\nOutput of DNN = " + str(output))
          self.q_values = output
          # print("Q Values = " + str(self.q_values))

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
            # action = self.bot.explore(img[self.number_of_images - 1])
            action = np.argmax(self.q_values)
            print("Action: " + self.action_strings.get(action))
            self.execute_action(action)
            self.curr_action = action
          # if exploiting: choose best action
          else:
            print("Exploiting")
            # choose best action by using the NN with the current
            # image, not updating it
            self.q_values = self.policy_net.use_network(
              images=my_img, targets=targets)
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

        # using trained network to drive
        else:
          # first time in else: save time that robot
          # spent learning
          if (episode_counter == self.max_episodes + 1):
            self.learning_time = time.time()

          print("Driving!")
          # wait for new image
          my_img = self.get_image()
          # choose best next action from neural network
          # action = self.bot.drive(img)
          # use NN, do NOT update it
          targets = np.zeros(shape=[1, 7])
          for i in range(len(targets[0])):
            targets[0, i] = 0
          if not(self.curr_action == -1):
            targets[0, self.curr_action] = reward

          self.q_values = self.policy_net.use_network(
            images=my_img, targets = targets)

          action = np.argmax(self.q_values)
          print("Action: " + self.action_strings.get(action))
          # execute action
          self.execute_action(action)
          # stop if line is lost (hopefully never, if
          # robot learned properly)
          if (action == self.stop_action):
            self.reset_environment()

        step_counter += 1

    except rospy.ROSInterruptException:
      pass


# start node
if __name__ == '__main__':
  node = Node()
  node.reinf_main()

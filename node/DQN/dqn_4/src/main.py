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

    self.store_img_in_stack(self.my_img)

    print("Image number " + str(self.img_cnt))

  # store image in stack
  def store_img_in_stack(self, img):
    mod1 = self.img_cnt % (self.mini_batch_size *
                           self.images_per_memory)
    row = int(mod1) / int(self.images_per_memory)
    col = (mod1 % self.images_per_memory) * len(self.my_img)
    self.image_stack[row][col] = self.get_image()

  # receive ONE new image
  def get_image(self):
    nr_images = self.img_cnt
    while (self.img_cnt <= nr_images):
      pass
    return self.my_img

  # receive A COUPLE of new images
  def get_stack_of_images(self):
    # wait until callback is done
    self.get_image()
    # return current state of stack
    return self.image_stack

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
    # number of images received in total
    self.img_cnt = 0
    # number of images that will be stored for a single memory sample
    self.images_per_memory = 4

    # neural network
      # hyperparameters
        # number of memory samples that will be processed together in
        # one execution of the neural network
    self.mini_batch_size = 2
        # number of examples that will be extracted at once from
        # the memory
    self.batch_size = 4
      # tensorflow session object
    sess = tf.compat.v1.Session()
      # policy network
    self.policy_net = Network.Network(mini_batch_size=self.mini_batch_size,
                                      images_per_memory=self.images_per_memory,
                                      size_layer1=5, session=sess)

    # stack of the last couple of images
    self.image_stack = np.zeros(shape=[
      self.mini_batch_size, self.images_per_memory*len(
      self.my_img)])


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
        # straight line (long) going into left curve
        self.x_position = -0.9032014349
        self.y_position = -6.22487658223
        self.z_position = -0.0298790967155
    else:
        # straight line going into right curve
        self.x_position = 0.4132014349
        self.y_position = -2.89940826549
        self.z_position = -0.0298790967155
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

  # fill targets
  #   reward where index = action
  #   0 else
  def fill_targets(self, targets, action, reward, sample_no):
    for i in range(len(targets[0])):
      targets[sample_no, i] = 0
    if not (action == -1):
      targets[sample_no, action] = reward
    return targets

  # get random (batch of) experiences and learn with it
  def use_memory(self, targets, stack_of_images):
    memory_batch = self.memory.get_random_experience(
      batch_size=self.batch_size)
    mem_states = np.zeros(shape=[self.batch_size][
      self.images_per_memory*len(self.my_img)])
    mem_actions = np.zeros(self.batch_size)
    mem_rewards = np.zeros(self.batch_size)
    my_targets = targets
    for i in range(len(self.batch_size)):
      mem_states[i] = memory_batch[i].get("state")
      mem_actions[i] = memory_batch[i].get("action")
      mem_rewards[i] = memory_batch[i].get("reward")
      my_targets = self.fill_targets(my_targets, mem_actions[i],
                                  mem_rewards[i], i)
    output = self.policy_net.update_weights(images=stack_of_images,
                                            epochs=1,
                                            targets=my_targets,
                                            learning_rate=0.001)
    return output


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

    # wait for image

    # last image
    last_img = np.zeros(len(self.my_img))
    copy = np.zeros(len(self.my_img))

    # important steps of the algorithm
    try:
      rate = rospy.Rate(20)
      # main loop
      while not rospy.is_shutdown():
        # reinforcement learning
        if (episode_counter <= self.max_episodes):
          print("Learning")
          # wait for next image(s)
          last_img = copy
          my_img_stack = self.get_stack_of_images()
          print("stack of images = \n" +str(my_img_stack))
          my_img = np.zeros(shape=[1, len(self.my_img)])
          all_last_images = my_img_stack[-1]
          index = len(self.my_img)*(self.images_per_memory-1)
          my_img[0] = all_last_images[index:]
          print("last image = \n" +str(my_img))
          copy = np.copy(my_img)
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
          self.memory.store_experience(state=my_img_stack,
                                       last_state=last_img,
                                       action=self.curr_action,
                                       reward=reward)
          # get targets
          targets = np.zeros(shape=[self.batch_size, (len(
            self.action_strings))])
          targets = self.fill_targets(targets, self.curr_action,
                                      reward, sample_no=0)

          # update policy network and get optimal q-values as a result
          # every twentieth step use images from memory
          if(step_counter % 20 == 1):
            print("Memory")
            self.q_values = self.use_memory(targets,
                                            stack_of_images=my_img_stack)
          # else use current images
          else:
            self.q_values = self.policy_net.update_weights(
              images=my_img_stack, epochs=1, targets=targets,
              learning_rate=0.001)

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
            # take random action
            action = np.argmax(self.q_values)
            print("Action: " + self.action_strings.get(action))
            self.execute_action(action)
            self.curr_action = action
          # if exploiting: choose best action
          else:
            print("Exploiting")
            # get q-values by feeding images to the DQN
            # without updating its weights
            self.q_values = self.policy_net.use_network(images=my_img_stack)
            # choose action by selecting highest q -value
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
          # wait for new image(s)
          my_img_stack = self.get_stack_of_images()

          # get q-values by feeding images to the DQN
          # without updating its weights
          self.q_values = self.policy_net.use_network(
            images=my_img_stack)
          # choose action by selecting highest q -value
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

#!/usr/bin/env python

# import own scripts
import matrix as bt
import image as mi
import Memory
import Network
import sound

# import numpy
import numpy as np
from numpy import random

# tensorflow
import numpy as np
import tensorflow as tf

# Import OpenCV libraries and tools
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

# ROS
import rospy
import rospkg
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

# other
import random
#import simpleaudio as sa


class Node:
  '''-------------------------Constructor--------------------------'''
  def __init__(self):
    # helper classes
    self.bot = bt.Bot()  # reward + q-matrix
    self.imgHelper = mi.MyImage()  # image processing
    self.memory = Memory.Memory(1000) # replay buffer

    # hyperparameters to experiment with
    # number of learning episodes
    self.max_episodes = 20
    # speed of the robot's wheels
    self.speed = 7.0
    # number of memory samples that will be processed together in
    # one execution of the neural network
    self.mini_batch_size = 2
    # number of examples that will be extracted at once from
    # the memory
    self.batch_size = 10
    # variables for Bellman equation
    self.gamma = 0.999

    # current image
    self.my_img = np.zeros(50)
    # current multiple images
    self.my_mult_img = np.zeros(shape=[1, self.mini_batch_size*50])
    # last images
    self.last_imgs = np.copy(self.my_mult_img)
    # number of images received in total
    self.img_cnt = 0
    # episodes
    self.explorationMode = False  # exploring or exploiting?
    self.episode_counter = 0  # number of done episodes
    self.step_counter = 0  # counts every while iteration
    # variables deciding whether to explore or to exploit
    self.exploration_prob = 0.99
    self.decay_rate = 0.01
    self.min_exploration_rate = 0.01
    self.max_exploration_rate = 1

    # strings to display actions
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

    # strings to display states
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
    # tensorflow session object
    self.sess = tf.compat.v1.Session()
    # policy network
    self.policy_net = Network.Network(mini_batch_size=self.mini_batch_size,
                                      size_layer1=5,
                                      session=self.sess)
    # target array (for simple way)
    self.targets = np.zeros(shape=[1, (len(self.action_strings) - 1)])
    # target network to calculate 'optimal' q-values
    self.target_net = Network.Network(mini_batch_size=self.mini_batch_size,
                                      size_layer1=5,
                                      session=self.sess)
    # copy weights and layers from the policy net into the target net
    self.target_net = self.policy_net.copy(self.target_net)

    # message to post on topic /cmd_vel
    self.vel_msg = Twist()
    # Bool: exploring (True) or exploiting (False)?
    self.explorationMode = False

    # terminal states
    self.lost_line = 7
    self.stop_action = 7

    # current state and action (initially negative)
    self.curr_state = -1
    self.curr_action = -1
    self.last_state = self.curr_state

    # starting coordinates of the robot
    self.x_position = -0.9032014349
    self.y_position = -6.22487658223
    self.z_position = -0.0298790967155

    # deviation from speed to turn the robot to the left or right
    # sharp curve => big difference
    self.sharp = self.speed * (1.0 / 7.0)
    # middle curve => middle difference
    self.middle = self.speed * (1.0 / 8.5)
    # slight curve => slight difference
    self.slightly = self.speed * (1.0 / 10.0)

    # flag to indicate end of learning
    self.first_time = True

    # ROS variables
    # publisher to publish on topic /cmd_vel
    self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist,
                                              queue_size=100)
    # initializing ROS-node
    rospy.init_node('reinf_matrix_driving', anonymous=True)
    # subscriber for topic '/camera/image_raw'
    self.sub = rospy.Subscriber('/camera/image_raw', Image,
                                self.cam_im_raw_callback)

  '''-----------------------Image methods--------------------------'''
  # callback; copies the received image into a global numpy-array
  def cam_im_raw_callback(self, msg):
    # convert ROS image to cv image, copy it and save it as a global
    # numpy-array
    img = self.imgHelper.img_conversion(msg)
    self.my_img = np.copy(img)

    print("Image number " + str(self.img_cnt))
    self.img_cnt += 1

  # receive ONE new image
  def get_image(self):
    # wait for the next image
    nr_images = self.img_cnt
    while (self.img_cnt <= nr_images):
      pass
    # return current image
    return self.my_img

  # receive multiple images
  def get_multiple_images(self):
    # helping array
    images = np.zeros(shape=[self.mini_batch_size, 50])
    # wait for the next images
    cnt = 0
    while (cnt < self.mini_batch_size):
      images[cnt] = self.get_image()
      cnt += 1
    # flatten helping array
    self.my_mult_img[0] = images.flatten()
    # return current images
    return self.my_mult_img

  # get multiple images for network
  # get single image for state
  def shape_images(self):
    # get multiple images for network
    my_imgs = self.get_multiple_images()
    # get single image to calculate state
    my_img = np.zeros(shape=[1, 50])
    my_img[0] = my_imgs[:, (
      self.mini_batch_size-1)*50:self.mini_batch_size*50]
    return my_imgs, my_img

  '''-----------------------Driving methods------------------------'''
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

  '''----------------------Position methods------------------------'''
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

  # choose one of five given positions randomly
  def choose_random_starting_position(self):
    # sharp right curve
    self.x_position = 1.1291257432
    self.y_position = -3.37940826549
    self.z_position = -0.0298815752691
    '''
    # choose random number between 0 and 1
    rand = random.uniform(0, 1)
    #print("rand = " + str(rand))
    if(rand <= (1.0/5.0)):
      #initial starting position
      self.x_position = -3.4032014349
      self.y_position = -6.22487658223
      self.z_position = -0.0298790967155
      #print("case 0")
    elif (rand > (1.0/5.0) and rand <= (2.0 / 5.0)):
      # straight line (long) going into left curve
      self.x_position = -0.9032014349
      self.y_position = -6.22487658223
      self.z_position = -0.0298790967155
      #print("case 1")
    elif (rand > (2.0 / 5.0) and rand <= (3.0 / 5.0)):
      # sharp left curve
      self.x_position = 0.930205421421
      self.y_position = -5.77364575559
      self.z_position = -0.0301045554742
      #print("case 2")
    elif (rand > (3.0 / 5.0) and rand <= (4.0 / 5.0)):
      # sharp right curve
      self.x_position = 1.1291257432
      self.y_position = -3.37940826549
      self.z_position = -0.0298815752691
      #print("case 3")
    else:
      # straight line going into right curve
      self.x_position = 0.4132014349
      self.y_position = -2.89940826549
      self.z_position = -0.0298790967155
      #print("case 4")
      '''

  '''
  https://answers.ros.org/question/261782/how-to-use-getmodelstate-service-from-gazebo-in-python/
  '''
  def get_position(self):
    model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state',
                                           GetModelState)
    object_coordinates = model_coordinates("three_pi", "")
    x_position = object_coordinates.pose.position.x
    y_position = object_coordinates.pose.position.y
    z_position = object_coordinates.pose.position.z
    return x_position, y_position, z_position

  # puts robot back to starting position
  def reset_environment(self):
    # select position
    self.choose_random_starting_position()

    # set robot to it
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

  '''-----------------------Network methods------------------------'''
  # fill targets
  def fill_targets(self, targets, action, reward):
    # fill with all zeros
    for i in range(len(targets[0])):
      targets[0, i] = 0
    # put reward where index = action, but only if it's not the
    # first iteration (if action is not -1)
    if not (action == -1):
      targets[0, action] = reward
    return targets

  # get random (batch of) experiences and learn with it
  # use 'simple' targets for optimal q-values
  def use_memory(self, targets):
    # get random batch of memories
    memory_batch = self.memory.get_random_experience(
      batch_size=self.batch_size)
    # update network for every memory
    for i in range(len(memory_batch)):
      mem_last_state = memory_batch[i].get("last_state")
      mem_action = memory_batch[i].get("action")
      mem_reward = memory_batch[i].get("reward")
      my_targets = self.fill_targets(targets, mem_action, mem_reward)
      output, loss = self.policy_net.update_weights(
        state=mem_last_state,targets=my_targets)
    print("last output y =\n\t" + str(output))
    print("last loss = " + str(loss))

  # bellman equation for double q network
  def bellman_with_tn(self, targets, reward):
    # qstar(s, a) = Rt+1 + gamma*maxqstar(s', a')
    # max = np.argmax(targets)
    # add reward to highest q-value
    for i in range(len(targets[0])):
      targets[0, i] = (targets[0, i] * self.gamma) + reward
    return targets

  # get random (batch of) experiences and learn with it
  # use target network for optimal q-values
  def use_memory_with_tn(self):
    # get random batch of memories
    memory_batch = self.memory.get_random_experience(
      batch_size=self.batch_size)
    # update network for every memory
    for i in range(len(memory_batch)):
      mem_state = memory_batch[i].get("state")
      mem_last_state = memory_batch[i].get("last_state")
      mem_reward = memory_batch[i].get("reward")
      # compute target q-values with the 'current' state of the memory
      my_targets = self.target_net.use_network(state=mem_state)
      # add reward to highest q-value*gamma to fulfill the bellman
      # equation
      my_targets = self.bellman_with_tn(my_targets, mem_reward)
      # update policy network based on target q-values
      output, loss = self.policy_net.update_weights(
        state=mem_last_state,targets=my_targets)
    print("last output y =\n\t" + str(output))
    print("last loss = " + str(loss))

  '''--------------------Robot process methods---------------------'''
  # get current state and reward
  def get_robot_state(self, my_img):
    # safe last state
    self.last_state = self.curr_state
    print("Last state: " + str(self.state_strings.get(self.last_state)))
    # get new / current state
    self.curr_state = self.bot.get_state(my_img)
    print("Current state: " + str(self.state_strings.get(
      self.curr_state)))

    # stop robot immediatley if state is 'line lost', but still
    # do calculations
    stop_arr=[1, 6, self.lost_line]
    #if (self.curr_state == self.lost_line):
    if(self.curr_state in stop_arr):
      self.stopRobot()

    # calculate reward
    reward = self.bot.calculate_reward(self.curr_state)
    print("Reward: " + str(reward))

    return self.curr_state, reward

  # stop robot, reset environment and increase episode counter
  def begin_new_episode(self):
    # stop robot
    self.stopRobot()
    # set robot back to starting position
    self.reset_environment()
    print("NEW EPISODE: ", self.episode_counter)
    # skip the next image
    self.get_image()

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

  # select and execute action
  def get_next_action(self, my_imgs):
    # if exploring: choose random action
    print("Exploration prob = " + str(self.exploration_prob))
    if (self.epsilon_greedy(self.exploration_prob)):
      print("Exploring")
      # take random action
      action = np.random.randint(low=0, high=7)
      print("Action: " + self.action_strings.get(action))
      self.execute_action(action)
      self.curr_action = action
    # if exploiting: choose best action
    else:
      print("Exploiting")
      # get q-values by feeding images to the DQN
      # without updating its weights
      self.q_values = self.policy_net.use_network(
        state=my_imgs)
      # choose action by selecting highest q -value
      action = np.argmax(self.q_values)
      print("Action: " + self.action_strings.get(action))
      self.execute_action(action)
      self.curr_action = action
    return self.curr_action

  '''------------------------Ending methods------------------------'''
  # publishes stopping message
  def stopRobot(self):
    vel = self.stop()
    self.velocity_publisher.publish(vel)

  # if user pressed ctrl+c --> stop the robot
  def shutdown(self):
    print("Stopping")
    # publish
    self.vel_msg = self.stop()
    self.velocity_publisher.publish(self.vel_msg)

  '''--------------------Drive without learning--------------------'''
  # use trained network to drive
  def drive(self):
    # make sound to signal that learning is finished
    if(self.first_time):
      sound.make_sound()
      self.first_time = False

    # wait for new image(s)
    my_imgs, my_img = self.shape_images()
    # get new / current state
    self.curr_state = self.bot.get_state(my_img)

    # get q-values by feeding images to the DQN
    # without updating its weights
    self.q_values = self.policy_net.use_network(
      state=my_imgs)
    # choose action by selecting highest q -value
    action = np.argmax(self.q_values)
    print("Action: " + self.action_strings.get(action))

    # execute action
    self.execute_action(action)
    # stop if line is lost (hopefully never, if
    # robot learned properly)
    if (self.curr_state == self.lost_line):
      self.reset_environment()

  '''------------------------Main Program--------------------------'''
  # main program
  def reinf_main(self):
    # tell program what to do on shutdown (user presses ctrl+c)
    rospy.on_shutdown(self.shutdown)

    # set starting position of the robot
    self.reset_environment()

    # wait for first images before starting
    my_img = self.get_image()

    # main loop
    try:
      while not rospy.is_shutdown():
        # reinforcement learning
        if (self.episode_counter <= self.max_episodes):
          print("-" * 100)
          print("Learning")
          # wait for next image(s)
          my_imgs, my_img = self.shape_images()

          # get state and reward
          self.curr_state, reward = self.get_robot_state(my_img)

          # store experience
          self.memory.store_experience(state=my_imgs,
                                       last_state=self.last_imgs,
                                       action=self.curr_action,
                                       reward=reward)

          '''
          # update target net every 20th step
          if(self.step_counter % 20 == 0):
            self.target_net = self.policy_net.copy(self.target_net)
            print("Updated target network")
          '''

          # softly update target net EVERY step
          self.target_net = self.policy_net.copy_softly(self.target_net)

          # get targets simple way
          '''
          self.targets = self.fill_targets(self.targets,
                                           self.curr_action, reward)
          '''

          # get targets via target_network
          # self.targets = self.target_net.use_network(
            # state=self.last_imgs)

          # use states from memory to update policy network in
          # order to not 'forget' previously learned things
          # self.use_memory(self.targets)
          self.use_memory_with_tn()

          # begin a new episode if robot lost the line
          if (self.curr_state == self.lost_line):
            self.begin_new_episode()
            # episode is done => increase counter
            self.episode_counter += 1
            # skip the next steps and start a new loop
            continue

          #print("begin driving")
          # get the next action
          self.curr_action = self.get_next_action(my_imgs)
          # wait for some images
          #for i in range(2):
           # self.get_image()
          # stop robot in order to not miss any images
          #self.stopRobot()
          #print("end driving")

          # learn a second time
          # self.use_memory(self.targets)
          # self.use_memory_with_tn()

          # decay the probability of exploring
          self.exploration_prob = self.min_exploration_rate + (
            self.max_exploration_rate - self.min_exploration_rate) * \
              np.exp(-self.decay_rate * self.episode_counter)

          # end of iteration
          print("-" * 100)

          # start a new loop at the current position
          # (robot will only be set back to starting
          # position if line is lost)

        # using trained network to drive
        else:
          print("Driving!")
          self.drive()

        # count taken steps (while cycles)
        self.step_counter += 1

    except rospy.ROSInterruptException:
      pass

# start node
if __name__ == '__main__':
  node = Node()
  node.reinf_main()

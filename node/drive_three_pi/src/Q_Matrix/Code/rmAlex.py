#!/usr/bin/env python

# import own scripts
import Bot_4 as bt
import MyImage_4 as mi

# import numpy
import numpy as np
from numpy import random

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
        print("Neues Bild", self.img_cnt)

        # convert ROS image to cv image, copy it and save it as a global numpy-array
        img = self.imgHelper.img_conversion(msg)
        self.my_img = np.copy(img)
        # im_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        self.cnt_all_img += 1
        #cv.imwrite('Bilder/img_' + str(self.cnt_all_img) + '.jpg', self.my_img);

        # count the received images
        self.img_cnt += 1
        #print("Image counter = " + str(self.img_cnt))

    def receiveImage(self):
      imgCountSave = self.img_cnt ;
      while(imgCountSave == self.img_cnt):
        pass ;
      print ("got img", self.img_cnt)
      return self.my_img ;


    # constructor
    def __init__(self):
        # helper classes
        #self.bot = bt.Bot()

        self.sensoryState = -1 ;
        self.motorState = -1 ;

        # global variables
        self.my_img = []
        self.curve = "start"
        self.vel_msg = Twist()
        self.flag = False
        self.second_image = False
        self.start = time.time()
        self.img_cnt = 0
        self.cnt_all_img = 0

        # terminal states
        self.lost_line = 5 ;
        self.terminalStates = [self.lost_line]

        # starting coordinates of the robot
        self.x_position, self.y_position, self.z_position = self.get_start_position()
        # self.save_position()

        # inital values
        self.speed = 10.0

        # deviation from speed so average speed stays the same
        self.sharp = self.speed * (1.0 / 8.5)  # sharp curve => big difference
        self.middle = self.speed * (1.0 / 9.25)  # middle curve => middle difference
        self.slightly = self.speed * (1.0 / 10.0)  # slight curve => slight difference


        # helper classes
        self.imgHelper = mi.MyImage() ;
        self.actions = [0,1,2,3,4,5,6] ;
        self.sensoryStates = [0,1,2,3,4,self.lost_line] ;
        self.expl = [""]
        self.actionMethods = [self.sharp_left, self.left, self.slightly_left,
          self.forward, self.slightly_right, self.right, self.sharp_right] ;
        self.directions = { action : proc for action,proc in zip(self.actions, self.actionMethods) } ;

        self.nrActions = len(self.actions)
        self.nrSensoryStates = len(self.actions)
        self.Q = np.zeros([self.nrActions, self.nrSensoryStates]) ;

        # publisher to publish on topic /cmd_vel
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=100)

        # Add here the name of the ROS. In ROS, names are unique named.
        rospy.init_node('reinf_matrix_driving', anonymous=True)
        # subscribe to a topic using rospy.Subscriber class
        self.sub = rospy.Subscriber('/camera/image_raw', Image, self.cam_im_raw_callback)

    # sets fields of Twist variable so robot drives sharp left
    def sharp_left(self):
        vel_msg = Twist()
        vel_msg.linear.x = self.speed + self.sharp
        vel_msg.linear.y = self.speed - self.sharp
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        # print("LEFT")
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
        # print("LEFT")
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
        # print("RIGHT")
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
        # print("RIGHT")
        return vel_msg

    # sets fields of Twist variable to stop robot and puts the robot back to starting position
    def stop(self):
        # self.episodes_counter += 1
        # self.choose_random_starting_position()
        # self.set_position(self.x_position, self.y_position, self.z_position)

        print("Stop")
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        # print("RIGHT")
        return vel_msg

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
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)

        except rospy.ServiceException, e:
            print
            "Service call failed: %s" % e
    ###########################################################################################

    # Check where robot is at the start of the simulation
    def get_start_position(self):
        model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        object_coordinates = model_coordinates("three_pi", "")
        x_position = object_coordinates.pose.position.x
        y_position = object_coordinates.pose.position.y
        z_position = object_coordinates.pose.position.z
        # print("x = " + str(x_position))
        # print("y = " + str(y_position))
        # print("z = " + str(z_position))
        return x_position, y_position, z_position

    # Save a position in a .txt-file
    def save_position(self):
        try:
            # open correct file
            f = open("/home/elisabeth/catkin_ws/src/drive_three_pi/src/Q_Matrix/Code/Learn_Simple_3/position.txt", "a")
            # f = open("../Q_Matrix/Q-Matrix-Records.txt", "a")

            # pretty print matrix
            end = time.time()
            readable_time = time.ctime(end)
            string = str(readable_time)
            string += ("\n[x=" + str(self.x_position))
            string += (", y=" + str(self.y_position))
            string += (", z=" + str(self.z_position) + "]\n\n")

            # write into file
            f.write(string)

            # close file
            f.close()
        except Exception as e:
            print(str(e) + "\nFile not written")

    # choose one of five given positions randomly
    def choose_random_starting_position(self):

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

        end = time.time()
        total = end - self.start
        minutes = total / 60.0
        speed = self.speed
        distance = speed * total
        print("Total time = " + str(total) + " seconds = " + str(minutes) + " minutes")
        print("Distance = " + str(distance) + " meters")
        print("Speed = " + str(speed) + " m/s)")

        # save q matrix and records for later
        self.save_q_matrix()

    #pretty print matrix and return as a nice string
    def print_matrix(self):
        end = time.time()
        readable_time = time.ctime(end)
        string = "\n\n" + str(readable_time) + ")\n"
        string += "\nALEX\n["
        # col_max = np.argmax(self.Q, axis=0)
        for i in range(len(self.Q)):
            string += " ["
            for j in range(len(self.Q[i])):
                '''
                if (i == col_max[j]):
                    number = np.round(self.Q[i, j], 3)
                    string += " **{:04.3f}**, ".format(number)
                '''
                number = np.round(self.Q[i, j], 3)
                string += "  {:04.3f}  , ".format(number)
            string += "]\n"
        string += "]"

        print(string)
        return string

    # save q-matrix as a .txt-file
    def save_q_matrix(self):
        try:
            # open correct file
            f = open(
                "/home/elisabeth/catkin_ws/src/rl_matrix/src/Q_Matrix/Code/Learn_Simple_3/Q-Matrix-Records.txt",
                "a")

            #get nice matrix string
            string = self.print_matrix()

            # write into file
            f.write(string)

            # close file
            f.close()
        except Exception as e:
            print(str(e) + "\nFile not written")

    # puts robot back to starting position
    def reset_environment(self):
        self.choose_random_starting_position()
        self.set_position(self.x_position, self.y_position, self.z_position)
        #print(self.x_position, self.y_position)
        # time.sleep(0.25)

    # decide whether to explore or to exploit
    def exploration(self):
      # random number
      rd = random.uniform(0, 1)

      if (rd < self.explorationProb):
        # explore
        self.explorationMode = True ;
        # self.explorationProb -= self.decay_rate
        return True
      else:
        # exploit
        self.explorationMode = False;
        return False

    def inExplorationMode(self):
      return self.explorationMode ;

    def getRandomAction(self):
      index = random.randint(0,len(self.actions)-1) ;
      print(index,"INDEX");
      return self.actions[index] ;

    # send the ROS message. Action is an int with the action code 0-7
    def setMotorState(self, action):
      # execute action
      vel = self.directions.get(action)() ;
      self.motorState = action ;
      # print("Vel msg =\n" + str(vel))
      # publish
      self.velocity_publisher.publish(vel)

    def stopRobot(self):
      vel = self.stop() ;
      self.velocity_publisher.publish(vel)

    def saveLastSensoryState(self):
      self.lastSensoryState = self.sensoryState ;

    def computeSensoryState(self, img):
      #print("img size", img.shape) ;
      line = img[0,:].astype("int32") ; # convert3gray
      lineMask = line > 100 ;
      lineIndices = np.where(lineMask)[0]
      if len(lineIndices) > 0:
        leftBorder = lineIndices[0] ;
        #print(leftBorder, "leftB")
        diff2Middle = img.shape[1]/2 - leftBorder ;
        #print(diff2Middle)

        diff2Middle = diff2Middle // 10 ;
        if diff2Middle < -2 or diff2Middle > 2:
          return self.lost_line ;
        return diff2Middle + 2 ;
      else:
        return self.lost_line ;

    def computeReward(self, img):
      if self.sensoryState == self.lost_line:
        return -1000 ;
      return -math.fabs(self.sensoryState -2) ;


    # Bellman equation
    def updateQ(self, reward):
      self.Q[self.motorState, self.lastSensoryState] *= (1.-self.alpha) ;
      update = reward + self.gamma * self.Q[:,self.sensoryState].max() ;
      self.Q[self.motorState, self.lastSensoryState] += self.alpha * update ;

    def getBestAction(self):
      return self.Q[:,self.sensoryState].argmax() ;

    def isTerminalState(self):
      return self.sensoryState in self.terminalStates ;


    # main program
    def reinf_main(self):
      rospy.on_shutdown(self.shutdown)
      self.start = time.time()

      episode_counter = 0
      self.gamma = 0.95
      self.alpha = 0.8

      self.explorationProb= 0.99 ;
      self.explorationMode = False ;
      decay_rate = 0.001
      min_exploration_rate = 0.01
      max_exploration_rate = 1

      # start at defined point
      self.reset_environment();

      self.motorState = -1 ;
      self.sensoryState = -1 ;
      self.lastSensoryState = -1000 ;

      max_episodes = 1000

      try:
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
          if(episode_counter <= max_episodes):
              img = self.receiveImage() ;
              self.saveLastSensoryState() ;
              self.sensoryState = self.computeSensoryState(img) ;
              curReward = self.computeReward(img);
              print("SensState",self.sensoryState) ;
              print("MotorState",self.motorState) ;

              # update Q matrix with sensory/motor state from last loop
              if self.inExplorationMode() == True:
                self.updateQ(curReward) ;

              # if terminal state (lost line) is reached, get out of for loop
              # stop robot and put it back to starting position
              if self.isTerminalState():
                print ("NEW EPISODE!", episode_counter+1);
                self.stopRobot() ;
                self.reset_environment()
                episode_counter += 1
                self.print_matrix()
                continue ;

              if self.exploration():
                action = self.getRandomAction() ;
                self.setMotorState(action) ;
              else:
                action = self.getBestAction() ;
                self.setMotorState(action) ;
              print("new motor state =", self.motorState)
              print("-"*30)
              # ------------------
              #rate.sleep()
          else:
              print("Driving!")
              img = self.receiveImage();
              self.saveLastSensoryState();
              self.sensoryState = self.computeSensoryState(img);
              curReward = self.computeReward(img);
              print("SensState", self.sensoryState);
              print("MotorState", self.motorState);

              # if terminal state (lost line) is reached, get out of for loop
              # stop robot and put it back to starting position
              if self.isTerminalState():
                  #print("NEW EPISODE!", episode_counter + 1);
                  self.stopRobot();
                  self.reset_environment()
                  #episode_counter += 1
                  continue;

              action = self.getBestAction();
              self.setMotorState(action);
              print("new motor state =", self.motorState)
              print("-" * 30)

      except rospy.ROSInterruptException:
          pass


if __name__ == '__main__':
    # try:
    node = Node()
    # node.main()
    node.reinf_main()
    # except Exception:
    # print("EXC")
    # pass

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
        # print("Neues Bild")

        # convert ROS image to cv image, copy it and save it as a global numpy-array
        img = self.imgHelper.img_conversion(msg)
        self.my_img = np.copy(img)
        # im_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        self.cnt_all_img += 1
        #cv.imwrite('Bilder/img_' + str(self.cnt_all_img) + '.jpg', self.my_img);

        # count the received images
        self.img_cnt += 1
        #print("Image counter = " + str(self.img_cnt))

    def get_image(self):
        nr_images = self.img_cnt
        while(self.img_cnt <= nr_images):
            pass
        return self.my_img

    # constructor
    def __init__(self):
        # helper classes
        self.bot = bt.Bot()

        # global variables
        self.my_img = []
        self.curve = "start"
        self.vel_msg = Twist()
        self.flag = False
        self.second_image = False
        self.start = time.time()
        self.img_cnt = 0
        self.cnt_all_img = 0
        self.explorationMode = False

        # terminal states
        self.lost_line = 7
        self.stop_action = 7

        # current state and action
        self.curr_state = -1
        self.curr_action = -1

        #last state
        self.last_state = -1000

        # starting coordinates of the robot
        self.x_position, self.y_position, self.z_position = self.get_start_position()
        # self.save_position()

        # inital values
        self.speed = 15.0

        # deviation from speed so average speed stays the same
        self.sharp = self.speed * (1.0 / 8.5)  # sharp curve => big difference
        self.middle = self.speed * (1.0 / 9.25)  # middle curve => middle difference
        self.slightly = self.speed * (1.0 / 10.0)  # slight curve => slight difference

        '''
        Did work with 'old' reward function:
        #inital values
        self.speed = 20.0

        #deviation from speed so average speed stays the same
        self.sharp = self.speed * (1.0/7.0)         #sharp curve => big difference
        self.middle = self.speed * (1.0/8.5)        #middle curve => middle difference
        self.slightly = self.speed * (1.0/10.0)     #slight curve => slight difference
        '''

        # helper classes
        self.imgHelper = mi.MyImage()

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

        #wait for next image (do not start with state 'lost')
        curr_number_img = self.img_cnt
        while (self.img_cnt <= curr_number_img + 2):
            i = 1

    def stopRobot(self):
        vel = self.stop()
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
        # choose random number between 0 and 1
        rand = random.uniform(0, 1)

        '''
        if(rand <= (1.0/5.0)):
            #initial starting position
            self.x_position = -3.4032014349
            self.y_position = -6.22487658223
            self.z_position = -0.0298790967155
        if (rand > (1.0/5.0) and rand <= (2.0 / 5.0)):
            # straight line going into left curve
            self.x_position = 0.4132014349
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

        end = time.time()
        total = end - self.start
        minutes = total / 60.0
        speed = self.speed
        distance = speed * total
        print("Total time = " + str(total) + " seconds = " + str(minutes) + " minutes")
        print("Distance = " + str(distance) + " meters")
        print("Speed = " + str(speed) + " m/s)")

        # save q matrix and records for later
        self.bot.save_q_matrix(self.start, speed, distance)

    # puts robot back to starting position
    def reset_environment(self):
        self.choose_random_starting_position()
        self.set_position(self.x_position, self.y_position, self.z_position)
        #print(self.x_position, self.y_position)
        # time.sleep(0.25)

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
            return False
            self.explorationMode = False

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
        # print("Vel msg =\n" + str(vel))
        # publish
        self.velocity_publisher.publish(vel)

    # main program
    def reinf_main(self):
        rospy.on_shutdown(self.shutdown)
        self.start = time.time()

        # episodes = 2000
        max_episodes = 500
        self.explorationMode = False
        episode_counter = 0
        gamma = 0.95
        alpha = 0.8

        exploration_prob = 0.99
        # decay_rate = 0.001
        # min_exploration_rate = 0.01
        # max_exploration_rate = 1

        all_rewards = []

        # start at random point
        self.choose_random_starting_position()
        self.set_position(self.x_position, self.y_position, self.z_position)

        self.curr_state = -1
        self.curr_action = -1
        self.last_state = -1000

        try:
            rate = rospy.Rate(20)
            while not rospy.is_shutdown():
                if(episode_counter <= max_episodes):
                    img = self.get_image()
                    self.last_state = self.curr_state
                    print("Last state: " + str(self.state_strings.get(self.last_state)))
                    self.curr_state = self.bot.get_state(img)
                    print("Current state: " + str(self.state_strings.get(self.curr_state)))
                    reward = self.bot.calculate_reward(self.curr_state)
                    print("Reward: " + str(reward))

                    if(self.explorationMode == True):
                        self.bot.update_q_table(self.last_state, self.curr_action, alpha, reward, gamma, self.curr_state)

                    if(self.curr_state == self.lost_line):
                        self.stopRobot()
                        self.reset_environment()
                        episode_counter += 1
                        print("NEW EPISODE: ", episode_counter)
                        print("-" * 60)
                        continue

                    if(self.epsilon_greedy(exploration_prob)):
                        print("Exploring")
                        action = self.bot.explore(img)
                        print("Action: " + self.action_strings.get(action))
                        self.execute_action(action)
                        self.curr_action = action
                    else:
                        print("Exploiting")
                        action = self.bot.exploit(img, self.curr_state)
                        print("Action: " + self.action_strings.get(action))
                        self.execute_action(action)
                        self.curr_action = action

                    print("-" * 60)

                else:
                    print("Driving!")
                    img = self.get_image()
                    action = self.bot.drive(img)
                    print("Action: " + self.action_strings.get(action))
                    self.execute_action(action)
                    if(action == self.stop_action):
                        self.reset_environment()

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

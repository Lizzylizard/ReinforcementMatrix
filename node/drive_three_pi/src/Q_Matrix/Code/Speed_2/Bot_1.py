#!/usr/bin/env python

#import own scripts
import node_1 as rm
import myImage_1 as mi

#import numpy
import numpy as np
from numpy import random

# Import OpenCV libraries and tools
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

#ROS
import rospy
import rospkg 
from std_msgs.msg import String, Float32, Int32
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

#other
import math 
import time

class Bot:
    def __init__(self):
        #start ROS node 
        self.node = rm.Node()
        #image helper
        self.img_helper = mi.MyImage()
        
        #class variables
        self.learning_done = False 
        
        #speed 
        self.left_wheel = 0.0
        self.right_wheel = 0.0
        self.max_speed = 30.0
        self.speed_change = 5           #step size to change speed 
        
        #Hyperparameters
        self.epi_cnt = 0
        self.episodes = 400
        self.sequence = 20
        self.gamma = 0.95
        self.alpha = 0.8
        
        self.states = np.zeros(shape=[26])
        for i in range(len(self.states)):
            self.states[i] = i 
        '''        
        0: line missing (speed does not matter) -> terminal state 
        
        line far left 
        1: speed = 0, 2: speed < 25%, 3: speed < 50%, 4 = speed < 75%, 5 = speed < 100%

        line left
        6: speed = 0, 7: speed < 25%, 8: speed < 50%, 9 = speed < 75%, 10 = speed < 100%
        
        usw.        
        '''
        
        self.actions = [0, 1, 2, 3, 4, 5, 6]
        '''
        0 = stop, 
        1 = left wheel faster, 2 = left wheel slower, 
        3 = right wheel faster, 4 = right wheel slower, 
        5 = bother faster, 6 = both slower 
        '''
        
        self.Q = np.zeros(shape=[len(self.states), len(self.actions)])
        
    def get_curr_speed(self):
        curr_speed = (self.left_wheel + self.right_wheel) / 2.0
        print("Current speed = " + str(curr_speed))
        return curr_speed
        
    def get_curr_state(self, img, speed):
        left = self.img_helper.count_pxl(img)
        
        #width of image 
        width = np.size(img[0])
        
        #string for debugging
        line_string = "Line state = " 
        
        #return value 
        state = 0
        
        if(left >= width):
            #line is missing -> terminal state 
            string += "missing"
            state = 0
        elif(left > 0 and left <= width/5):
            #line is far left 
            string += "far left"
            if(speed == 0):
                #robot stopped
                state = 1
            elif(speed > 0 and speed <= self.max_speed * (25/100)):
                #speed slower than 25% of max speed 
                state = 2
            elif(speed > self.max_speed * (25/100) and speed <= self.max_speed * (50/100)):
                #speed slower than 50% of max speed 
                state = 3
            elif(speed > self.max_speed * (50/100) and speed <= self.max_speed * (75/100)):
                #speed slower than 75% of max speed 
                state = 4
            elif(speed > self.max_speed * (75/100)):
                #speed slower than 100% of max speed 
                state = 5
                
        elif(left > width/5 and left <= width * (2/5)):
            #line is left 
            string += "left"
            if(speed == 0):
                #robot stopped
                state = 6
            elif(speed > 0 and speed <= self.max_speed * (25/100)):
                #speed slower than 25% of max speed 
                state = 7
            elif(speed > self.max_speed * (25/100) and speed <= self.max_speed * (50/100)):
                #speed slower than 50% of max speed 
                state = 8
            elif(speed > self.max_speed * (50/100) and speed <= self.max_speed * (75/100)):
                #speed slower than 75% of max speed 
                state = 9
            elif(speed > self.max_speed * (75/100)):
                #speed slower than 100% of max speed 
                state = 10
                
        elif(left > width * (2/5) and left <= width * (3/5)):
            #line is in the middle
            string += "middle"
            if(speed == 0):
                #robot stopped
                state = 11
            elif(speed > 0 and speed <= self.max_speed * (25/100)):
                #speed slower than 25% of max speed 
                state = 12
            elif(speed > self.max_speed * (25/100) and speed <= self.max_speed * (50/100)):
                #speed slower than 50% of max speed 
                state = 13
            elif(speed > self.max_speed * (50/100) and speed <= self.max_speed * (75/100)):
                #speed slower than 75% of max speed 
                state = 14
            elif(speed > self.max_speed * (75/100)):
                #speed slower than 100% of max speed 
                state = 15
                
        elif(left > width * (3/5) and width * (4/5)):
            #line is right
            string += "right"
            if(speed == 0):
                #robot stopped
                state = 16
            elif(speed > 0 and speed <= self.max_speed * (25/100)):
                #speed slower than 25% of max speed 
                state = 17
            elif(speed > self.max_speed * (25/100) and speed <= self.max_speed * (50/100)):
                #speed slower than 50% of max speed 
                state = 18
            elif(speed > self.max_speed * (50/100) and speed <= self.max_speed * (75/100)):
                #speed slower than 75% of max speed 
                state = 19
            elif(speed > self.max_speed * (75/100)):
                #speed slower than 100% of max speed 
                state = 20 
        else:
            #line is far right
            string += "far right"
            if(speed == 0):
                #robot stopped
                state = 21
            elif(speed > 0 and speed <= self.max_speed * (25/100)):
                #speed slower than 25% of max speed 
                state = 22
            elif(speed > self.max_speed * (25/100) and speed <= self.max_speed * (50/100)):
                #speed slower than 50% of max speed 
                state = 23
            elif(speed > self.max_speed * (50/100) and speed <= self.max_speed * (75/100)):
                #speed slower than 75% of max speed 
                state = 24
            elif(speed > self.max_speed * (75/100)):
                #speed slower than 100% of max speed 
                state = 25 
                
        print(line_string)
        return state 
        
    #epsilon greedy algorithm
    #decide whether to explore or to exploit
    def explore_vs_exploit(self):        
        #return value 
        explore = True 
        
        #have at least explored for 10% of all episodes 
        if(self.epi_cnt < float(self.episodes) * (float(10) / float(100))):
            explore = True 
        else :
            #rate completed empisodes / all episodes 
            rel = float(self.epi_cnt) / float(self.episodes)
            
            #random float between 0 and 1
            x = random.rand()
            
            #decrease epsilon over time 
            epsilon = 1 - rel 
            
            #do not explore if random float is greater than epsilon 
            if(x > epsilon):
                explore = False
            else:
                explore = True 
        
        print("Exploring = " + str(explore))
        
        return explore
        
    def calc_reward(self, curr_state):
        ######QUELLE######
        ''' 
        ../ml-se/processing/unity_simulation_scene/scripts/ml/matrix/MQControl.py
        '''
        
        # weight the factors
        lineFactor = 2
        speedFactor = 1
        punishment = 50
        
        #line reward 
        line_reward = 0
        if(curr_state >= 11 and curr_state <= 15):
            #best case -> line is in the middle 
            line_reward += 10
        elif((curr_state >= 6 and curr_state <=10) or (curr_state >= 16 and curr_state <=20)):
            #second best case -> line is left or right
            line_reward += 5
        elif((curr_state >= 1 and curr_state <=5) or (curr_state >= 21 and curr_state <=25)):
            #bad case -> line is far left or far right
            line_reward += 0
        else:
            #worst case -> line is missing -> terminal state -> punish 
            line_reward -= punishment 
            
        #speed reward 
        speed_reward = 0
        if(curr_state == 0):
            #lost line is already punsihed
            pass 
        else:
            speed_index = (curr_state-1) % 5 
            if(speed_index == 0):
                #worst case -> robot stopped 
                speed_reward -= punishment 
            elif(speed_index == 1):
                #bad case -> robot drives very slowly (<25%) 
                speed_reward += 0
            elif(speed_index == 2):
                #bad case -> robot drives slowly (<50%) 
                speed_reward += (10 * (1.0/3.0))
            elif(speed_index == 3):
                #good case -> robot drives fast (<75%) 
                speed_reward += (10 * (2.0/3.0))
            elif(speed_index == 4):
                #best case -> robot drives very fast(<100%) 
                speed_reward += 10
                
        whole_reward = (lineFactor * line_reward) + (speedFactor * speed_reward)
        print("Reward = " + str(whole_reward))
        return whole_reward
                
    
    def reinf_learning(self): 
        #get image 
        img = self.node.my_img
        
        #get speed 
        curr_speed = self.get_curr_speed()
        
        #get current state 
        curr_state = self.get_curr_state(img, curr_speed)
        
        explore = self.explore_vs_exploit()
        
        for i in range(self.sequence):
            if(explore):
                #random action
                next_action_arr = np.random.choice(self.actions, 1)
                next_action = next_action_arr[0]
                
            else:
                #use q-table
                next_action = np.argmax(self.Q[curr_state])
                
            #calculate reward -> first time: robot sits in the middle of the line -> gets a reward even though it did not do anything
            reward = self.calc_reward(curr_state, curr_speed)
                
            self.Q[curr_state, next_action] = (1-alpha) * self.Q[curr_state, next_action] + alpha * (reward + gamma * np.max(self.Q[next_action, :]))
            
        return next_action 
        
        
    def execute_action(self, next_action):
        if(next_action == 0):
            #stop
            print("Next action = stop")
            self.left_wheel = 0
            self.right_wheel = 0
        elif(next_action == 1):
            #left wheel faster 
            print("Next action = left wheel faster")
            self.left_wheel += self.speed_change
        elif(next_action == 2):
            #left wheel slower
            print("Next action = left wheel slower")
            self.left_wheel -= self.speed_change
        elif(next_action == 3):
            #right wheel faster  
            print("Next action = right wheel faster")
            self.right_wheel += self.speed_change
        elif(next_action == 4):
            #right wheel slower 
            print("Next action = right wheel slower")
            self.right_wheel -= self.speed_change
        elif(next_action == 5):
            #both wheels faster
            print("Next action = both wheels faster")
            self.left_wheel += self.speed_change
            self.right_wheel += self.speed_change 
        else:
            #both wheels slower 
            print("Next action = both wheels slower")
            self.left_wheel -= self.speed_change
            self.right_wheel -= self.speed_change  
            
        msg = Twist()
        msg.linear.x = self.right_wheel 
        msg.linear.y = self.left_wheel 
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        
        self.node.publish(msg)
        
      
    def drive(self):
        #use q-table
        next_action = np.argmax(self.Q[curr_state])
        return next_action 
    
    def main(self):
        rospy.on_shutdown(self.node.shutdown) 
        try:
            rate = rospy.Rate(50)
            while not rospy.is_shutdown(): 
                if(self.node.flag):
                    #only do something when new picture is ready
                    if (self.epi_cnt <= self.episodes):
                        #learn 
                        print("Learning")
                        next_action = self.reinf_learning()
                        self.execute_action(next_action)
                        self.epi_cnt += 1
                    else:
                        #drive
                        print("Driving")
                        next_action = self.drive()
                        self.execute_action(next_action)
                        
                    
                
            rate.sleep()
        except rospy.ROSInterruptException:
            pass
        

if __name__ == '__main__':
    bot = Bot()
    bot.main()
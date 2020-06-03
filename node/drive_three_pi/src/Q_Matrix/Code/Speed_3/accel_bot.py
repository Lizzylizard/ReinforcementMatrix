#!/usr/bin/env python

#import own scripts
import accel_reinf_matrix as rm
import accel_myImage as mi

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
    possible_moves = {}
    possible_states = {}
    R = [[]]

    def __init__(self, speed):          
        #helper classes 
        self.myImage = mi.MyImage()

        #initial state is quite in the middle of the line (hopefully)
        self.state = 12
                
        #actions the bot can take
        self.possible_moves = [0, 1, 2, 3, 4, 5, 6]
        '''
        0   =   left wheel faster
        1   =   left wheel slower
        2   =   right wheel faster 
        3   =   right wheel slower 
        4   =   both wheels faster 
        5   =   both wheels slower 
        6   =   stop
        '''
        
        #states the bot can be in 
        self.possible_states = np.arange(26)
        '''
        line is far left 
        0: speed = 0, 1: speed < 25%, 2: speed < 50%, 3: speed < 75%, 4: speed < 100%
        
        line is left 
        5: speed = 0, 6: speed < 25%, 7: speed < 50%, 8: speed < 75%, 9: speed < 100%
        
        line is in the middle
        usw.  
        .
        .
        .
        .        
        25: line is lost 
        '''
        
        #q-matrix (empty in the beginning)
        self.Q = np.zeros(shape=[len(self.possible_states), len(self.possible_moves)])

    #check where the line is --> check current state of the bot 
    def set_state(self, img, speed, max_speed): 
        self.state = self.myImage.state(img, speed, max_speed)
        
    #calculate the reward for a given state 
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
        if(curr_state >= 10 and curr_state <= 14):
            #best case -> line is in the middle 
            line_reward += 10
        elif((curr_state >= 5 and curr_state <=9) or (curr_state >= 15 and curr_state <=19)):
            #second best case -> line is left or right
            line_reward += 5
        elif((curr_state >= 0 and curr_state <= 4) or (curr_state >= 20 and curr_state <=24)):
            #bad case -> line is far left or far right
            line_reward += 0
        else:
            #worst case -> line is missing -> terminal state -> punish 
            line_reward -= punishment 
            
        #speed reward 
        speed_reward = 0
        if(curr_state == 25):
            #lost line is already punsihed
            pass 
        else:
            speed_index = curr_state % 5 
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
        
    
        
    #fill q-matrix
    def q_learning(self, gamma, alpha, img, explore, speed, max_speed):            
        #set current state 
        self.set_state(img, speed, max_speed) 
        
        if(self.state == 25):       #line is missing -> terminal state 
            return 25         
        
        #exploration
        if(explore):                            
            #random action
            next_state_arr = np.random.choice(self.possible_moves, 1)
            next_state = next_state_arr[0]
            
        #exploitation (decide using q-matrix with already explored rewards)
        else:
            #get reward using q-matrix 
            next_state = np.argmax(self.Q[self.state])        
            
        #reward for next state
        reward = self.calc_reward(next_state)
        
        #update q-matrix 
        self.Q[self.state, next_state] = (1-alpha) * self.Q[self.state, next_state] + alpha * (reward + gamma * np.max(self.Q[next_state, :]))
               
        return next_state
        
    def use_q_matrix(self, img, speed, max_speed):            
        #set current state 
        self.set_state(img, speed, max_speed)
        
        #decide action using q-matrix 
        action = np.argmax(self.Q[self.state])
        
        return action
        
    #print q-matrix into a .txt-file 
    def save_q_matrix(self, start, speed, distance):
        try:
            #open correct file 
            f = open("/home/elisabeth/catkin_ws/src/drive_three_pi/src/Q_Matrix/Q-Matrix-Records.txt", "a")
            #f = open("../Q_Matrix/Q-Matrix-Records.txt", "a")
            
            #pretty print matrix 
            end = time.time() 
            readable_time = time.ctime(end)
            string = "\n\n" + str(readable_time) + ")\n["
            for i in range(len(self.Q)):
                string += " ["
                for j in range (len(self.Q[i])):
                    number = np.round(self.Q[i], 3)
                    string += " {:04.3f}, ".format(number[j])
                string += "]\n"
            string += "]"
            
            #pretty print results
            total = end - start
            minutes = total / 60.0 
            string += "\nAverage speed = " 
            string += str(speed)
            string += "m/s\nSeconds = " 
            string += str(total)
            string += "\nMinutes = " 
            string += str(minutes)            
            string += "\nDistance = " 
            string += str(distance)
            string += "m"
            
            #write into file 
            f.write(string)  
            
            #close file 
            f.close() 
        except Exception as e:
            print(str(e) + "\nFile not written")
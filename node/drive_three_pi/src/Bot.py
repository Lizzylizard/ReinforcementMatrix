#!/usr/bin/env python

#import own scripts
import reinf_matrix as rm
import MyImage as mi

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

    def __init__(self):          
        #helper classes 
        self.myImage = mi.MyImage()

        #initial state is quite in the middle of the line (hopefully)
        self.state = 2
        
        #actions the bot can take
        self.possible_moves = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        '''
        0   =   sharp left
        1   =   left
        2   =   slightly left
        3   =   slightly right
        4   =   right
        5   =   sharp right
        6   =   stop
        7   =   accelerate
        8   =   brake
        '''
        
        
        #actions the bot can take
        self.possible_moves = [0, 1, 2, 3, 4, 5, 6]
        
        '''
        0   =   sharp left
        1   =   left
        2   =   slightly left
        3   =   slightly right
        4   =   right
        5   =   sharp right
        6   =   stop
        '''
        
        #translation to work with older scripts 
        self.curve = {
            0: "sharp left",
            1: "left",
            2: "slightly left",
            3: "slightly right",
            4: "right",
            5: "sharp right",
            6: "stop"
        }
        
        #states the bot can be in 
        self.possible_states = [0, 1, 2, 3, 4, 5, 6]
        '''
        0   =   > 0% && <=  10%
        1   =   >10% && <=  25%
        2   =   >25% && <=  50%
        3   =   >50% && <=  75%
        4   =   >75% && <=  90%
        5   =   >90%
        6   =   else (stop)
        '''
        
        '''
        #reward-matrix
        self.R = np.zeros(shape=[len(self.possible_states), len(self.possible_moves)])
        #line is far left --> go sharp left (reward), go left (none), go right (punsihment)
        self.R[0, :] = [ 1,  0,  0, -1, -1, -1,   -1, -1,  1]      
        self.R[1, :] = [ 0,  1,  0, -1, -1, -1,   -1, -1,  0]
        self.R[2, :] = [ 0,  0,  1, -1, -1, -1,   -1,  1, -1]
        self.R[3, :] = [-1, -1, -1,  1,  0,  0,   -1,  1, -1]
        self.R[4, :] = [-1, -1, -1,  0,  1,  0,   -1, -1,  0]
        self.R[5, :] = [-1, -1, -1,  0,  0,  1,   -1, -1,  1]
        #only stop if really necessary (no line)
        self.R[6, :] = [-1, -1, -1, -1, -1, -1,  0.5, -1, -1]  
        #acceleration
        self.R[7, :] = [ 1,  1,  1,  1,  1,  1,    1,  1,  1] 
        '''
    
        
        #reward-matrix
        self.R = np.zeros(shape=[len(self.possible_states), len(self.possible_moves)])
        self.R[0, :] = [ 1,  0,  0, -1, -1, -1,   -1]      #line is far left --> go sharp left (reward), go left (none), go right (punsihment)
        self.R[1, :] = [ 0,  1,  0, -1, -1, -1,   -1]
        self.R[2, :] = [ 0,  0,  1, -1, -1, -1,   -1]
        self.R[3, :] = [-1, -1, -1,  1,  0,  0,   -1]
        self.R[4, :] = [-1, -1, -1,  0,  1,  0,   -1]
        self.R[5, :] = [-1, -1, -1,  0,  0,  1,   -1]
        self.R[6, :] = [-1, -1, -1, -1, -1, -1,  0.5]      #only stop if really necessary (no line)
        
        
        #q-matrix (empty in the beginning)
        self.Q = np.zeros(shape=[len(self.possible_states), len(self.possible_moves)])
    
    #check where the line is --> check current state of the bot 
    def set_state(self, img):        
        self.state = self.myImage.state(img)
        
    #fill q-matrix
    def q_learning(self, gamma, alpha, img, explore):            
        #set current state 
        self.set_state(img)
            
        #all possible rewards for current state 
        rewards = self.R[self.state]
        
        #exploration
        if(explore):                            
            #random action
            next_state_arr = np.random.choice(self.possible_moves, 1)
            next_state = next_state_arr[0]
            curve = self.curve.get(next_state)
            
        #exploitation (decide using q-matrix with already explored rewards)
        else:
            #get reward using q-matrix 
            next_state = np.argmax(self.Q[self.state])
            
            #translate into a string
            curve = self.curve.get(next_state)
        
        #update q-matrix 
        '''
        print("R Shape = " + str(np.shape(self.R)))
        print("self.state = " + str(self.state))
        print("next_state = " + str(next_state))
        print("Q Shape = " + str(np.shape(self.Q)))
        '''
        self.Q[self.state, next_state] = (1-alpha) * self.Q[self.state, next_state] + alpha * (self.R[self.state, next_state] + gamma * np.max(self.Q[next_state, :]))

        return curve 
        
    def use_q_matrix(self, img):            
        #set current state 
        self.set_state(img)
        
        #decide action using q-matrix 
        action = np.argmax(self.Q[self.state])
        
        #translate into a string
        curve = self.curve.get(action)
        
        return curve
        
    #print q-matrix into a .txt-file 
    def save_q_matrix(self, start):
        try:
            #open correct file 
            f = open("/home/elisabeth/catkin_ws/src/drive_three_pi/src/Q_Matrix/Q-Matrix-Records.txt", "a")
            #f = open("../Q_Matrix/Q-Matrix-Records.txt", "a")
            
            #pretty print matrix 
            string = "\n\n["
            for i in range(len(self.Q)):
                string += "\n["
                for j in range (len(self.Q[i])):
                    number = np.round(self.Q[i], 3)
                    string += " {:4.3f}, ".format(number[j])
                string += "]"
            string += "]"
            
            #pretty print results
            end = time.time() 
            total = end - start
            minutes = total / 60.0 
            string += "\nSeconds = " 
            string += str(total)
            string += "\nMinutes = " 
            string += str(minutes)
            
            #write into file 
            f.write(string)  
            
            #close file 
            f.close() 
        except Exception as e:
            print(str(e) + "\nFile not written")
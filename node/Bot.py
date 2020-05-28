#!/usr/bin/env python

#import own scripts
import reinf_matrix as rm
import MyImage as mi

#import numpy
import numpy as np

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
    
        #reward-matrix
        self.R = np.zeros(shape=[len(self.possible_moves), len(self.possible_states)])
        self.R[0, :] = [ 1,  0,  0, -1, -1, -1,   -1]      #line is far left --> go sharp left (reward), go left (none), go right (punsihment)
        self.R[1, :] = [ 0,  1,  0, -1, -1, -1,   -1]
        self.R[2, :] = [ 0,  0,  1, -1, -1, -1,   -1]
        self.R[3, :] = [-1, -1, -1,  1,  0,  0,   -1]
        self.R[4, :] = [-1, -1, -1,  0,  1,  0,   -1]
        self.R[5, :] = [-1, -1, -1,  0,  0,  1,   -1]
        self.R[6, :] = [-1, -1, -1, -1, -1, -1,  0.5]      #only stop if really necessary (no line)
        
        #q-matrix (empty in the beginning)
        self.Q = np.zeros(shape=[len(self.possible_moves), len(self.possible_states)])
    
    #check where the line is --> check current state of the bot 
    def set_state(self, img):        
        self.state = self.myImage.state(img)
        
    #fill q-matrix
    def q_learning(self, gamma, alpha, img):
        print("Learning")
        #set current state 
        self.set_state(img)
            
        #all possible rewards for current state 
        rewards = self.R[self.state]
                        
        #random action
        next_state = np.random.choice(self.possible_moves, 1)
        curve = self.curve.get(next_state[0])
        
        #update q-matrix 
        self.Q[self.state, next_state] = (1-alpha) * self.Q[self.state, next_state] + alpha * (self.R[self.state, next_state] + gamma * np.max(self.Q[next_state, :]))
        
        if self.R[self.state, next_state] == 0.5:
            curve = "stop"
            
        return curve 
        
    def use_q_matrix(self, img):
        print("Driving")
        #set current state 
        self.set_state(img)
        
        #decide action using q-matrix 
        arr_max = np.amax(self.Q[self.state])
        action = np.where(self.Q == arr_max)
        
        #translate into a string
        curve = self.curve.get(action[0][0])
        
        return curve
        
#if __name__=='__main__':
    #node = rm.Node()
    
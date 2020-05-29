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
        self.state = 2
        
        #initial speed
        self.speed = speed 
        #last speed 
        self.last_speed = speed 
        
        #actions the bot can take
        self.possible_moves = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        '''
        0   =   sharp left
        1   =   left
        2   =   slightly left
        3   =   slightly right
        4   =   right
        5   =   sharp right
        6   =   stop
        
        7   =   sharp left accelerate
        8   =   left accelerate
        9   =   slightly left accelerate
        10  =   slightly right accelerate
        11  =   right accelerate
        12  =   sharp right accelerate
        13  =   stop accelerate
        
        14  =   sharp left brake
        15  =   left brake 
        16  =   slightly left brake 
        17  =   slightly right brake
        18  =   right brake 
        19  =   sharp right brake 
        20  =   stop brake 
        '''
        
        #translation to work with older scripts 
        self.curve = {
            0: "sharp left",
            1: "left",
            2: "slightly left",
            3: "slightly right",
            4: "right",
            5: "sharp right",
            6: "stop",
            7: "accelerate",
            8: "brake"
        }
        
        #states the bot can be in 
        self.possible_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        '''
        0   =   > 0% && <=  10%
        1   =   >10% && <=  25%
        2   =   >25% && <=  50%
        3   =   >50% && <=  75%
        4   =   >75% && <=  90%
        5   =   >90%
        6   =   else (stop)
        
        accelerating and 
        7   =   > 0% && <=  10%
        8   =   >10% && <=  25%
        9   =   >25% && <=  50%
        10  =   >50% && <=  75%
        11  =   >75% && <=  90%
        12  =   >90%
        13  =   else (stop)
        
        braking and 
        14  =   > 0% && <=  10%
        15  =   >10% && <=  25%
        16  =   >25% && <=  50%
        17  =   >50% && <=  75%
        18  =   >75% && <=  90%
        19  =   >90%
        20  =   else (stop)
        '''
        
        
        #reward-matrix
        self.R = np.zeros(shape=[len(self.possible_states), len(self.possible_moves)])
        #no changed velocity
        #line is far left --> go sharp  left (reward), go   left (none), go right (punsihment) 
        #               same velocity                       accelerate                          brake 
        self.R[0, :] = [ 1,  0,  0, -1, -1, -1,   -1,       -1,  -1,  -1, -1, -1, -1,  -1,      1,   0,   0, -1, -1, -1, -1]      
        self.R[1, :] = [ 0,  1,  0, -1, -1, -1,   -1,        0,   0,   0, -1, -1, -1,  -1,      0,   0,   0, -1, -1, -1, -1]
        self.R[2, :] = [ 0,  0,  1, -1, -1, -1,   -1,        1,   0,   0, -1, -1, -1,  -1,      1,   0,   0, -1, -1, -1, -1]
        self.R[3, :] = [-1, -1, -1,  1,  0,  0,   -1,        1,   0,   0, -1, -1, -1,  -1,      1,   0,   0, -1, -1, -1, -1]
        self.R[4, :] = [-1, -1, -1,  0,  1,  0,   -1,        0,   0,   0, -1, -1, -1,  -1,      0,   0,   0, -1, -1, -1, -1]
        self.R[5, :] = [-1, -1, -1,  0,  0,  1,   -1,       -1,  -1,  -1, -1, -1, -1,  -1,      1,   0,   0, -1, -1, -1, -1]
        #only stop if really necessary (no line)
        self.R[6, :] = [-1, -1, -1, -1, -1, -1,  0.5,       -1,  -1,  -1, -1, -1, -1,  -1,     -1,  -1,  -1, -1, -1, -1, -1]  
        
        #changed velocity to faster(but does not matter, rewards are given in the same way
        #               same velocity                       accelerate                          brake 
        self.R[7, :] = [ 1,  0,  0, -1, -1, -1,   -1,       -1,  -1,  -1, -1, -1, -1,  -1,      1,   0,   0, -1, -1, -1, -1]      
        self.R[8, :] = [ 0,  1,  0, -1, -1, -1,   -1,        0,   0,   0, -1, -1, -1,  -1,      0,   0,   0, -1, -1, -1, -1]
        self.R[9, :] = [ 0,  0,  1, -1, -1, -1,   -1,        1,   0,   0, -1, -1, -1,  -1,      1,   0,   0, -1, -1, -1, -1]
        self.R[10, :] = [-1, -1, -1,  1,  0,  0,   -1,        1,   0,   0, -1, -1, -1,  -1,      1,   0,   0, -1, -1, -1, -1]
        self.R[11, :] = [-1, -1, -1,  0,  1,  0,   -1,        0,   0,   0, -1, -1, -1,  -1,      0,   0,   0, -1, -1, -1, -1]
        self.R[12, :] = [-1, -1, -1,  0,  0,  1,   -1,       -1,  -1,  -1, -1, -1, -1,  -1,      1,   0,   0, -1, -1, -1, -1]
        #only stop if really necessary (no line)
        self.R[13, :] = [-1, -1, -1, -1, -1, -1,  0.5,       -1,  -1,  -1, -1, -1, -1,  -1,     -1,  -1,  -1, -1, -1, -1, -1]
        
        #changed velocity to slower(but does not matter, rewards are given in the same way
        #               same velocity                       accelerate                          brake 
        self.R[14, :] = [ 1,  0,  0, -1, -1, -1,   -1,       -1,  -1,  -1, -1, -1, -1,  -1,      1,   0,   0, -1, -1, -1, -1]      
        self.R[15, :] = [ 0,  1,  0, -1, -1, -1,   -1,        0,   0,   0, -1, -1, -1,  -1,      0,   0,   0, -1, -1, -1, -1]
        self.R[16, :] = [ 0,  0,  1, -1, -1, -1,   -1,        1,   0,   0, -1, -1, -1,  -1,      1,   0,   0, -1, -1, -1, -1]
        self.R[17, :] = [-1, -1, -1,  1,  0,  0,   -1,        1,   0,   0, -1, -1, -1,  -1,      1,   0,   0, -1, -1, -1, -1]
        self.R[18, :] = [-1, -1, -1,  0,  1,  0,   -1,        0,   0,   0, -1, -1, -1,  -1,      0,   0,   0, -1, -1, -1, -1]
        self.R[19, :] = [-1, -1, -1,  0,  0,  1,   -1,       -1,  -1,  -1, -1, -1, -1,  -1,      1,   0,   0, -1, -1, -1, -1]
        #only stop if really necessary (no line)
        self.R[20, :] = [-1, -1, -1, -1, -1, -1,  0.5,       -1,  -1,  -1, -1, -1, -1,  -1,     -1,  -1,  -1, -1, -1, -1, -1]
        
        
        
        #q-matrix (empty in the beginning)
        self.Q = np.zeros(shape=[len(self.possible_states), len(self.possible_moves)])
    
    #check where the line is --> check current state of the bot 
    def set_state(self, img, speed): 
        self.speed = speed 
        self.state = self.myImage.state(img, speed, self.last_speed)
        self.last_speed = speed 
        
    #fill q-matrix
    def q_learning(self, gamma, alpha, img, explore, speed):            
        #set current state 
        self.set_state(img, speed)
        
        #abortion criterion
        if(self.state == 6 or self.state == 13 or self.state == 20):
            return "stop"               ################## do I have to punish here???
            
        #all possible rewards for current state 
        rewards = self.R[self.state]
        
        #exploration
        if(explore):                            
            #random action
            next_state_arr = np.random.choice(self.possible_moves, 1)
            next_state = next_state_arr[0]
            if(next_state > 6 and next_state < 14):
                curve = self.curve.get(7)
            elif(next_state > 13):
                curve = self.curve.get(8)
            else:
                curve = self.curve.get(next_state)
            
        #exploitation (decide using q-matrix with already explored rewards)
        else:
            #get reward using q-matrix 
            next_state = np.argmax(self.Q[self.state])
            
            #translate into a string
            if(next_state > 6 and next_state < 14):
                curve = self.curve.get(7)
            elif(next_state > 13):
                curve = self.curve.get(8)
            else:
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
        
    def use_q_matrix(self, img, speed):            
        #set current state 
        self.set_state(img, speed)
        
        #decide action using q-matrix 
        action = np.argmax(self.Q[self.state])
        
        #translate into a string
        if(next_state > 6 and next_state < 14):
            curve = self.curve.get(7)
        elif(next_state > 13):
            curve = self.curve.get(8)
        else:
            curve = self.curve.get(next_state)
        
        return curve
        
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
                    string += " {:4.3f}, ".format(number[j])
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
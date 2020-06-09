#!/usr/bin/env python

#import own scripts
import reinf_matrix_4 as rm
import MyImage_4 as mi

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
        #action space
        self.actions = np.arange(7)
        '''
        0 = sharp left, 1 = left, 2 = slightly left, 3 = forward, 
        4 = slightly right, 5 = right, 6 = sharp right, (not in array: 7 = stop)
        '''
        self.stop_action = 7
        
        #state space 
        self.states = np.arange(8)
        '''
        0 = line is far left, 1 = line is left, 2 = line is slightly left, 3 = line is in the middle, 
        4 = line is slightly right, 5 = line is right, 6 = line is far right, 7 = line is lost 
        '''
        self.lost_line = 7
        
        #q-matrix (empty in the beginning)
        self.Q = np.zeros(shape=[len(self.states), len(self.actions)])
        
        #image helper 
        self.img_helper = mi.MyImage()

        # returns the reward for a taken action

    def calculate_reward(self, curr_state, last_action):
        # return value
        reward = 0

        if (curr_state == 3):
            # best case: middle
            reward = 10
        elif (curr_state == 2):
            # second best case: slightly left
            reward = 5
        elif (curr_state == 1):
            # third best case: left
            reward = 2
        elif (curr_state == 0):
            # fourth best case: far left
            reward = 1
        elif (curr_state == 4):
            # second best case: slightly right
            reward = 5
        elif (curr_state == 5):
            # third best case: right
            reward = 2
        elif (curr_state == 6):
            # fourth best case: far right
            reward = 1
        else:
            # worst case: line is lost
            reward = (-50)

        return reward

        #returns the reward for a taken action

    def calculate_reward_old(self, curr_state, last_action):
        #return value 
        reward = 0
        
        if(curr_state == 3):
            #best case: middle
            if(last_action == 3):
            #best action is to move forward
                reward = 10
            else:
                reward = -10
        elif(curr_state == 2):
            #second best case: slightly left
            if(last_action == 1):
                reward = 10
            elif(last_action == 0 or last_action == 2):
                reward = 0
            else:
                reward = -10
        elif(curr_state == 1):
            #third best case: left
            if(last_action == 1):
                reward = 10
            elif(last_action == 0 or last_action == 2):
                reward = 0
            else:
                reward = -10
        elif(curr_state == 0):
            #fourth best case: far left
            if(last_action == 0):
                reward = 10
            elif(last_action == 1 or last_action == 2):
                reward = 0
            else:
                reward = -10
        elif(curr_state == 4):
            #second best case: slightly right
            if(last_action == 4):
                reward = 10
            elif(last_action == 5 or last_action == 6):
                reward = 0
            else:
                reward = -10
        elif(curr_state == 5):
            #third best case: right
            if(last_action == 5):
                reward = 10
            elif(last_action == 4 or last_action == 6):
                reward = 0
            else:
                reward = -10
        elif(curr_state == 6):
            #fourth best case: far right
            if(last_action == 6):
                reward = 10
            elif(last_action == 4 or last_action == 5):
                reward = 0
            else:
                reward = -10
        else:
            #worst case: line is lost 
            reward = (-50)
        
        return reward 
    
    #check where the line is --> check current state of the bot 

    def get_state(self, img):
        line_state = self.img_helper.get_line_state(img)
        #for later: check speed here and update state
        return line_state 
        
    #explore by chosing a random action

    def explore(self, img):    
        #chose a random action
        action_arr =  np.random.choice(self.actions, 1)
        action = action_arr[0]        
        return action
        
    #use values already in q-matrix, but still update it 
    def exploit(self, img, state):
        action = np.argmax(self.Q[state,:]) 
        return action
        
    #fill q-matrix 
    def update_q_table(self, curr_state, action, alpha, reward, gamma, next_state):
        #update q-matrix 
        self.Q[curr_state, action] = (1-alpha) * self.Q[curr_state, action] + \
            alpha * (reward + gamma * np.max(self.Q[next_state, :]))
        self.printMatrix(self.Q)
      
    #use filled q-matrix to simply drive 
    def drive(self, img):
        state = self.get_state(img)
        action = np.argmax(self.Q[state,:])
        if(state == self.lost_line):
            #stop robot if line is lost
            action = self.stop_action
        return action 
        
    #save q-matrix as a .txt-file
    def save_q_matrix(self, start, speed, distance):
        try:
            #open correct file 
            f = open("/home/elisabeth/catkin_ws/src/drive_three_pi/src/Q_Matrix/Code/Learn_Simple_3/Q-Matrix-Records.txt", "a")
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

    # pretty print matrix
    def printMatrix(self, M):
        string = "Current matrix = \n"
        for i in range(len(self.Q)):
            for j in range(len(self.Q[i])):
                row_max = np.argmax(self.Q[i, :])
                if(j == row_max):
                    number = np.round(self.Q[i], 3)
                    string += " *{:04.3f}*, ".format(number[j])

                else:
                    number = np.round(self.Q[i], 3)
                    string += " {:04.3f}, ".format(number[j])
            string += "\n"
        print(string + "\n")

    #use pre defined q matrix to drive, to see whether driving works or not
    def own_q_matrix(self, img):
        q = np.zeros(shape = [len(self.states), len(self.actions)])
        q[0] = [1, 0, 0, 0, 0, 0, 0]    #line = far left, action = sharp left
        q[1] = [0, 1, 0, 0, 0, 0, 0]    #line = left, action = left
        q[2] = [0, 0, 1, 0, 0, 0, 0]    #line = slightly left, action = slightly left
        q[3] = [0, 0, 0, 1, 0, 0, 0]    #line = middle, action = forward
        q[4] = [0, 0, 0, 0, 1, 0, 0]    #line = slightly right, action = slightly right
        q[5] = [0, 0, 0, 0, 0, 1, 0]    #line = right, action = right
        q[6] = [0, 0, 0, 0, 0, 0, 1]    #line = far right, action = sharp right
        q[7] = [0, 0, 0, 0, 0, 0, 0]    #line = lost, action = stop

        state = self.get_state(img)
        action = np.argmax(q[state, :])
        if(state == self.lost_line):
            action = self.stop_action
        return action


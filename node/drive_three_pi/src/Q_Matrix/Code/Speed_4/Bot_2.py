#!/usr/bin/env python

#import own scripts
import reinf_matrix_2 as rm
import MyImage_2 as mi

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
        self.actions = np.arange(8)
        '''
        0 = sharp left,     1 = left,   2 = slightly left,  3 = forward, 
        4 = slightly right, 5 = right,  6 = sharp right,    7 = slower 
        '''
        self.stop_action = 8
        
        #state space 
        self.states = np.arange(30)
        '''
        0 = line is far left and speed < 25%
        1 = line is far left and speed >= 25% and speed < 50%, 
        2 = line is far left and speed >= 50% and speed < 75%,
        3 = line is far left and speed >= 75% and speed <= 100%,
        
        4 = line is left and speed < 25%,
        5 = line is left and speed >= 25% and speed < 50%,
        6 = line is left and speed >= 50% and speed < 75%,
        7 = line is left and speed >= 75% and speed <= 100%,
        
        8 = line is slightly left and speed < 25%,
        9 = line is slightly left and speed >= 25% and speed < 50%,
        10 = line is slightly left and speed >= 50% and speed < 75%,
        11 = line is slightly left and speed >= 75% and speed <= 100%,
        
        12 = line is in the middle and speed < 25%,
        13 = line is in the middle and speed >= 25% and speed < 50%,
        14 = line is in the middle and speed >= 50% and speed < 75%,
        15 = line is in the middle and speed >= 75% and speed <= 100%, 
        
        16 = line is slightly right and speed < 25%,
        17 = line is slightly right and speed >= 25% and speed < 50%,
        18 = line is slightly right and speed >= 50% and speed < 75%,
        19 = line is slightly right and speed >= 75% and speed <= 100%,
        
        20 = line is right and speed < 25%,
        21 = line is right and speed >= 25% and speed < 50%,
        22 = line is right and speed >= 50% and speed < 75%,
        23 = line is right and speed >= 75% and speed <= 100%,
        
        24 = line is far right and speed < 25%,
        25 = line is far right and speed >= 25% and speed < 50%,
        26 = line is far right and speed >= 50% and speed < 75%,
        27 = line is far right and speed >= 75% and speed <= 100%,
        
        28 = line is lost 
        29 = speed is 0
        '''
        #status of lost line
        self.lost_line = 28

        #q-matrix (empty in the beginning)
        self.Q = np.zeros(shape=[len(self.states), len(self.actions)])
        
        #image helper 
        self.img_helper = mi.MyImage()
    
    #returns the reward for a taken action 
    def calculate_reward(self, last_state, last_action):
        #easier states
        speed_state = last_state % 4
        line_state = int(last_state) / int(4)

        #weight for speed and line 
        weight_line = 2
        weight_speed = 1
        
        #rewards for speed and line 
        line_reward = 0
        speed_reward = 0
        
        #return value 
        reward = 0
        
        #calculation
        speed_reward = 0
        if(line_state == 0 or line_state == 6):
            #line was far left or far right => very bad
            line_reward = 10
            if(line_state == 0):
                if(last_action == 0):
                    line_reward += 10
                if(last_action == 7):
                    speed_reward += 10
                else:
                    line_reward -= 10
            else:
                if (last_action == 6):
                    line_reward += 10
                if (last_action == 7):
                    speed_reward += 10
                else:
                    line_reward -= 10
            if(speed_state == 0):
                #robot is slowest => very good
                speed_reward += 0
            elif(speed_state == 1):
                #robot is slow => good
                speed_reward += 0
            elif(speed_state == 2):
                #robot is fast => bad
                speed_reward += -10
            else:
                #robot is very fast => very bad
                speed_reward += -10

        elif (line_state == 1 or line_state == 5):
            # line was left or right => bad
            line_reward = 10
            if(line_state == 1):
                if(last_action == 1):
                    line_reward += 10
                else:
                    line_reward -= 10
            else:
                if (last_action == 5):
                    line_reward += 10
                else:
                    line_reward -= 10
            if (speed_state == 0):
                # robot is slowest => good
                speed_reward += 0
            elif (speed_state == 1):
                # robot is slow => good
                speed_reward += 0
            elif (speed_state == 2):
                # robot is fast => bad
                speed_reward += -10
            else:
                # robot is very fast => bad
                speed_reward += -10

        elif (line_state == 2 or line_state == 4):
            # line was slightly left or slightly right => good
            line_reward = 10
            if(line_state == 2):
                if(last_action == 2):
                    line_reward += 10
                if(last_action == 7):
                    speed_reward -= 10
                else:
                    line_reward -= 10
            else:
                if (last_action == 4):
                    line_reward += 10
                if (last_action == 7):
                    speed_reward -= 10
                else:
                    line_reward -= 10
            if (speed_state == 0):
                # robot is slowest => bad
                speed_reward += -10
            elif (speed_state == 1):
                # robot is slow => good
                speed_reward += 0
            elif (speed_state == 2):
                # robot is fast => good
                speed_reward += 10
            else:
                # robot is very fast => bad
                speed_reward += 0

        elif (line_state == 3):
            # line was in the middle => very good
            line_reward = 10
            if(last_action == 3):
                line_reward += 10
            elif(last_action == 7):
                speed_reward -= 20
            else:
                line_reward -= 10
            if (speed_state == 0):
                # robot is slowest => very bad
                speed_reward += -10
            elif (speed_state == 1):
                # robot is slow => bad
                speed_reward += -10
            elif (speed_state == 2):
                # robot is fast => good
                speed_reward += 10
            else:
                # robot is very fast => very good
                speed_reward = 10
        else:
            #line is lost => very, very bad
            line_reward = -100

        if(last_state == 29):
            #speed is zero => very, very bad
            speed_reward = -100
        
        reward = (weight_line * line_reward) + (weight_speed * speed_reward)
        return reward 
    
    #check where the line is --> check current state of the bot 

    def get_state(self, img, all_speeds, max_speed, min_speed):
        #get line state
        line_state = self.img_helper.get_line_state(img)

        if not (line_state == 7):
            #get speed state
            speed_state = 0
            curr_speed = all_speeds[len(all_speeds)-1] - min_speed
            speed_range = max_speed - min_speed
            #print("Speed minus min speed = " + str(curr_speed))
            #print("Speed range = " + str(speed_range))
            percent_of_range = (float(curr_speed) / float(speed_range)) * 100.0
            #print("% of range = " + str(percent_of_range))
            if(curr_speed <= min_speed):
                state = 29
            else:
                if(percent_of_range < 25):
                    speed_state = 0
                elif(percent_of_range >= 25 and percent_of_range < 50):
                    speed_state = 1
                elif(percent_of_range >= 50 and percent_of_range < 75):
                    speed_state = 2
                else:
                    speed_state = 3

            number_of_speed_states = 4
            state = (line_state * number_of_speed_states) + speed_state
        else:
            state = 28

        return state
        
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
    def update_q_table(self, state, action, alpha, reward, gamma, next_state):        
        #update q-matrix 
        self.Q[state, action] = (1-alpha) * self.Q[state, action] + \
            alpha * (reward + gamma * np.max(self.Q[next_state, :]))
      
    #use filled q-matrix to simply drive 
    def drive(self, img, all_speeds, max_speed, min_speed):
        state = self.get_state(img, all_speeds, max_speed, min_speed)
        action = np.argmax(self.Q[state,:])
        if(state == self.lost_line):
            #stop robot if line is lost
            action = self.stop_action
        return action 
        
    #print q-matrix into a .txt-file 

    def save_q_matrix(self, start, speed, distance):
        try:
            #open correct file 
            f = open("/home/elisabeth/catkin_ws/src/drive_three_pi/src/Q_Matrix/Code/Speed_4/Q-Matrix-Records.txt", "a")
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
#!/usr/bin/env python

#import own scripts
import accel_bot as bt
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
        
class Node:
    #callback; copies the received image into a global numpy-array
    def cam_im_raw_callback(self, msg):     
        #rospy.loginfo(msg.header)  

        #convert ROS image to cv image, copy it and save it as a global numpy-array
        img = self.imgHelper.img_conversion(msg) 
        self.my_img = np.copy(img)    
           
        #set flag to true, so main-loop knows, there's a new image to work with
        self.flag = True
        
    #constructor
    def __init__(self):            
        #global variables 
        self.my_img = []   
        self.curve =  "start"  
        self.vel_msg = Twist()
        self.flag = False
        self.start = time.time() 
        self.episodes_counter = 0
        
        #starting coordinates of the robot
        self.x_position, self.y_position, self.z_position = self.get_start_position()
        
        #define initial velocities
        self.start_speed = 12.0
        self.left_wheel = self.start_speed
        self.right_wheel = self.start_speed
        self.max_speed = 30.0
        self.speed_change = 3.0         #step size to change speed 
        
        self.speed = self.get_speed() 
        self.changed = 0
        self.distance = 0 
        
        #helper classes 
        self.imgHelper = mi.MyImage()
        
        #publisher to publish on topic /cmd_vel 
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=100)
    
        #Add here the name of the ROS. In ROS, names are unique named.
        rospy.init_node('reinf_matrix_driving', anonymous=True)  
        #subscribe to a topic using rospy.Subscriber class
        self.sub=rospy.Subscriber('/camera/image_raw', Image, self.cam_im_raw_callback)  
     
    def save_position(self):
        try:
            #open correct file 
            f = open("/home/elisabeth/catkin_ws/src/drive_three_pi/src/Q_Matrix/position.txt", "a")
            #f = open("../Q_Matrix/Q-Matrix-Records.txt", "a")
            
            #pretty print matrix 
            end = time.time() 
            readable_time = time.ctime(end)
            string = str(readable_time)
            string += ("\n[x=" + str(self.x_position)) 
            string += (", y=" + str(self.y_position))
            string += (", z=" + str(self.z_position) + "]\n\n")
            
            #write into file 
            f.write(string)  
            
            #close file 
            f.close() 
        except Exception as e:
            print(str(e) + "\nFile not written")
    
    #returns the average speed of the robot 
    def get_speed(self):
        speed = (self.left_wheel + self.right_wheel) / 2.0
        return speed 
        

    #returns velocity message to actually make the robot drive 
    def set_msg(self, next_action):
        if(next_action == 6):
            #stop
            print("Next action = stop")
            self.left_wheel = self.start_speed
            self.right_wheel = self.start_speed
            self.set_position(self.x_position, self.y_position, self.z_position)
        elif(next_action == 0):
            #left wheel faster 
            print("Next action = left wheel faster")
            if(self.left_wheel <= self.max_speed - self.speed_change):
                self.left_wheel += self.speed_change
        elif(next_action == 1):
            #left wheel slower
            print("Next action = left wheel slower")
            if(self.left_wheel >= self.speed_change):
                self.left_wheel -= self.speed_change
        elif(next_action == 2):
            #right wheel faster  
            print("Next action = right wheel faster")
            if(self.right_wheel <= self.max_speed - self.speed_change):
                self.right_wheel += self.speed_change
        elif(next_action == 3):
            #right wheel slower 
            print("Next action = right wheel slower")
            if(self.right_wheel >= self.speed_change):
                self.right_wheel -= self.speed_change
        elif(next_action == 4):
            #both wheels faster
            print("Next action = both wheels faster")
            if(self.left_wheel <= self.max_speed - self.speed_change and self.right_wheel <= self.max_speed - self.speed_change):
                self.left_wheel += self.speed_change
                self.right_wheel += self.speed_change 
        else:
            #both wheels slower 
            print("Next action = both wheels slower")
            if(self.left_wheel >= self.speed_change and self.right_wheel >= self.speed_change):
                self.left_wheel -= self.speed_change
                self.right_wheel -= self.speed_change  
            
        msg = Twist()
        print("right_Wheel: " + str(self.right_wheel))
        print("left_Wheel: " + str(self.left_wheel))
        msg.linear.x = self.right_wheel 
        msg.linear.y = self.left_wheel 
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        
        return msg 
        
        
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
            resp = set_state( state_msg )

        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    ########################################################################################### 
        
    def get_start_position(self):
        model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        object_coordinates = model_coordinates("three_pi", "")
        x_position = object_coordinates.pose.position.x
        y_position = object_coordinates.pose.position.y
        z_position = object_coordinates.pose.position.z
        #print("x = " + str(x_position))
        #print("y = " + str(y_position))
        #print("z = " + str(z_position))
        return x_position, y_position, z_position
 
    ######################################### QUELLE ##########################################  
    '''
    1) https://medium.com/analytics-vidhya/the-epsilon-greedy-algorithm-for-reinforcement-learning-5fe6f96dc870
    2) Malte Hargarten_5_Deep-Q-Learning_Extended_4112.pdf
    3) https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/
    ''' 
    #epsilon greedy algorithm
    #decide whether to explore or to exploit
    def explore_vs_exploit(self, cnt, episodes):        
        #return value 
        explore = True 
        
        #have at least explored for 10% of all episodes 
        if(cnt < float(episodes) * (float(10) / float(100))):
            explore = True 
        else :
            #rate completed empisodes / all episodes 
            rel = float(cnt) / float(episodes)
            
            #random float between 0 and 1
            x = random.rand()
            
            #decrease epsilon over time 
            epsilon = 1 - rel 
            
            #do not explore if random float is greater than epsilon 
            if(x > epsilon):
                explore = False
            else:
                explore = True 
        
        return explore
    ###########################################################################################
        
    #if user pressed ctrl+c --> stop the robot
    def shutdown(self):
        print("Stopping")  
        #publish   
        self.vel_msg = self.set_msg(6)  
        self.velocity_publisher.publish(self.vel_msg)
        self.episodes_counter = 0
        
        end = time.time() 
        total = end - self.start
        minutes = total / 60.0 
        self.distance = self.speed * total 
        print("Total time = " + str(total) + " seconds = " + str(minutes) + " minutes")
        print("Distance = " + str(self.distance) + " meters" + " (ca. " + str(self.speed) + " m/s)")
            
    def reinf_main(self):
        bot = bt.Bot(self.speed)
        rospy.on_shutdown(self.shutdown) 
        
        #save starting position in txt file 
        #self.save_position()
        
        #put robot in front of sharp curve (try and error tested values)
        #self.x_position = 1.0
        #self.y_position = -5.77237738523
        #self.z_position = -0.0301354416609
        self.set_position(self.x_position, self.y_position, self.z_position)
        
        #get the time when training is finished and robot starts using filled q-matrix 
        driving_time = 0
        flag_driving_time = False 
        
        #Hyperparameters
        episodes = 1000
        sequence = 50
        gamma = 0.95
        alpha = 0.8
                
        try:        
            rate = rospy.Rate(50)
            while not rospy.is_shutdown():            
                if(self.flag): 
                    #segmentation
                    seg_img = self.imgHelper.segmentation(self.my_img)
                    
                    #explanation string for debugging 
                    explanation = "Explore = "
                    
                    #do reinforcement learning
                    if(self.episodes_counter < episodes):                        
                        #decide whether to explore or to exploit 
                        explore = self.explore_vs_exploit(self.episodes_counter, episodes)
                        
                        explanation +=  str(explore) + "\n Learning"
                        
                        #fill q -matrix 
                        for j in range(sequence):
                            self.speed = self.get_speed()
                            state = bot.q_learning(gamma, alpha, seg_img, explore, self.speed, self.max_speed)
                            if(state == 25):
                                self.episodes_counter += 1
                                state = 6       #stop robot
                                                
                            #publish 
                            print("Got action = " + str(state))
                            self.vel_msg = self.set_msg(state)
                            self.velocity_publisher.publish(self.vel_msg) 
                            
                    else:
                        #time measurement for comparing algorithms 
                        if not(flag_driving_time):
                            self.start = time.time()
                            flag_driving_time = True
                        #use q-matrix 
                        state = bot.use_q_matrix(seg_img, self.speed, self.max_speed)
                        
                        #publish 
                        print("Got action = " + str(state))
                        self.vel_msg = self.set_msg(state)
                        self.velocity_publisher.publish(self.vel_msg) 
                        
                        explanation = "Driving"
                    
                    #Print what's happening (learning/driving?, exploring/exploiting?, direction)
                    print(explanation + "\n")
                    
                    #set flag back to false to wait for a new image
                    self.flag = False 
            
            #average_speed = float(self.speed) / float(self.changed)
            bot.save_q_matrix(self.start, self.speed, self.distance)
            rate.sleep()
        except rospy.ROSInterruptException:
            pass
    
if __name__=='__main__':
    #try:
    node = Node()
    #node.main()
    node.reinf_main()
    #except Exception:
        #print("EXC")
        #pass
    
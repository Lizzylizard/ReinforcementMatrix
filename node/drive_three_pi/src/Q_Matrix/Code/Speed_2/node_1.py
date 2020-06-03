#!/usr/bin/env python

#import own scripts
import Bot_1 as bt
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
        
class Node:
        
    #constructor
    def __init__(self):            
        #global variables 
        self.my_img = []   
        self.curve =  "start"  
        self.vel_msg = Twist()
        self.flag = False
        self.start = time.time() 
        
        #starting coordinates of the robot
        self.x_position, self.y_position, self.z_position = self.get_start_position()
        
        #helper classes 
        self.imgHelper = mi.MyImage()
        
        #publisher to publish on topic /cmd_vel 
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=100)
    
        #Add here the name of the ROS. In ROS, names are unique named.
        rospy.init_node('reinf_matrix_driving', anonymous=True)  
        #subscribe to a topic using rospy.Subscriber class
        self.sub=rospy.Subscriber('/camera/image_raw', Image, self.cam_im_raw_callback)  
 

    ######################################### QUELLE ##########################################    
    '''
    https://answers.gazebosim.org//question/18372/getting-model-state-via-rospy/
    '''    
    def set_position(self):
        state_msg = ModelState()
        state_msg.model_name = 'three_pi'
        state_msg.pose.position.x = self.x_position
        state_msg.pose.position.y = self.y_position
        state_msg.pose.position.z = self.z_position
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
 
    
    def publish(self, msg):
        #publish  
        self.velocity_publisher.publish(msg) 
    
        
    #if user pressed ctrl+c --> stop the robot
    def shutdown(self):
        print("Stopping")

        #stop robot 
        vel_msg = Twist()
        vel_msg.linear.x = 0.0
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = 0.0
        self.publish(vel_msg)
        
        #put robot back to starting position 
        self.set_position()
        
        
    #callback; copies the received image into a global numpy-array
    def cam_im_raw_callback(self, msg):     
        #rospy.loginfo(msg.header)  

        #convert ROS image to cv image, copy it and save it as a global numpy-array
        img = self.imgHelper.img_conversion(msg) 
        self.my_img = np.copy(img)    
           
        #set flag to true, so main-loop knows, there's a new image to work with
        self.flag = True
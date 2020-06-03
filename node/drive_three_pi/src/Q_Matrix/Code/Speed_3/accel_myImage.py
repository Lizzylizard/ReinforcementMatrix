#!/usr/bin/env python

#import own scripts
import accel_reinf_matrix as rm
import accel_bot

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
        
class MyImage:
    def __init__(self):        
        # Initialize the CvBridge class
        self.bridge = CvBridge()
        
    # Try to convert the ROS Image message to a cv Image
    def img_conversion(self, ros_img):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_img, "passthrough")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        return cv_image 

    ########################################### QUELLE ###############################################
    ''' https://realpython.com/python-opencv-color-spaces/ oder 3-openCV_img_segmentation.pdf '''  
    #divides picture into two segments: 0 = floor (grey) 1 = line (black)
    #sets the floor pixels to white and the line pixels to black for an easier
    #edge detection
    def segmentation(self, img):
        #set color range
        
        light_black = (0, 0, 0)
        dark_black = (25, 25, 25)    
        
        #black and white image (2Darray): 255 => in color range, 0 => NOT in color range
        mask = cv.inRange(img, light_black, dark_black)                 
        #print(mask)
        
        return mask
    ##################################################################################################

    #counts the amount of white pixels of an image 
    #starting from the upper left corner 
    #to the upper right corner 
    #until the first black pixel is found
    #started counting the first ten rows -- changed to just one row
    def count_pxl(self, img):
        result = 0
        
        for i in range(1):                 #go from row 0 to 1 in steps of 1 (= the first row)
            k = 0
            j = img[i, k]                   
            while j <= 254:                 #as long as current pixel is black (!=255)
                result += 1
                k += 1
                if(k < len(img[i])):        #check if it's still in bounds
                    j = img[i, k]           #jump to next pixel
                else:
                    break
                
        return result
        
    def state(self, img, speed, max_speed):
        left = self.count_pxl(img)
        reversed_img = np.flip(img, 1)
        right = self.count_pxl(reversed_img)
        
        #width of image 
        width = np.size(img[0])
        
        #string for debugging
        line_string = "Line state = " 
        
        #return value 
        state = 0
        
        print("Width = " + str(width))
        print("Left = " + str(left))
        
        if(left >= (width * (99.0/100.0)) or right >= (width * (99.0/100.0))):
            #line is missing -> terminal state 
            line_string += "missing"
            state = 25
        elif(left > 0 and left <= (width/5.0)):
            #line is far left 
            line_string += "far left"
            if(speed == 0):
                #robot stopped
                state = 0
            elif(speed > 0 and speed <= max_speed * (25.0/100.0)):
                #speed slower than 25% of max speed 
                state = 1
            elif(speed > max_speed * (25.0/100.0) and speed <= max_speed * (50.0/100.0)):
                #speed slower than 50% of max speed 
                state = 2
            elif(speed > max_speed * (50.0/100.0) and speed <= max_speed * (75.0/100.0)):
                #speed slower than 75% of max speed 
                state = 3
            elif(speed > max_speed * (75.0/100.0)):
                #speed slower than 100% of max speed 
                state = 4
                
        elif(left > (width * (1.0/5.0)) and left <= (width * (2.0/5.0))):
            #line is left 
            line_string += "left"
            if(speed == 0):
                #robot stopped
                state = 5
            elif(speed > 0 and speed <= max_speed * (25.0/100.0)):
                #speed slower than 25% of max speed 
                state = 6
            elif(speed > max_speed * (25.0/100.0) and speed <= max_speed * (50.0/100.0)):
                #speed slower than 50% of max speed 
                state = 7
            elif(speed > max_speed * (50.0/100.0) and speed <= max_speed * (75.0/100.0)):
                #speed slower than 75% of max speed 
                state = 8
            elif(speed > max_speed * (75.0/100.0)):
                #speed slower than 100% of max speed 
                state = 9
                
        elif(left > (width * (2.0/5.0)) and left <= (width * (3.0/5.0))):
            #line is in the middle
            line_string += "middle"
            if(speed == 0):
                #robot stopped
                state = 10
            elif(speed > 0 and speed <= max_speed * (25/100.0)):
                #speed slower than 25% of max speed 
                state = 11
            elif(speed > max_speed * (25/100.0) and speed <= max_speed * (50/100.0)):
                #speed slower than 50% of max speed 
                state = 12
            elif(speed > max_speed * (50/100.0) and speed <= max_speed * (75/100.0)):
                #speed slower than 75% of max speed 
                state = 13
            elif(speed > max_speed * (75/100.0)):
                #speed slower than 100% of max speed 
                state = 14
                
        elif(left > (width * (3.0/5.0)) and left <= (width * (4.0/5.0))):
            #line is right
            line_string += "right"
            if(speed == 0):
                #robot stopped
                state = 15
            elif(speed > 0 and speed <= max_speed * (25.0/100.0)):
                #speed slower than 25% of max speed 
                state = 16
            elif(speed > max_speed * (25.0/100.0) and speed <= max_speed * (50.0/100.0)):
                #speed slower than 50% of max speed 
                state = 17
            elif(speed > max_speed * (50.0/100.0) and speed <= max_speed * (75.0/100.0)):
                #speed slower than 75% of max speed 
                state = 18
            elif(speed > max_speed * (75.0/100.0)):
                #speed slower than 100% of max speed 
                state = 19 
        else:
            #line is far right
            line_string += "far right"
            if(speed == 0):
                #robot stopped
                state = 20
            elif(speed > 0 and speed <= max_speed * (25.0/100.0)):
                #speed slower than 25% of max speed 
                state = 21
            elif(speed > max_speed * (25.0/100.0) and speed <= max_speed * (50.0/100.0)):
                #speed slower than 50% of max speed 
                state = 22
            elif(speed > max_speed * (50.0/100.0) and speed <= max_speed * (75.0/100.0)):
                #speed slower than 75% of max speed 
                state = 23
            elif(speed > max_speed * (75.0/100.0)):
                #speed slower than 100% of max speed 
                state = 24
                
        print(line_string)
        return state 
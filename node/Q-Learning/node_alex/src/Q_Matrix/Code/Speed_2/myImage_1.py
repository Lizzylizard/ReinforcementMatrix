#!/usr/bin/env python

#import own scripts
import node_1 as rm
import Bot_1

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
            while j <= 250:                 #as long as current pixel is black (!=255)
                result += 1
                k += 1
                if(k < len(img[i])):        #check if it's still in bounds
                    j = img[i, k]           #jump to next pixel
                else:
                    break
                
        return result
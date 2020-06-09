#!/usr/bin/env python

#import own scripts
import reinf_matrix_4 as rm
import Bot_4

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
    #returns already segmented image (black and white)
    def img_conversion(self, ros_img):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_img, "passthrough")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error: {0}".format(e))       
        
        seg_img = self.segmentation(cv_image)

        return seg_img 

    ########################################### QUELLE ###############################################
    ''' https://realpython.com/python-opencv-color-spaces/ oder 3-openCV_img_segmentation.pdf '''  
    #divides picture into two segments: 0 = floor (grey) 1 = line (black)
    #sets the floor pixels to white and the line pixels to black for an easier
    #edge detection
    def segmentation(self, img):
        #set color range
        
        light_black = (0, 0, 0)
        #dark_black = (25, 25, 25)    
        #dark_black = (70, 70, 70)   
        #dark_black = (1, 1, 1)   
        dark_black = (50, 50, 50)   
        
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
            #print("J = " + str(j))            
            while j < 251:                 #as long as current pixel is black (!= 255)
                result += 1
                k += 1
                if(k < len(img[i])):        #check if it's still in bounds
                    j = img[i, k]           #jump to next pixel
                else:
                    break
                
        return result
        
    def get_line_state_old(self, img):
        left = self.count_pxl(img)
        reversed_img = np.flip(img, 1)
        right = self.count_pxl(reversed_img)
        
        width = np.size(img[0])
        
        #print("Pixel number left = " + str(left))
        
        if(left >= (width * (99.0/100.0)) or right >= (width * (99.0/100.0))):
            #line is lost 
            #just define that if line is ALMOST lost, it is completely lost,
            #so terminal state gets reached 
            state = 7        
        elif(left > (width * (0.0/100.0)) and left <= (width * (10.0/100.0))):
            #line is far left 
            state = 0
        elif(left > (width * (10.0/100.0)) and left <= (width * (25.0/100.0))):
            #line is left 
            state = 1
        elif(left > (width * (25.0/100.0)) and left <= (width * (48.0/100.0))):
            #line is slightly left 
            state = 2
        elif(left > (width * (48.0/100.0)) and left <= (width * (52.0/100.0))):
            #line is in the middle 
            state = 3
        elif(left > (width * (52.0/100.0)) and left <= (width * (75.0/100.0))):
            #line is slightly right 
            state = 4        
        elif(left > (width * (75.0/100.0)) and left <= (width * (90.0/100.0))):
            #line is right 
            state = 5
        elif(left > (width * (90.0/100.0)) and left < (width * (99.0/100.0))):
            #line is far right 
            state = 6

        else:
            #left edge of line is out of the image (left = 0)
            if(right >= width * (52.00/100.0) and right < width * (75.00/100.0)):
                #line is slightly left
                state = 2
            if(right >= width * (75.00/100.0) and right < width * (90.0/100.0)):
                #line is left
                state = 1
            elif(right >= width * (90.0/100.0) and right < width * (99.0/100.0)):
                #line is far left
                state = 0
            else:
                #line is lost
                print("HIER NICHT REIN")
                state = 7
        print("Left: " + str(left))
        print("Right = " + str(right))
        return state


    def get_line_state(self, img):
        left = self.count_pxl(img)
        reversed_img = np.flip(img, 1)
        right = self.count_pxl(reversed_img)

        width = np.size(img[0])

        # print("Pixel number left = " + str(left))

        if (left >= (width * (99.0 / 100.0)) or right >= (width * (99.0 / 100.0))):
            # line is lost
            # just define that if line is ALMOST lost, it is completely lost,
            # so terminal state gets reached
            state = 7
        elif (left >= (width * (0.0 / 100.0)) and left <= (width * (15.0 / 100.0))):
            # line is far left
            state = 0
        elif (left > (width * (15.0 / 100.0)) and left <= (width * (30.0 / 100.0))):
            # line is left
            state = 1
        elif (left > (width * (30.0 / 100.0)) and left <= (width * (45.0 / 100.0))):
            # line is slightly left
            state = 2
        elif (left > (width * (45.0 / 100.0)) and left <= (width * (55.0 / 100.0))):
            # line is in the middle
            state = 3
        elif (left > (width * (55.0 / 100.0)) and left <= (width * (70.0 / 100.0))):
            # line is slightly right
            state = 4
        elif (left > (width * (70.0 / 100.0)) and left <= (width * (85.0 / 100.0))):
            # line is right
            state = 5
        elif (left > (width * (85.0 / 100.0)) and left < (width * (99.0 / 100.0))):
            # line is far right
            state = 6
        else:
            '''
            # left edge of line is out of the image (left = 0)
            if (right >= width * (52.00 / 100.0) and right < width * (75.00 / 100.0)):
                # line is slightly left
                state = 2
            if (right >= width * (75.00 / 100.0) and right < width * (90.0 / 100.0)):
                # line is left
                state = 1
            elif (right >= width * (90.0 / 100.0) and right < width * (99.0 / 100.0)):
                # line is far left
                state = 0
            else:
            '''
            # line is lost
            state = 7

        states_to_words = {
            0: "far left",
            1: "left",
            2: "slightly left",
            3: "middle",
            4: "slightly right",
            5: "right",
            6: "far right",
            7: "lost"
        }

        print("Left: " + str(left))
        print("Right:" + str(right))
        print("Line state is: " + states_to_words.get(state))

        return state


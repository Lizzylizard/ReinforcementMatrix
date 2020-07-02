#!/usr/bin/env python3

'''
rosrun image_view image_view image:=/camera/image_raw
right click to save image
'''
import math
import rospy

########################################### QUELLE ###############################################
''' https://dabit-industries.github.io/turtlebot2-tutorials/14b-OpenCV2_Python.html oder 5-ROS_to_OpenCV.pdf '''
#import numpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Import OpenCV libraries and tools
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

# Initialize the CvBridge class
bridge = CvBridge()

# Define a function to show the image in an OpenCV Window
def show_image(img):
    cv.imshow("Image Window", img)
    cv.waitKey(3)

################################################################################################## 

########################################### QUELLE ###############################################
''' https://realpython.com/python-opencv-color-spaces/ oder 3-openCV_img_segmentation.pdf '''  
#divides picture into two segments: 0 = floor (grey) 1 = line (black)
#sets the floor pixels to white and the line pixels to black for an easier
#edge detection
def segmentation(img):
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
def count_pxl(img):
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
    
#counts the amount of white pixel from the upper left corner 
#to the bottom left corner 
#until the first black pixel is found   
def count_pxl_vert(img):
    result = 0
    for i in range(0, len(img), 1):
        if(img[i, 0] <= 250):
            result += 1
        else:
            break
            
    #return distance to black pixel
    return result
    
#counts the distances from the line to the edge of the image from all four sides
def calc_ratios(img):      
    ############## HORIZONTALLY ###############
    #LEFT TOP
    cnt_left_top = count_pxl(img)      
    
    #LEFT BOTTOM      
    reversed_img = np.flip(img, 0)
    cnt_left_bot = count_pxl(reversed_img)
    
    #RIGHT TOP  
    vert_reversed_img = np.flip(img, 1)
    cnt_right_top = count_pxl(vert_reversed_img)     
    
    #RIGHT BOTTOM
    double_reversed_img = np.flip(vert_reversed_img, 0)
    cnt_right_bot = count_pxl(double_reversed_img)

    
    ############## VERTICALLY ###############
    #LEFT VERT TOP
    cnt_left_vert_top = count_pxl_vert(img)
    
    #LEFT VERT BOTTOM
    reversed_img = np.flip(img, 0)
    cnt_left_vert_bot = count_pxl_vert(reversed_img)
    
    #RIGHT VERT TOP
    vert_reversed_img = np.flip(img, 1)
    cnt_right_vert_top = count_pxl_vert(vert_reversed_img)
    
    #RIGHT VERT BOTTOM
    double_reversed_img = np.flip(vert_reversed_img, 0)
    cnt_right_vert_bot = count_pxl_vert(double_reversed_img)
    
    return (cnt_left_top, cnt_left_bot, cnt_left_vert_top, cnt_left_vert_bot, 
    cnt_right_top, cnt_right_bot, cnt_right_vert_top, cnt_right_vert_bot)
    
#calculates the curve with a picture of size 1x50 pixel 
def curve_one_row(img):
    left = count_pxl(img)
    reversed_img = np.flip(img, 1)
    right = count_pxl(reversed_img)
    
    width = np.size(img[0])
    half = width/2
    ten = width * (10.0/100.0)
    twenty_five = width * (25.0/100.0)
    seventy_five = width * (75.0/100.0)
    ninety = width * (90.0/100.0)
    
    if(left <= ten):
        curve = "sharp left"
    elif(left > ten and left <= twenty_five):
        curve = "left"
    elif(left > twenty_five and left <= half):
        curve = "slightly left"
    elif(left > half and left <= seventy_five):
        curve = "slightly right"
    elif(left > seventy_five and left <= ninety):
        curve = "right"
    else:
        curve = "sharp right"
        
    return curve    

#calculates the curve with a picture of size 1x50 pixel in a more complicated way 
#works with:
def complicated_curve_one_row(img):
    left = count_pxl(img)
    reversed_img = np.flip(img, 1)
    right = count_pxl(reversed_img) 
    width = np.size(img[0])
    line_width = width - left - right 
    middle = line_width / 2.0 
    location = left + middle 
    
    half = width/2
    spl = width * (3.0/100.0)
    sl = width * (10.0/100.0)
    lef = width * (25.0/100.0)
    sly_lef = width * (45.0/100.0)
    sly_righ = width * (55.0/100.0)
    righ = width * (75.0/100.0)
    sr = width * (90.0/100.0)
    spr = width * (97.0/100.0)
    
    if(location <= spl):
        curve = "sharpest left"
    if(location > spl and location <= sl):
        curve = "sharp left"
    elif(location > sl and location <= lef):
        curve = "left"
    elif(location > lef and location <= sly_lef):
        curve = "slightly left"
    elif(location > sly_lef and location <= sly_righ):
        curve = "forward"
    elif(location > sly_righ and location <= righ):
        curve = "slightly right"
    elif(location > righ and location <= sr):
        curve = "right"
    elif(location > sr and location <= spr):
        curve = "sharp right"
    else:
        curve = "sharpest right"
        
    return curve    
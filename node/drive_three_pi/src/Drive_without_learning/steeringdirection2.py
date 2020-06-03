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
#works with:
'''
Parameters:
camera angle: rpy="0.0 ${pi/7} 0.0"
image dimensions: 1x50
function: curve = sd.curve_one_row(seg_img)

Results:
Total time = 41.8598649502 seconds = 0.697664415836 minutes
Distance = 1255.79594851 meters (ca. 30.0 m/s)

biggest = 33.5 
big = 30.1
middle = 28.7
small = 27.3
smallest = 26.5
'''

'''
Parameters:
camera angle: rpy="0.0 ${pi/7} 0.0"
image dimensions: 1x20
function: curve = sd.curve_one_row(seg_img)

Results:
Total time = 137.323367119 seconds = 2.28872278531 minutes
Distance = 1098.58693695 meters (ca. 8.0 m/s)
frequency   /camerea/image_raw: ca. 14.5 hz
            /cmd_vel: ca. 14.5 hz

biggest = 10.0 
big = 9.0
middle = 8.0
small = 7.0
smallest = 6.0
'''
def curve_one_row(img):
    left = count_pxl(img)
    reversed_img = np.flip(img, 1)
    right = count_pxl(reversed_img)
    
    '''
    #number of all black pixel 
    all_black_pixels =  np.count_nonzero(img)
    
    #lost the line --> stop
    if(all_black_pixels <= 0):
        rospy.signal_shutdown("Lost the line!")
        curve = "stop"
        return curve '''
    
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
    
#decides whether the robot has to drive left or right in a more complicated way
#works with:
'''
Parameters:
camera angle: rpy="0.0 ${pi/2} 0.0"
image dimensions: 640x480
function: curve = sd.complicated_calc_curve(seg_img, curve)

Results: 
Total time = 70.063313961 seconds = 1.16772189935 minutes
Distance = 1261.1396513 meters (ca. 18.0 m/s)
frequency   /camerea/image_raw: ca. 11.5 hz
            /cmd_vel: ca. 11.4 hz

biggest = 20.0
big = 18.8
middle = 18.0
small = 17.2
smallest = 16.0
'''
def complicated_calc_curve(img):
    #calculate (count) white pixel and corresponding ratios
    (cnt_left_top, cnt_left_bot, cnt_left_vert_top, cnt_left_vert_bot, 
    cnt_right_top, cnt_right_bot, cnt_right_vert_top, cnt_right_vert_bot) = calc_ratios(img)  
    
    #number of all black pixel 
    all_black_pixels = np.count_nonzero(img)
    
    #number of pixel in one row and one column
    num_row = np.size(img[0])
    num_col = img.shape[0]
    
    #half of the picture
    half = float(num_row) / float(2.0)
    ten_perc_right = cnt_right_bot * (10.0/100.0)
    fifty_perc_right = cnt_right_bot * (50.0/100.0)    
    ten_perc_left = cnt_left_bot * (10.0/100.0)
    fifty_perc_left = cnt_left_bot * (50.0/100.0)
    
    #decide direction            
    
    #dummy
    curve = "forward" 
        
    #lost the line --> stop
    if(all_black_pixels <= 0):
        rospy.signal_shutdown("Lost the line!")
        curve = "stop"
        return curve
    
    #sharp left
    if(cnt_left_top <= 50):
        curve = "sharp left"
    elif(cnt_left_top < ten_perc_left):
        curve = "sharp left"
    #left
    elif(cnt_left_top > ten_perc_left and cnt_left_top < fifty_perc_left):
        curve = "left"
    #slighty left
    elif(cnt_left_top >= fifty_perc_left and cnt_left_top < cnt_left_bot):
        curve = "slightly left"
        
    #sharp right
    elif(cnt_right_top <= 50):
        curve = "sharp right"
    elif(cnt_right_top < ten_perc_right):
        curve = "sharp right"
    #right
    elif(cnt_right_top > ten_perc_right and cnt_right_top < fifty_perc_right):
        curve = "right"
    #slightly right 
    else:
        curve = "slightly right"
    
    #return curve for robot
    return curve 

if __name__=='__main__':   
    #open test image
    #img = cv.imread('img/test.jpg')        #check
    #img = cv.imread('img/test1.jpg')       #check
    #img = cv.imread('img/test2.jpg')       #check
    #img = cv.imread('img/test3.jpg')       #check
    #img = cv.imread('img/test4.jpg')       #check
    #img = cv.imread('img/test5.jpg')       #check
    #img = cv.imread('img/test6.jpg')       #check
    #img = cv.imread('img/test7.jpg')       #check
    #img = cv.imread('img/test8.jpg')       #check
    #img = cv.imread('img/test9.jpg')       #check
    #img = cv.imread('img/test10.jpg')      #check
    #img = cv.imread('img/test11.jpg')      #check
    #img = cv.imread('img/test12.jpg')      #check
    #img = cv.imread('img/test13.jpg')      #check
    #img = cv.imread('img/test14.jpg')      #check
    #img = cv.imread('img/test15.jpg')      #check           #why is the whole picture black?? should be white
    #img = cv.imread('img/test16.jpg')      #check
    #img = cv.imread('img/test17.jpg')      #check
    #img = cv.imread('img/test18.jpg')      #check
    #img = cv.imread('img/test19.jpg')      #
    #img = cv.imread('img/test20.jpg')      #
    #img = cv.imread('img/test21.jpg')      #
    #img = cv.imread('img/test22.jpg')      #
    #img = cv.imread('img/test23.jpg')      #
    #img = cv.imread('img/test24.jpg')      #
    #img = cv.imread('img/test25.jpg')      #
    #img = cv.imread('img/test26.jpg')      #
    #img = cv.imread('img/test27.jpg')      #
    #img = cv.imread('img/test28.jpg')      #
    #img = cv.imread('img/test29.jpg')      #
    #img = cv.imread('img/test30.jpg')      #
    #img = cv.imread('img/test31.jpg')      #
    #img = cv.imread('img/test32.jpg')      #
    img = cv.imread('img/test33.jpg')      #
    #img = cv.imread('img/test34.jpg')      #
    #img = cv.imread('img/test35.jpg')      #
    #img = cv.imread('img/test36.jpg')      #
    #img = cv.imread('img/test37.jpg')      #
    #img = cv.imread('img/test38.jpg')      #
    #img = cv.imread('img/test39.jpg')      #
    #img = cv.imread('img/test40.jpg')      #
    #img = cv.imread('img/test41.jpg')      #
    #img = cv.imread('img/test42.jpg')      #
    #img = cv.imread('img/test43.jpg')      #
    #img = cv.imread('img/test44.jpg')      #
    #img = cv.imread('img/test45.jpg')      #
    #img = cv.imread('img/test46.jpg')      #
    #img = cv.imread('img/test47.jpg')      #
    #img = cv.imread('img/test48.jpg')      #
    #img = cv.imread('img/test49.jpg')      #
    #img = cv.imread('img/test50.jpg')      #
    #img = cv.imread('img/test51.jpg')      #
    #print(type(img))
    plt.imshow(img)
    plt.title("Original")
    plt.show()
    
    #crop image
    #print("Dimensions before = " + str(img.shape))
    cropped_img = crop_image(img)
    #print("Dimensions after = " + str(cropped_img.shape))
    
    #segmentation
    seg_img = segmentation(cropped_img)
    #print(seg_img)
    plt.imshow(seg_img, cmap = "gray")
    plt.title("Segmented")
    plt.show()
    
    #choose steering direction
    curve = calc_curve(seg_img)
    print("Robot should drive " +  str(curve))    
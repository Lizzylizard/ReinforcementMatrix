#!/usr/bin/env python

#import own scripts
import steeringdirection2 as sd

import time
import rospkg 

########################################### QUELLE ########################################
''' https://dabit-industries.github.io/turtlebot2-tutorials/14b-OpenCV2_Python.html oder 5-ROS_to_OpenCV.pdf '''
#remove or add the library/libraries for ROS
import rospy

#import numpy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Import OpenCV libraries and tools
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

#remove or add the message type
from std_msgs.msg import String, Float32, Int32
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

#Initialize the CvBridge class
bridge = CvBridge()

# Try to convert the ROS Image message to a cv Image
def img_conversion(ros_img):
    try:
        cv_image = bridge.imgmsg_to_cv2(ros_img, "passthrough")
    except CvBridgeError, e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    return cv_image  
###########################################################################################

######################################### QUELLE ##########################################
''' Quelle:
https://entwickler.de/online/python/switch-case-statement-python-tutorial-579894245.html oder 1-switch_python.pdf
'''    
#'switch'; calls function for curve string 
def translateToVel(curve, biggest, big, middle, small, smallest):
    vel = Twist()
    directions = {
        "forward": forward,
        "backwards": backwards,
        "backwards right": backwards_right,
        "backwards left": backwards_left,
        "slightly left": slightly_left,
        "left": left,
        "sharp left": sharp_left,
        "sharpest left": sharpest_left,
        "slightly right": slightly_right,
        "right": right,
        "sharp right": sharp_right,
        "sharpest right": sharpest_right,
        "stop": stop
        }
    function = directions.get(curve)
    vel = function(biggest, big, middle, small, smallest)
    return vel
    
#sets fields of Twist variable so robot drives forward
def forward(biggest, big, middle, small, smallest):
    vel_msg = Twist()
    vel_msg.linear.x = biggest      
    vel_msg.linear.y = biggest 
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    #print("RIGHT")
    return vel_msg
    
#sets fields of Twist variable so robot drives backwards
def backwards(biggest, big, middle, small, smallest):
    vel_msg = Twist()
    vel_msg.linear.x = biggest * (-1.0)     
    vel_msg.linear.y = biggest * (-1.0)
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    #print("RIGHT")
    return vel_msg    

#sets fields of Twist variable so robot drives backwards coming from the left
def backwards_left(biggest, big, middle, small, smallest):
    vel_msg = Twist()
    vel_msg.linear.x = biggest * (-1.0)    
    vel_msg.linear.y = middle * (-1.0)
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    #print("RIGHT")
    return vel_msg
    
#sets fields of Twist variable so robot drives backwards coming from the right
def backwards_right(biggest, big, middle, small, smallest):
    vel_msg = Twist()
    vel_msg.linear.x = middle * (-1.0)     
    vel_msg.linear.y = biggest * (-1.0)
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    #print("RIGHT")
    return vel_msg  

def slightly_left(biggest, big, middle, small, smallest):
    vel_msg = Twist()
    vel_msg.linear.x = biggest    
    vel_msg.linear.y = big
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    #print("LEFT")
    return vel_msg
    
#sets fields of Twist variable so robot drives left
def left(biggest, big, middle, small, smallest):
    vel_msg = Twist()
    vel_msg.linear.x = biggest     
    vel_msg.linear.y = middle
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    #print("LEFT")
    return vel_msg
    
#sets fields of Twist variable so robot drives sharp left
def sharp_left(biggest, big, middle, small, smallest):
    vel_msg = Twist()
    vel_msg.linear.x = biggest      
    vel_msg.linear.y = small
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    #print("LEFT")
    return vel_msg
    
#sets fields of Twist variable so robot drives sharpest left
def sharpest_left(biggest, big, middle, small, smallest):
    vel_msg = Twist()
    vel_msg.linear.x = biggest     
    vel_msg.linear.y = smallest
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    #print("LEFT")
    return vel_msg
    
#sets fields of Twist variable so robot drives slightly right
def slightly_right(biggest, big, middle, small, smallest):
    vel_msg = Twist()
    vel_msg.linear.x = big 
    vel_msg.linear.y = biggest
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    #print("RIGHT")
    return vel_msg
    
#sets fields of Twist variable so robot drives right
def right(biggest, big, middle, small, smallest):
    vel_msg = Twist()
    vel_msg.linear.x = middle  
    vel_msg.linear.y = biggest
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    #print("RIGHT")
    return vel_msg
    
#sets fields of Twist variable so robot drives sharp right
def sharp_right(biggest, big, middle, small, smallest):
    vel_msg = Twist()
    vel_msg.linear.x = small   
    vel_msg.linear.y = biggest
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    #print("RIGHT")
    return vel_msg
    
#sets fields of Twist variable so robot drives sharpest right
def sharpest_right(biggest, big, middle, small, smallest):
    vel_msg = Twist()
    vel_msg.linear.x = smallest    
    vel_msg.linear.y = biggest
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    #print("RIGHT")
    return vel_msg
    
#sets fields of Twist variable to stop robot and puts the robot back to starting position 
def stop(biggest, big, middle, small, smallest):
    set_position()
    vel_msg = Twist()
    vel_msg.linear.x = 0.0       
    vel_msg.linear.y = 0.0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0
    #print("RIGHT")
    return vel_msg

########################################################################################### 

######################################### QUELLE ##########################################    
'''
https://answers.gazebosim.org//question/18372/getting-model-state-via-rospy/
'''    
def set_position():
    global x_position, y_position, z_position
    state_msg = ModelState()
    state_msg.model_name = 'three_pi'
    state_msg.pose.position.x = x_position
    state_msg.pose.position.y = y_position
    state_msg.pose.position.z = z_position
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

######################################### QUELLE ##########################################
'''
https://answers.ros.org/question/261782/how-to-use-getmodelstate-service-from-gazebo-in-python/
'''
def get_start_position():
    model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    object_coordinates = model_coordinates("three_pi", "")
    x_position = object_coordinates.pose.position.x
    y_position = object_coordinates.pose.position.y
    z_position = object_coordinates.pose.position.z
    #print("x = " + str(x_position))
    #print("y = " + str(y_position))
    #print("z = " + str(z_position))
    return x_position, y_position, z_position
###########################################################################################

######################################### QUELLE ##########################################
''' Quelle:
https://www.intorobotics.com/template-for-a-ros-subscriber-in-python/ oder 2-subscriber_node.pdf und
https://www.intorobotics.com/template-for-a-ros-publisher-in-python/ oder 1-publisher_node.pdf
'''

#if user pressed ctrl+c --> stop the robot
def shutdown():
    print("Stopping")  
    #publish   
    vel_msg = stop(0.0, 0.0, 0.0, 0.0, 0.0)  
    velocity_publisher.publish(vel_msg)
    
    global start, biggest, smallest
    end = time.time() 
    total = end - start
    minutes = total / 60.0 
    speed = (biggest + smallest) / 2.0
    distance = speed * total 
    print("Total time = " + str(total) + " seconds = " + str(minutes) + " minutes")
    print("Distance = " + str(distance) + " meters" + " (ca. " + str(speed) + " m/s)")


#callback; copies the received image into a global numpy-array
def cam_im_raw_callback(msg):     
    #rospy.loginfo(msg.header)  

    #convert ROS image to cv image, copy it and save it as a global numpy-array
    global my_img
    img = img_conversion(msg) 
    my_img = np.copy(img)    
       
    #set flag to true, so main-loop knows, there's a new image to work with
    global flag 
    flag = True  

#publisher to publish on topic /cmd_vel 
velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=100)

if __name__=='__main__':  
    #global variables 
    my_img = []   
    curve =  "start"  
    vel_msg = Twist()
    flag = False
    start = time.time() 
    
    #starting coordinates of the robot
    x_position, y_position, z_position = get_start_position()
    
    #define velocities
    biggest = 25.0
    big = 22.6
    middle = 21.6
    small = 20.6
    smallest = 20.0
            
    rospy.on_shutdown(shutdown) 
    
    try:
        #Add here the name of the ROS. In ROS, names are unique named.
        rospy.init_node('drive_on_line', anonymous=True)  
        #subscribe to a topic using rospy.Subscriber class
        sub=rospy.Subscriber('/camera/image_raw', Image, cam_im_raw_callback) 
        
        get_start_position()
        
        rate = rospy.Rate(50)
        #while(True): 
        while not rospy.is_shutdown():
            
            #print(flag)
            
            if(flag):            
                #cut the last 30 rows off the image
                #cropped_img = sd.crop_image(my_img)
                
                #segmentation
                seg_img = sd.segmentation(my_img)

                #choose steering direction
                curve = sd.curve_one_row(seg_img)
                #curve = sd.complicated_curve_one_row(seg_img)
                #curve = sd.complicated_calc_curve(seg_img)
                print(curve)    
                
                #turn the curve-string into a valid message type
                vel_msg = translateToVel(curve, biggest, big, middle, small, smallest)
                
                #publish  
                velocity_publisher.publish(vel_msg) 
                
                #set flag back to false to wait for a new image
                flag = False 
        
        rate.sleep()
    except rospy.ROSInterruptException:
        pass     
    
########################################################################################### 
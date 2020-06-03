#!/usr/bin/env python

#import own scripts
import steeringdirection as sd

#################################### QUELLE ANGEBEN #######################################
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

class Driver():
    #class variables 
    img_cnt = 0 
    # Initialize the CvBridge class
    bridge = CvBridge()
    
    #constructor
    def __init__(self):
        #Add here the name of the ROS. In ROS, names are unique named.
        rospy.init_node('drive_on_line', anonymous=True)
        #subscribe to a topic using rospy.Subscriber class
        sub=rospy.Subscriber('/camera/image_raw', Image, self.cam_im_raw_callback)
        rospy.spin()
            
    # Define a function to show the image in an OpenCV Window
    def show_image(self,img):
        cv.imshow("Image Window", img)
        cv.waitKey(3)

    # Try to convert the ROS Image message to a cv Image
    def img_conversion(self, ros_img):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_img, "passthrough")
        except CvBridgeError, e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        # Show the converted image
        #show_image(cv_image) 
        return cv_image  
    ###########################################################################################

    #define function/functions to provide the required functionality
    def cam_im_raw_callback(self, msg):
        #print("Image Count = " + str(self.img_cnt))
                    
        #publish on topic /cmd_vel 
        velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)            
        vel_msg = Twist()
        vel_msg = self.straight()
        
        #only react to every 10th image --> else: drive straight
        if(self.img_cnt % 10 == 0): 
            print("TEN!")        
            #rospy.loginfo(msg.header)   
           
            #convert ROS image to cv image
            img = self.img_conversion(msg)
            
            #crop image
            #print("Dimensions before = " + str(img.shape))
            cropped_img = sd.crop_image(img)
            #print("Dimensions after = " + str(cropped_img.shape))
            
            #segmentation
            seg_img = sd.segmentation(cropped_img)
            #print(seg_img)
            #plt.imshow(seg_img, cmap = "gray")
            #plt.show()
           
            #detect edges
            #edge_img = edge_detection(seg_img)

            #choose steering direction
            curve = sd.calc_curve(seg_img)
            print("Robot should drive " +  str(curve))    

            vel_msg = self.translateToVel(curve)
            #print(translateToVel(curve))
            self.img_cnt = 1
        else:
            self.img_cnt += 1
        
        #print(vel_msg)
        velocity_publisher.publish(vel_msg)

    ######################################### QUELLE ##########################################
    ''' Quelle:
    https://entwickler.de/online/python/switch-case-statement-python-tutorial-579894245.html oder 1-switch_python.pdf
    '''    
    #'switch' - calls function for curve string 
    def translateToVel(self, curve):
        vel = Twist()
        directions = {
            "straight": self.straight,
            "backwards": self.backwards,
            "left": self.left,
            "right": self.right,
            "sharp left": self.sharp_left,
            "sharp right": self.sharp_right,
            }
        function = directions.get(curve)
        vel = function()
        return vel
        
    #sets fields of Twist variable so robot drives straight
    def straight(self):
        vel_msg = Twist()
        vel_msg.linear.x = 2.0        
        vel_msg.linear.y = 2.0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        #print("STRAIGHT")
        return vel_msg
        
    #sets fields of Twist variable so robot drives backwards
    def backwards(self):
        vel_msg = Twist()
        vel_msg.linear.x = -2.0        
        vel_msg.linear.y = -2.0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        #print("BACKWARDS")
        return vel_msg
        
    #sets fields of Twist variable so robot drives left
    def left(self):
        vel_msg = Twist()
        vel_msg.linear.x = 1.5        
        vel_msg.linear.y = 2.0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        #print("LEFT")
        return vel_msg
        
    #sets fields of Twist variable so robot drives sharp left
    def sharp_left(self):
        vel_msg = Twist()
        vel_msg.linear.x = 1.0        
        vel_msg.linear.y = 2.0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        #print("SHARP LEFT")
        return vel_msg
        
    #sets fields of Twist variable so robot drives right
    def right(self):
        vel_msg = Twist()
        vel_msg.linear.x = 2.0        
        vel_msg.linear.y = 1.5
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        #print("RIGHT")
        return vel_msg
        
    #sets fields of Twist variable so robot drives sharp right
    def sharp_right(self):
        vel_msg = Twist()
        vel_msg.linear.x = 2.0        
        vel_msg.linear.y = 1.0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        #print("SHARP RIGHT")
        return vel_msg
    ###########################################################################################       

if __name__=='__main__':
    driver1 = Driver()
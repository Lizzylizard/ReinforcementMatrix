#!/usr/bin/env python

#import own scripts
import Bot as bt
import MyImage as mi

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
        self.biggest = 25.0
        self.big = 22.6
        self.middle = 21.6
        self.small = 20.6
        self.smallest = 20.0        
        
        self.speed = (self.biggest + self.smallest) / 2.0
        self.distance = 0 
        
        #helper classes 
        self.imgHelper = mi.MyImage()
        
        #publisher to publish on topic /cmd_vel 
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=100)
    
        #Add here the name of the ROS. In ROS, names are unique named.
        rospy.init_node('reinf_matrix_driving', anonymous=True)  
        #subscribe to a topic using rospy.Subscriber class
        self.sub=rospy.Subscriber('/camera/image_raw', Image, self.cam_im_raw_callback)  
        
    ######################################### QUELLE ##########################################
    ''' Quelle:
    https://entwickler.de/online/python/switch-case-statement-python-tutorial-579894245.html oder 1-switch_python.pdf
    '''    
    #'switch'; calls function for curve string 
    def translateToVel(self, curve):
        vel = Twist()
        directions = {
            "sharp left": self.sharp_left,
            "left": self.left,
            "slightly left": self.slightly_left,
            "slightly right": self.slightly_right,
            "right": self.right,
            "sharp right": self.sharp_right,
            "stop": self.stop,
            "accelerate": self.increase_velocity,
            "brake": self.decrease_velocity
            }
        function = directions.get(curve)
        vel = function(self.biggest, self.big, self.middle, self.small, self.smallest)
        return vel
        
    #increase velocity 
    def increase_velocity(self):
        self.biggest += 2
        self.big += 2
        self.middle += 2
        self.small += 2
        self.smallest += 2
        
        vel_msg = Twist()
        self.vel_msg.linear.x += 2
        self.vel_msg.linear.y += 2
        vel_msg.linear.x = self.vel_msg.linear.x   
        vel_msg.linear.y = self.vel_msg.linear.y
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0 
        
        return vel_msg
        
    #decrease velocity 
    def decrease_velocity(self):
        self.biggest -= 2
        self.big -= 2
        self.middle -= 2
        self.small -= 2
        self.smallest -= 2
        
        vel_msg = Twist()
        self.vel_msg.linear.x -= 2
        self.vel_msg.linear.y -= 2
        vel_msg.linear.x = self.vel_msg.linear.x   
        vel_msg.linear.y = self.vel_msg.linear.y
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0 
        
        return vel_msg
        
    #sets fields of Twist variable so robot drives sharp left
    def sharp_left(self, biggest, big, middle, small, smallest):
        vel_msg = Twist()
        vel_msg.linear.x = biggest      
        vel_msg.linear.y = small
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0        
        
        self.vel_msg = vel_msg
        
        #print("LEFT")
        return vel_msg

    def slightly_left(self, biggest, big, middle, small, smallest):
        vel_msg = Twist()
        vel_msg.linear.x = biggest    
        vel_msg.linear.y = big
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        
        self.vel_msg = vel_msg
        
        #print("LEFT")
        return vel_msg
        
    #sets fields of Twist variable so robot drives left
    def left(self, biggest, big, middle, small, smallest):
        vel_msg = Twist()
        vel_msg.linear.x = biggest     
        vel_msg.linear.y = middle
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        
        self.vel_msg = vel_msg
        
        #print("LEFT")
        return vel_msg
        
    #sets fields of Twist variable so robot drives slightly right
    def slightly_right(self, biggest, big, middle, small, smallest):
        vel_msg = Twist()
        vel_msg.linear.x = big 
        vel_msg.linear.y = biggest
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        
        self.vel_msg = vel_msg
        
        #print("RIGHT")
        return vel_msg
        
    #sets fields of Twist variable so robot drives right
    def right(self, biggest, big, middle, small, smallest):
        vel_msg = Twist()
        vel_msg.linear.x = middle  
        vel_msg.linear.y = biggest
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        
        self.vel_msg = vel_msg
        
        #print("RIGHT")
        return vel_msg
        
    #sets fields of Twist variable so robot drives sharp right
    def sharp_right(self, biggest, big, middle, small, smallest):
        vel_msg = Twist()
        vel_msg.linear.x = small   
        vel_msg.linear.y = biggest
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        
        self.vel_msg = vel_msg
        
        #print("RIGHT")
        return vel_msg
        
    #sets fields of Twist variable to stop robot and puts the robot back to starting position 
    def stop(self, biggest, big, middle, small, smallest):
        self.episodes_counter += 1
        self.set_position()
        vel_msg = Twist()
        vel_msg.linear.x = 0.0       
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        
        self.vel_msg = vel_msg
        
        #print("RIGHT")
        return vel_msg

    ########################################################################################### 

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
        self.vel_msg = self.stop(0.0, 0.0, 0.0, 0.0, 0.0)  
        self.velocity_publisher.publish(self.vel_msg)
        
        end = time.time() 
        total = end - self.start
        minutes = total / 60.0 
        self.distance = self.speed * total 
        print("Total time = " + str(total) + " seconds = " + str(minutes) + " minutes")
        print("Distance = " + str(self.distance) + " meters" + " (ca. " + str(self.speed) + " m/s)")
            
    def main(self):        
        rospy.on_shutdown(self.shutdown) 
        
        try:        
            rate = rospy.Rate(50)
            while not rospy.is_shutdown():            
                if(self.flag): 
                    #segmentation
                    seg_img = self.imgHelper.segmentation(self.my_img)

                    #choose steering direction
                    self.curve = self.imgHelper.curve_one_row(seg_img)
                    #curve = sd.complicated_curve_one_row(seg_img)
                    #curve = sd.complicated_calc_curve(seg_img)
                    print(self.curve)    
                    
                    #turn the curve-string into a valid message type
                    self.vel_msg = self.translateToVel()
                    
                    #publish  
                    self.velocity_publisher.publish(self.vel_msg) 
                    
                    #set flag back to false to wait for a new image
                    self.flag = False 
            
            rate.sleep()
        except rospy.ROSInterruptException:
            pass
            
    def reinf_main(self):
        self.start = time.time()
        bot = bt.Bot()
        rospy.on_shutdown(self.shutdown) 
        
        #Hyperparameters
        episodes = 400
        sequence = 20
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
                            self.curve = bot.q_learning(gamma, alpha, seg_img, explore)
                            #if(self.curve == "stop"):
                                #break 
                    else:
                        #use q-matrix 
                        self.curve = bot.use_q_matrix(seg_img)
                        
                        explanation = "Driving"
                    
                    #Print what's happening (learning/driving?, exploring/exploiting?, direction)
                    print(explanation + "\n" + str(self.curve))
                    
                    #turn the curve-string into a valid message type
                    self.vel_msg = self.translateToVel(self.curve)
                    
                    #publish  
                    self.velocity_publisher.publish(self.vel_msg) 
                    
                    #set flag back to false to wait for a new image
                    self.flag = False 
            
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
    
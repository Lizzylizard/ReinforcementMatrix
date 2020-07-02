#!/usr/bin/env python

import rospy 
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Float32, Int32
from geometry_msgs.msg import Twist

import time

def time_callback(msg):
    #print("Topic /camera/image_raw")
    global message_counter, all_messages, flag 
    message_counter += 1
    all_messages += 1
    #print("Current messages = " + str(message_counter))
    #print("All messages = " + str(all_messages))
    flag = True
    
def vel_callback(msg):
    #print("Topic /cmd_vel")
    global vel_counter, all_vels, flag2
    vel_counter += 1
    all_vels += 1
    flag2 = True 

if __name__=='__main__':
    message_counter = 0
    all_messages = 0
    second_counter = 0
    frequency = 0.0
    start_time = time.time() * 1000
    
    vel_counter = 0
    all_vels = 0
    vel_seconds = 0
    vel_freq = 0.0
    vel_start = time.time() * 1000    
    
    flag = False 
    flag2 = False 
    
    #publisher to publish on topic /sub_measure_time
    time_publisher = rospy.Publisher('/sub_measure_time', String, queue_size=100)
    
    #Add here the name of the ROS. In ROS, names are unique named.
    rospy.init_node('measure_time', anonymous=True)  
    #subscribe to a topic using rospy.Subscriber class
    sub=rospy.Subscriber('/camera/image_raw', Image, time_callback) 
    sub2 = rospy.Subscriber('/cmd_vel', Twist, vel_callback)
    
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        if(flag):
            print("\n\nIMAGES")
            current_time = time.time() * 1000
            #print(str(start_time - current_time))
            if((current_time - start_time) >= 1000.0):
                frequency = message_counter
                message_counter = 0
                start_time = time.time() * 1000
                second_counter += 1
            if not second_counter == 0:
                average = float(all_messages) / float(second_counter)
            else:
                average = 0.0
            
            #publish  
            message = "\nFrequency = " + str(frequency) + "\naverage = " + str(average)
            time_publisher.publish(message)
            
            print(message)
            flag = False
            
        if(flag2):
            print("\n\nVELOCITY")
            current_time = time.time() * 1000
            #print(str(start_time - current_time))
            if((current_time - vel_start) >= 1000.0):
                vel_freq = vel_counter
                vel_counter = 0
                vel_start = time.time() * 1000
                vel_seconds += 1
            if not vel_seconds == 0:
                average = float(all_vels) / float(vel_seconds)
            else:
                average = 0.0
            
            #publish  
            message = "\nFrequency = " + str(frequency) + "\naverage = " + str(average)
            
            print(message)
            flag = False
        
        rate.sleep()
#!/usr/bin/env python

#import own scripts
import Bot_2 as bt
import MyImage_2 as mi

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
import random 
        
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
        #helper classes         
        self.bot = bt.Bot()
        
        #global variables 
        self.my_img = []   
        self.curve =  "start"  
        self.vel_msg = Twist()
        self.flag = False
        self.start = time.time()

        #status of lost line
        self.lost_line = 28
        #acion to stop
        self.stop_action = 8
        
        #starting coordinates of the robot
        self.x_position, self.y_position, self.z_position = self.get_start_position()        
        #self.save_position()
        
        #inital values 
        self.inital_biggest = 25.0
        self.initial_big = 22.6
        self.initial_middle = 21.6
        self.initial_small = 20.6
        self.initial_smallest = 20.0        
                
        #define velocities
        self.biggest = self.inital_biggest
        self.big = self.initial_big
        self.middle = self.initial_middle
        self.small = self.initial_small
        self.smallest = self.initial_smallest
        
        #define how fast /slow robot gets max!
        self.max_speed = 40.0
        self.min_speed = 20.0

        #define how fast speed changes if it changes
        self.speed_change = self.biggest * (5.0/100.0)
        
        #keep track of current speed and save initial speed
        self.all_speeds = [self.initial_middle]
        
        #helper classes 
        self.imgHelper = mi.MyImage()
        
        self.action_strings = {
            0: "sharp left",
            1: "left",
            2: "slightly left",
            3: "forward",
            4: "slightly right",
            5: "right",
            6: "sharp right",
            7: "slower",
            8: "stop"
            }
        
        #publisher to publish on topic /cmd_vel 
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=100)
    
        #Add here the name of the ROS. In ROS, names are unique named.
        rospy.init_node('reinf_matrix_driving', anonymous=True)  
        #subscribe to a topic using rospy.Subscriber class
        self.sub=rospy.Subscriber('/camera/image_raw', Image, self.cam_im_raw_callback) 
        
        
    #sets fields of Twist variable so robot drives sharp left
    def sharp_left(self, msg):
        #get faster
        vel_msg = Twist()
        vel_msg = self.faster(msg)

        '''
        #return correct speed
        #vel_msg = Twist()
        vel_msg.linear.x = self.biggest
        vel_msg.linear.y = self.small
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        #print("LEFT")
        '''
        return vel_msg

    # sets fields of Twist variable so robot drives slightly left
    def slightly_left(self, msg):
        #get faster
        vel_msg = Twist()
        vel_msg = self.faster(msg)

        '''
        #return correct speed
        #vel_msg = Twist()
        vel_msg.linear.x = self.biggest
        vel_msg.linear.y = self.big
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        #print("LEFT")
        '''
        return vel_msg
        
    #sets fields of Twist variable so robot drives left
    def left(self, msg):
        # get faster
        vel_msg = Twist()
        vel_msg = self.faster(msg)

        '''
        # return correct speed
        #vel_msg = Twist()
        vel_msg.linear.x = self.biggest
        vel_msg.linear.y = self.middle
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        #print("LEFT")
        '''
        return vel_msg
        
    #sets fields of Twist variable so robot drives forward
    def forward(self, msg):
        #get faster
        vel_msg = Twist()
        vel_msg = self.faster(msg)

        '''
        #return correct speed
        #vel_msg = Twist()
        vel_msg.linear.x = self.biggest
        vel_msg.linear.y = self.biggest
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        #print("FORWARD")
        '''
        return vel_msg
        
    #sets fields of Twist variable so robot drives slightly right
    def slightly_right(self, msg):
        # get faster
        vel_msg = Twist()
        vel_msg = self.faster(msg)

        '''
        # return correct speed
        #vel_msg = Twist()
        vel_msg.linear.x = self.big
        vel_msg.linear.y = self.biggest
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        #print("RIGHT")
        '''
        return vel_msg
        
    #sets fields of Twist variable so robot drives right
    def right(self, msg):
        # get faster
        vel_msg = Twist()
        vel_msg = self.faster(msg)

        '''
        # return correct speed
        #vel_msg = Twist()
        vel_msg.linear.x = self.middle
        vel_msg.linear.y = self.biggest
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        #print("RIGHT")
        '''
        return vel_msg
        
    #sets fields of Twist variable so robot drives sharp right
    def sharp_right(self, msg):
        # get faster
        vel_msg = Twist()
        vel_msg = self.faster(msg)

        '''
        # return correct speed
        #vel_msg = Twist()
        vel_msg.linear.x = self.small
        vel_msg.linear.y = self.biggest
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        #print("RIGHT")
        '''
        return vel_msg
        
    #makes robot drive proportionally faster 
    def faster(self, msg):
        print("Faster")
        vel = self.change_speed(1, msg)
        #print("Vel msg in faster() = " + str(vel))
        return vel
        
    #makes robot drive proportionally slower 
    def slower(self, msg):
        print("Slower")
        return self.change_speed(-1, msg)
    
    #sign = +1 => robot gets faster 
    #sign = -1 => robot gets slower 
    def change_speed(self, sign, msg):
        change_factor = (5.0/100.0)
        if(msg.linear.x <= self.max_speed and msg.linear.x >= self.min_speed):
            print("Im if!")
            self.biggest += (self.biggest * sign * change_factor)
            self.big += (self.big * sign * change_factor)
            self.middle += (self.middle * sign * change_factor)
            self.small += (self.small * sign * change_factor)
            self.smallest += (self.smallest * sign * change_factor) 
            
            msg.linear.x += (msg.linear.x * sign * change_factor)
            msg.linear.y += (msg.linear.y * sign * change_factor)
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0
        else:
            pass

        print("Vel msg in change_speed() = " + str(msg))
        return msg

    #sets fields of Twist variable to stop robot and puts the robot back to starting position
    #stop robot and set it back to a random starting position
    def stop(self, msg):
        #self.episodes_counter += 1
        self.choose_random_starting_position()
        self.set_position(self.x_position, self.y_position, self.z_position)
        
        vel_msg = Twist()
        vel_msg.linear.x = 0.0       
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        #print("RIGHT")
        return vel_msg
    

    ######################################### QUELLE ##########################################    
    '''
    https://answers.gazebosim.org//question/18372/getting-model-state-via-rospy/
    '''    
    def set_position(self, x, y, z):
        state_msg = ModelState()
        state_msg.model_name = 'three_pi'
        state_msg.pose.position.x = x
        state_msg.pose.position.y = y
        state_msg.pose.position.z = z
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
        
    def save_position(self):
        try:
            #open correct file 
            f = open("/home/elisabeth/catkin_ws/src/drive_three_pi/src/Q_Matrix/Code/Speed_4/position.txt", "a")
            #f = open("../Q_Matrix/Q-Matrix-Records.txt", "a")
            
            #pretty print matrix 
            end = time.time() 
            readable_time = time.ctime(end)
            string = str(readable_time)
            string += ("\n[x=" + str(self.x_position)) 
            string += (", y=" + str(self.y_position))
            string += (", z=" + str(self.z_position) + "]\n\n")
            
            #write into file 
            f.write(string)  
            
            #close file 
            f.close() 
        except Exception as e:
            print(str(e) + "\nFile not written")
            
    def choose_random_starting_position(self):
        #choose random number between 0 and 1
        rand = random.uniform(0, 1)
        
        if(rand <= (1.0/4.0)):
            #straight line going into left curve 
            self.x_position = 0.4132014349
            self.y_position = -6.22487658223
            self.z_position = -0.0298790967155
        elif(rand > (1.0/4.0) and rand <= (2.0/4.0)):
            #sharp left curve 
            self.x_position = 0.930205421421
            self.y_position = -5.77364575559
            self.z_position = -0.0301045554742
        elif(rand > (2.0/4.0) and rand <= (3.0/4.0)):
            #sharp right curve 
            self.x_position = 1.1291257432
            self.y_position = -3.37940826549
            self.z_position = -0.0298815752691
        else:
            #straight line going into right curve 
            self.x_position = 0.4132014349
            self.y_position = -2.89940826549
            self.z_position = -0.0298790967155
         
    #if user pressed ctrl+c --> stop the robot
    def shutdown(self):
        print("Stopping")  
        #publish
        msg = Twist()
        vel_msg = self.stop(msg)
        self.velocity_publisher.publish(vel_msg)
        
        end = time.time() 
        total = end - self.start
        minutes = total / 60.0 
        speed = self.calc_average_speed()
        distance = speed * total 
        print("Total time = " + str(total) + " seconds = " + str(minutes) + " minutes")
        print("Distance = " + str(distance) + " meters" + " (ca. " + str(speed) + " m/s)")
        print("Average speed was = " + str(speed))
                    
        #save q matrix and records for later 
        self.bot.save_q_matrix(self.start, speed, distance)


    def reset_environment(self):
        #turn image to grayscale
        #self.my_img = self.imgHelper.segmentation(self.my_img)
    
        '''
        #put robot back to starting point
        self.set_position(self.x_position, self.y_position, self.z_position)
        self.x_position = 0.997423683876
        self.y_position = -5.77804235333
        self.z_position = -0.0301313744646
        self.set_position(self.x_position, self.y_position, self.z_position)
        '''
        
        #set speed back to initial values 
        self.biggest = self.inital_biggest
        self.big = self.initial_big
        self.middle = self.initial_middle
        self.small = self.initial_small
        self.smallest = self.initial_smallest          

    def epsilon_greedy(self, e):
        #random number 
        exploration_rate_threshold = random.uniform(0, 1)
        
        if(exploration_rate_threshold < e):
            #explore
            return True
        else:
            #exploit
            return False 
    
    def step(self, bot, action, last_state, msg):
        #execute action 
        vel =self.execute_action(action, msg)
        
        #get new state 
        new_state = bot.get_state(self.my_img, self.all_speeds, self.max_speed, self.min_speed)
        done = False 
        if(new_state == self.lost_line):
            #line is lost, episode has to end 
            #print("State is terminal")
            done = True 
        
        #get reward 
        reward = bot.calculate_reward(last_state, action)
        
        return new_state, reward, done, vel
        
    def execute_action(self, action, msg):
        #execute action
        vel = Twist()
        directions = {
            0: self.sharp_left,
            1: self.left,
            2: self.slightly_left,
            3: self.forward,
            4: self.slightly_right,
            5: self.right,
            6: self.sharp_right,
            7: self.slower,
            8: self.stop
            }
        function = directions.get(action)
        vel = function(msg)
        #print("X speed = " + str(vel.linear.x))
        #print("Y speed = " + str(vel.linear.y))
        #self.vel_msg = vel
        self.save_speed(msg)
        #publish  
        self.velocity_publisher.publish(vel)

        return vel
    
    #puts current speed in a list 
    def save_speed(self, msg):
        speed = (msg.linear.x + msg.linear.y) / 2.0
        print("Current speed = " + str(speed))
        self.all_speeds.append(speed)
        
    #calculates average speed of the robot 
    def calc_average_speed(self):
        avg_speed = 0
        for i in range(len(self.all_speeds)):
            avg_speed += self.all_speeds[i]
            
        avg_speed = (avg_speed / float((len(self.all_speeds))))
        return avg_speed 
    
    def reinf_main(self):
        rospy.on_shutdown(self.shutdown)
        self.start = time.time()
        
        episodes = 2000
        max_steps_per_episode = 100
        episode_counter = 0
        gamma = 0.95
        alpha = 0.8
        
        exploration_rate = 1
        decay_rate = 0.001
        min_exploration_rate = 0.01
        max_exploration_rate = 1
        
        all_rewards = []

        #check if robot is driving without learning for the first time
        firsttime = True

        #variable to save last driving message
        last_msg = Twist()
        last_msg.linear.x = self.middle
        last_msg.linear.y = self.middle
        last_msg.linear.z = 0.0
        last_msg.angular.x = 0.0
        last_msg.angular.y = 0.0
        last_msg.angular.z = 0.0
        
        #start at random point 
        self.choose_random_starting_position()
        self.set_position(self.x_position, self.y_position, self.z_position)
                
        try:        
            rate = rospy.Rate(50)
            while not rospy.is_shutdown():  
            #ROS main loop and outer reinforcement learning loop at the same time 
                if(self.flag): 
                #only do stuff if a new image is ready 
                          
                    if(episode_counter <= episodes):
                    #start episode 
                    #do reinforcement learning if not all episodes done
                    
                        #at start of each new episode 
                        #put robot back to starting position and set speed to initial values 
                        self.reset_environment()
                        
                        #keep track of if episode is done 
                        done = False 
                        
                        #no rewards at the beginning for the current episode  
                        rewards_current_episode = 0                        
                                                
                        #get current state 
                        curr_state = self.bot.get_state(self.my_img, self.all_speeds, self.max_speed, self.min_speed)
                        #print("\nState = " + str(curr_state))
                    
                        for i in range(max_steps_per_episode):
                        #try to reach goal (stay on line)
                            
                            if(self.epsilon_greedy(e=exploration_rate)):
                            #explore
                                print("Exploring")
                                #do the actual learning 
                                action = self.bot.explore(self.my_img)
                            else:
                            #exploit 
                                print("Exploiting")
                                #use q-matrix, but still update its' values 
                                action = self.bot.exploit(self.my_img, curr_state)
                                
                            #take the action 
                            new_state, reward, done, last_msg = self.step(self.bot, action, curr_state, last_msg)
                            #print("done = " + str(done))
                            
                            #update q-table
                            self.bot.update_q_table(curr_state, action, alpha, reward, gamma, new_state)
                            
                            #transition to new step 
                            curr_state = new_state 
                            rewards_current_episode += reward 
                            
                            #print which action is taken
                            #print(self.action_strings.get(action))
                                                        
                            #if terminal state (lost line) is reached, get out of for loop 
                            if done:  
                                #print("Done!")
                                #self.set_position(self.x_position, self.y_position, self.z_position)
                                last_msg = self.execute_action(self.stop_action, last_msg) #stop robot and put it back to starting position
                                break
                            
                        #at the end of each episode 
                        #decay the exploration rate 
                        # Exploration rate decay
                        exploration_rate = min_exploration_rate + \
                            (max_exploration_rate - min_exploration_rate) * np.exp(-decay_rate*episode_counter)   
                        #increase episode counter 
                        episode_counter += 1   
                        
                        #add rewards to list 
                        all_rewards.append(rewards_current_episode)
                    #end episode        
                    
                    else:
                        #print("Driving")
                        #drive with the filled q-matrix, but do NOT update its' values anymore
                        if(firsttime):
                            self.reset_environment()
                            firsttime = False
                        action = self.bot.drive(self.my_img, self.all_speeds, self.max_speed, self.min_speed)
                        #print which action is taken
                        print(self.action_strings.get(action))
                        self.execute_action(action)
                        
                    #set flag back to false to wait for a new image
                    self.flag = False 
                        
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
    
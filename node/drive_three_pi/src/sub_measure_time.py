#!/usr/bin/env python

import rospy

from std_msgs.msg import String
from random import randint


def time_callback(msg):
    rospy.loginfo(msg.data)

if __name__=='__main__':
    rospy.init_node('subscriber_measure_time', anonymous=True)
   
    sub=rospy.Subscriber('/sub_measure_time', String, time_callback)
    rospy.spin()
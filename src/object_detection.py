#!user/bin/env python3

import rospy
from sensor_msgs.msg import Image


if __name__ == '__main__':
    rospy.init_node('object_detector')
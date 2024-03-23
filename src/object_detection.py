#!user/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ultralytics import YOLO
from cv_bridge import CvBridge


if __name__ == '__main__':
    rospy.init_node('object_detector')
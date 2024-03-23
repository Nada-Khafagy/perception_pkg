#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class ImageProcessor:
    def __init__(self):
        rospy.init_node('image_processor', anonymous=True)
        self.image_pub = rospy.Publisher('processed_image', Image, queue_size=20)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/image', Image, self.image_callback)
    
    def image_callback(self, msg):
        rospy.loginfo('Received an image!')
        ros_image = None
        #For now i am testing the image processing by converting the image to grayscale
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            img_gray=cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            ros_image = self.bridge.cv2_to_imgmsg(img_gray, 'mono8')
            rospy.loginfo("Image is processed!")
        except CvBridgeError as e:
            rospy.loginfo("Error in processing the image!")
            rospy.logerr(e)


        self.image_pub.publish(ros_image)
        rospy.loginfo("Image is published!")

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == '__main__':
    try:
        image_node = ImageProcessor()
        image_node.run()
    except rospy.ROSInterruptException:
        pass
    

    
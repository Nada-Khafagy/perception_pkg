#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch
from perception_pkg.msg import bounding_box,bounding_box_array
from ultralytics.utils.plotting import Annotator, colors, save_one_box #this looks better, future me shoould handle this


import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

confidence = 0.8
class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/image', Image, self.detect_callback)
        self.image_pub = rospy.Publisher('/detected_image', Image, queue_size=20)
        self.bb_pub = rospy.Publisher('/bounding_boxes', bounding_box_array, queue_size=20)
        self.cv2_img = None
        self.model = torch.hub.load(str(ROOT.parents[0].resolve())+'/yolov5', 'custom', path= str(ROOT.parents[0].resolve())+ '/weights/best.pt', source='local') 


    def detect_callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.cv2_img = cv2_img
        except CvBridgeError as e:
            rospy.loginfo("Error in converting to cv2 image!")
            rospy.logerr(e)
        
        
    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            bbs_msg = bounding_box_array() #bb is bounding box
            bbs_msg.Header.stamp = rospy.Time.now()
            bbs_msg.Header.frame_id = 'bounding_boxes'
            bbs_msg.yolov8_boxes = []
            

            bounding_boxes = self.model(self.cv2_img).pandas().xyxy[0]
            box_color = (0, 255, 0)  # Green color
            text_color = (255, 255, 255)  # White color
            
            ros_image = None
            
            if bounding_boxes is not None:  
                for index, row in bounding_boxes.iterrows():
                    bb_msg = bounding_box()
                    
                    if row['confidence'] < confidence:
                        continue
                    
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    
                    cv2.rectangle(self.cv2_img, (x1, y1), (x2, y2), box_color, thickness=2)
                    label = f"{row['name']} (Confidence: {row['confidence']:.2f})"
                    cv2.putText(self.cv2_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness=2)
                    bb_msg.calss_name = row['name']
                    bb_msg.confidence = row['confidence']
                    bb_msg.xmin = x1
                    bb_msg.ymin = y1
                    bb_msg.xmax = x2
                    bb_msg.ymax = y2
                    
                    if bb_msg.calss_name != "":
                        bbs_msg.yolov8_boxes.append(bb_msg)
                        rospy.loginfo(f"{row['name']} is detected! with confidence {row['confidence']:.2f}")
                    
                    
                ros_image = self.bridge.cv2_to_imgmsg(self.cv2_img, 'bgr8')
                self.image_pub.publish(ros_image)
                self.bb_pub.publish(bbs_msg)
                
            rate.sleep()


if __name__ == '__main__':
    try:
        image_node = ObjectDetector()
        image_node.run()
    except rospy.ROSInterruptException:
        pass
    

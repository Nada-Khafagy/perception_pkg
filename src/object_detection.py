#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch
from perception_pkg.msg import bounding_box,bounding_box_array
import os
import sys
from pathlib import Path
import torchvision

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5" # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/image', Image, self.detect_callback)
        self.image_pub = rospy.Publisher('/detected_image', Image, queue_size=20)
        self.bb_pub = rospy.Publisher('/bounding_boxes', bounding_box_array, queue_size=20)
        pkg_path = str(FILE.parents[0].parents[0].resolve())
        model_path = pkg_path + rospy.get_param('~model_path')
        weights_path  = pkg_path + rospy.get_param('~weights_path')
        self.model = torch.hub.load(model_path, 'custom', path= weights_path, source='local') 
        self.conf = rospy.get_param("~confidence_threshold")  # NMS confidence threshold
        self.iou = rospy.get_param("~iou_threshold")  # NMS IoU threshold
            
        self.cv2_img = None
        self.ros_image = None

    def detect_callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.cv2_img = cv2_img
            self.new_image_received = True
        except CvBridgeError as e:
            rospy.loginfo("Error in converting to cv2 image!")
            rospy.logerr(e)
        
        
    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.cv2_img is not None and self.new_image_received: 
                self.new_image_received = False
                bbs_msg = bounding_box_array() #bb is bounding box
                bbs_msg.Header.stamp = rospy.Time.now()
                bbs_msg.Header.frame_id = 'bounding_boxes'
                bbs_msg.bbs_array = []
                bounding_boxes = self.model(self.cv2_img).pandas().xyxy[0]
                box_color = (0, 255, 0)  # Green color
                text_color = (255, 255, 255)  # White color
                if bounding_boxes is not None and not bounding_boxes.empty:
                    for _, row in bounding_boxes.iterrows():
                        bb_msg = bounding_box()
                        if row['confidence'] < self.conf:
                            continue
                        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                        
                        cv2.rectangle(self.cv2_img, (x1, y1), (x2, y2), box_color, thickness=2)
                        label = f"{row['name']} (Confidence: {row['confidence']:.2f})"
                        cv2.putText(self.cv2_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, thickness=2)
                        bb_msg.class_name = row['name']
                        bb_msg.confidence = row['confidence']
                        bb_msg.xmin = x1
                        bb_msg.ymin = y1
                        bb_msg.xmax = x2
                        bb_msg.ymax = y2
                        
                        if bb_msg.class_name != "":
                            bbs_msg.bbs_array.append(bb_msg)
                            #rospy.loginfo(f"{row['name']} is detected! with confidence {row['confidence']:.2f}")
                        self.bb_pub.publish(bbs_msg)   
                self.ros_image = self.bridge.cv2_to_imgmsg(self.cv2_img, 'bgr8')
                self.image_pub.publish(self.ros_image)
                    
            rate.sleep()


if __name__ == '__main__':
    try:
        image_node = ObjectDetector()
        image_node.run()
    except rospy.ROSInterruptException:
        pass
    

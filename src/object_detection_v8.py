#!/usr/bin/env python3
from typing import List, Dict
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch
from perception_pkg.msg import bounding_box,bounding_box_array
import os
import sys
from pathlib import Path
from ultralytics import YOLO

from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes



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
        
        self.model = self._initialize_model()
        
        self.conf = rospy.get_param("~confidence_threshold", default="0.5")  # NMS confidence threshold
        self.iou = rospy.get_param("~iou_threshold", default="0.6")  # NMS IoU threshold
            
        self.cv2_img = None
        self.ros_image = None
        
    def _initialize_model(self):    
        pkg_path = str(FILE.parents[0].parents[0].resolve())
        weights_path = pkg_path + rospy.get_param('~weights_path', default="/model/bestv8n.pt")
        #model_path = pkg_path + rospy.get_param('~model_path', default="/src/yolov5")
        return YOLO(weights_path)

    def detect_callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.cv2_img = cv2_img
            self.new_image_received = True
            
        except CvBridgeError as e:
            rospy.loginfo("Error in converting to cv2 image!")
            rospy.logerr(e)
        
    def publish_image_and_bounding_boxes(self, bbs_msg):
        self.ros_image = self.bridge.cv2_to_imgmsg(self.cv2_img, 'bgr8')
        self.image_pub.publish(self.ros_image)
        self.bb_pub.publish(bbs_msg)
    
    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.cv2_img is not None and self.new_image_received: 
                self.new_image_received = False
                bbs_msg = self.detect_objects()
                if bbs_msg:
                    self.publish_image_and_bounding_boxes(bbs_msg)
            rate.sleep()
            
    def detect_objects(self):
        bbs_msg = bounding_box_array()
        bbs_msg.Header.stamp = rospy.Time.now()
        bbs_msg.Header.frame_id = 'bounding_boxes'
        
        results: Results
        bounding_boxes: Boxes
        model_results = self.model(self.cv2_img, conf=self.conf, iou=self.iou)
        results = model_results[0]
        
        bounding_boxes = results.boxes
        #box_color = (0, 255, 0)  # Green color
        #text_color = (255, 255, 255)  # White color
        
        if bounding_boxes is not None:
            for bbox in bounding_boxes:
                #if bbox.conf.item() < self.conf:
                #    continue
                bb_msg = self.create_bounding_box_message(bbox, results.names)
                self.cv2_img = results.plot()
                if bb_msg.calss_name != "":
                    bbs_msg.yolov8_boxes.append(bb_msg)
        return bbs_msg if bbs_msg.yolov8_boxes else None

    def create_bounding_box_message(self, bbox, class_names):
        bb_msg = bounding_box()
        half_bbox = bbox.xyxy[0].half().tolist()
        x1, y1, x2, y2 = half_bbox[0:4]

        bb_msg.calss_name = class_names[int(bbox.cls)]
        bb_msg.confidence = bbox.conf.item()

        bb_msg.xmin = x1
        bb_msg.ymin = y1
        bb_msg.xmax = x2
        bb_msg.ymax = y2

        return bb_msg

if __name__ == '__main__':
    try:
        image_node = ObjectDetector()
        image_node.run()
    except rospy.ROSInterruptException:
        pass
    

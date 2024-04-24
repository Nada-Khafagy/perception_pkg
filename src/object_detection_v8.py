#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from perception_pkg.msg import bounding_box,bounding_box_array
from geometry_msgs.msg import Pose2D

import os
import sys
import cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov8" # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)
        self.bridge = CvBridge()

        self.conf = rospy.get_param("~confidence_threshold", default="0.5")  # NMS confidence threshold
        self.iou = rospy.get_param("~iou_threshold", default="0.6")  # NMS IoU threshold
        self.new_image_received = False
        self.new_depth_received = False
        self.depth_data = None
        self.cv2_img = None
        self.ros_image = None
        pkg_path = str(FILE.parents[0].parents[0].resolve())
        weights_path = pkg_path + rospy.get_param('~weights_path', default="/model/bestv8n.pt")
        #model_path = pkg_path + rospy.get_param('~model_path', default="/src/yolov5")

        self.model = YOLO(weights_path, verbose=True)
        self.is_shutdown = False
        self.old_image = None
        
        self.image_sub = rospy.Subscriber('/image', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/image_depth', Image, self.depth_callback)
        
        self.image_pub = rospy.Publisher('/detected_image', Image, queue_size=1)
        self.bb_pub = rospy.Publisher('/bounding_boxes', bounding_box_array, queue_size=1)
        
        
       

    def image_callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv2_img=cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
            
            self.cv2_img = cv2_img
            self.new_image_received = True
            self.run()
            
        except CvBridgeError as e:
            rospy.loginfo("Error in converting to cv2 image!")
            rospy.logerr(e)
         
    def depth_callback(self, msg):   
        self.depth_data = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))

        self.new_depth_received = True
        self.run()
            
        
            
    def run(self):
        #rate = rospy.Rate(20)
        if self.cv2_img is not None and self.new_image_received and self.new_depth_received : 
            if self.old_image is not None and (self.cv2_img is self.old_image):
                rospy.loginfo(f"detected on same image!!, self.new_image_received: {self.new_image_received}, self.new_depth_received: {self.new_depth_received}")
            self.new_image_received = False
            self.new_depth_received = False
            bbs_msg = self.detect_objects()
            self.ros_image = self.bridge.cv2_to_imgmsg(self.cv2_img, 'bgr8')
            self.image_pub.publish(self.ros_image)
            if bbs_msg is not None:
                self.bb_pub.publish(bbs_msg)
            self.old_image = self.cv2_img
            
            
        #rate.sleep()
            
    def detect_objects(self):
        bbs_msg = bounding_box_array()
        bbs_msg.Header.stamp = rospy.Time.now()
        bbs_msg.Header.frame_id = 'bounding_boxes'
        
        results: Results
        bounding_boxes: Boxes
        model_results = self.model(self.cv2_img, conf=self.conf, iou=self.iou, verbose=False)
        results = model_results[0]
        
        bounding_boxes = results.boxes

        
        if bounding_boxes is not None:
            for bbox in bounding_boxes:
                if bbox.conf.item() < self.conf:
                    continue
                bb_msg = self.create_bounding_box_message(bbox, results.names)
                self.cv2_img = results.plot()
                if bb_msg.class_name != "":
                    bbs_msg.bbs_array.append(bb_msg)
        return bbs_msg if bbs_msg.bbs_array else None
    
    def create_bounding_box_message(self, bbox, class_names):
        bb_msg = bounding_box()
        half_bbox = bbox.xyxy[0].half().tolist()
        x1, y1, x2, y2 = half_bbox[0:4]

        robot_frame = []
        camera_wrt_robot_frame = []
        
        bb_msg.class_name = class_names[int(bbox.cls)]
        bb_msg.confidence = bbox.conf.item()

        #x_pix = ((x1 + x2) / 2) 
        #y_pix = ((y1 + y2) / 2)
        #x_pos = (x_pix - cx_d) * self.depth_data(x_pix,y_pix) / fx_d
        #y_pos = (y_pix - cy_d) * self.depth_data(x_pix,y_pix) / fy_d
        #z_pos = self.depth_data(x_pix,y_pix)
        
        bb_msg.centeroid.theta = 0.0 #bb are not rotated
        
        

        return bb_msg
    


if __name__ == '__main__':
    try:
        image_node = ObjectDetector()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down...")
        image_node.is_shutdown = True
        image_node.cleanup()
        rospy.signal_shutdown("Ctrl+C was pressed")
    

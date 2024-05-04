#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from perception_pkg.msg import bounding_box,bounding_box_array
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3

import os
import sys
import cv2
import torch
from copy import deepcopy
from pathlib import Path
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.utils.plotting import colors
from ultralytics.utils.plotting import Annotator

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)
        self.bridge = CvBridge()
        
        self.conf = rospy.get_param("~confidence_threshold", default="0.5")  # NMS confidence threshold
        self.iou = rospy.get_param("~iou_threshold", default="0.6")  # NMS IoU threshold
        img_topic_name = rospy.get_param("~image_rgb_topic_name", default="/image")
        depth_topic_name = rospy.get_param("~image_depth_topic_name", default="/image_depth")
        odom_topic_name = rospy.get_param("~odom_topic_name", default="/odom")
        
        self.use_depth = rospy.get_param("~use_depth", default=False)
        self.use_tracking = rospy.get_param("~use_tracking", default=False)
        self.use_encoder = rospy.get_param("~use_encoder", default=False)
        self.perspective_angle = rospy.get_param("~prespective_angle_x", default=85.0)
        
        self.res_x = rospy.get_param("~resolution_x", default=960)
        self.res_y = rospy.get_param("~resolution_y", default=480)
        
        self.person_num = rospy.get_param("~person_num", default=0)
        self.car_num = rospy.get_param("~car_num", default=0)
        self.cone_num = rospy.get_param("~cone_num", default=0)
        
        self.new_image_received = False
        self.new_depth_received = False
        self.depth_data = None
        self.base_map_tf = None
        
        self.raw_cv2_img = None
        self.output_cv2_img = None
        self.raw_ros_image = None
        self.ros_image = None
        self.old_ros_img = None #to make sure the same image is not detected on twice 

        self.is_shutdown = False
        #get absolute path sometimes it causes errors with the relative path when loading yolo
        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[0] / "yolov8" # YOLOv5 root directory
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))  # add ROOT to PATH
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

        pkg_path = str(FILE.parents[0].parents[0].resolve())
        weights_path = pkg_path + rospy.get_param('~weights_path', default="/model/best.pt")
        
        self.model = YOLO(weights_path, verbose=True)
        self.calc_intrinsic_camera_info()   
        
        #intaloze puplishers and subscribers
        self.image_pub = rospy.Publisher('/detected_image', Image, queue_size=1)
        self.bb_pub = rospy.Publisher('/bounding_boxes', bounding_box_array, queue_size=1)
        #self.centroid_pub = rospy.Publisher('/centroid', PointStamped, queue_size=1)
        self.image_sub = rospy.Subscriber(img_topic_name, Image, self.image_callback)
        self.depth_sub = rospy.Subscriber(depth_topic_name, Image, self.depth_callback)

        if self.use_encoder:
            self.encoder_sub = rospy.Subscriber('/vehicle_position', Vector3, self.encoder_callback)
        else:
            self.odom_sub = rospy.Subscriber(odom_topic_name, Odometry, self.odom_callback)
        
        
    def image_callback(self, msg):
        try:
            self.raw_ros_image = msg
            #convert ros image to cv2 image
            self.raw_cv2_img  = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.new_image_received = True
            self.run()
        except CvBridgeError as e:
            rospy.logerr(e)
            rospy.loginfo("Shutting down...")
            self.is_shutdown = True
            self.cleanup()
            rospy.signal_shutdown("Error in converting to cv2 image!")
         
    def depth_callback(self, msg):   
        self.depth_data = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
        #self.depth_data = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.depth_data = np.transpose(self.depth_data)
        self.new_depth_received = True
        self.run()
        
    def odom_callback(self, msg):
        # construct transformation matrix from odom data
        msg: Odometry
        #translation components
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        self.base_map_tf = np.array([[1, 0, 0, x],
                                        [0, 1, 0, y],
                                        [0, 0, 1, z],
                                        [0, 0, 0, 1]])
        #print(f"base_map_tf: {self.base_map_tf}")

    
    def encoder_callback(self, msg):
        # construct transformation matrix from odom data
        msg: Vector3
        #translation components
        x = msg.x
        y = msg.y
        z = msg.z
        self.base_map_tf = np.array([[1, 0, 0, x],
                                    [0, 1, 0, y],
                                    [0, 0, 1, 0.86885],
                                    [0, 0, 0, 1]])
        #print(f"base_map_tf: {self.base_map_tf}")
        
           
    def calc_intrinsic_camera_info(self):
        # get horizontal and vertical field of view
        h_fov =  np.deg2rad(self.perspective_angle)
        v_fov = 2 * np.arctan((self.res_y * np.tan(h_fov/2)) / (self.res_x) )   
        #Compute focal length (fx, fy) based on field of view (fov) and resolution
        self.fx_d = self.res_x / (2 * np.tan(h_fov / 2))
        self.fy_d = self.res_y / (2 * np.tan(v_fov / 2))
        #Principal point (cx, cy) is assumed to be at the center of the image   
        self.cx_d = self.res_x / 2
        self.cy_d = self.res_y / 2
                
    def run(self):
        # run only if new image and depth data is received
        if self.raw_cv2_img is not None and self.new_image_received and self.new_depth_received and not self.base_map_tf is None: 
            if self.old_ros_img is not None and (self.raw_ros_image.header == self.old_ros_img.header):
                rospy.loginfo(f"called on same image!!")   
            else:
                self.new_image_received = False
                self.new_depth_received = False
                
                #box_results = self.draw_boxes()

                bbs_msg = self.create_bounding_boxes()
                self.ros_image = self.bridge.cv2_to_imgmsg(self.output_cv2_img, 'bgr8')
                              
                #publish only if bounding boxes are detected
                if bbs_msg is not None:
                    #rospy.loginfo(f"Publishing {len(bbs_msg.bbs_array)} bounding boxes")
                    self.bb_pub.publish(bbs_msg)
            
            self.image_pub.publish(self.ros_image)
            self.old_ros_img = self.raw_ros_image
            

    def create_bounding_boxes(self):
        model_results = self.model(self.raw_cv2_img, conf=self.conf, iou=self.iou, verbose=False)
        results = model_results[0]
        results: Results
        names = results.names
        is_obb = results.obb is not None
        pred_boxes = results.obb if is_obb else results.boxes
        pred_probs, show_probs = results.probs, True
        annotator = Annotator(
            deepcopy(results.orig_img) ,
            pil = ((pred_probs is not None and show_probs)),  # Classify tasks default to pil=True
            example=names,
        )
                
        if pred_boxes is not None:
            bbs_msg = bounding_box_array()
            bbs_msg.header.stamp = rospy.Time.now()
            bbs_msg.header.frame_id = 'bounding_boxes'
            person_count = 0
            car_count = 0
            cone_count = 0
            person_precsion = 1
            car_precsion = 1
            cone_precsion = 1
            
            for bbox in reversed(pred_boxes):
                c, conf, id = int(bbox.cls), float(bbox.conf) , None if bbox.id is None else int(bbox.id.item())
                name = ("" if id is None else f"id:{id} ") + names[c]
                text = (f"{name} {conf:.2f}" if conf else name) #you can change it here to any other custom texts
                box = bbox.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else bbox.xyxy.squeeze()
                annotator.box_label(box, text, color=colors(c, True), rotated=is_obb)
                
                # Draw centroid
                centroid_x = (box[0] + box[2]) / 2
                centroid_y = (box[1] + box[3]) / 2
                centroid = (int(centroid_x), int(centroid_y))
                annotator.draw_specific_points([centroid],indices=[0])
                
                if name == "person":
                    person_count += 1
                elif name == "car":
                    car_count += 1
                elif name == "traffic cone":
                    cone_count += 1
                
                if self.use_depth:
                    bb_msg = self.create_bounding_box_msg(bbox, results.names)
                    if bb_msg is not None:
                        bbs_msg.bbs_array.append(bb_msg)
                # Plot Classify results
            if pred_probs is not None and show_probs:
                text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
                x = round(self.orig_shape[0] * 0.03)
                annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors
       
            self.output_cv2_img = annotator.result()
            if self.person_num != 0: 
                person_precsion = person_count / self.person_num
            if self.car_num != 0:
                car_precsion = car_count / self.car_num
            if self.cone_num != 0:
                cone_precsion = cone_count / self.cone_num
            rospy.loginfo(f"Person percision: {person_precsion}, Car percision: {car_precsion}, Cone percision: {cone_precsion}")
        else:
            bbs_msg = None
            self.output_cv2_img = self.raw_cv2_img
        return bbs_msg if bbs_msg.bbs_array else None
    
    #WIP
    def track_objects(self):
        pass
    
    
    def create_bounding_box_msg(self, bbox, class_names):
        bb_msg = bounding_box()
        bb_msg.header.stamp = rospy.Time.now()
        bb_msg.header.frame_id = 'bounding_box'
        #the box coordinates in (left, top, right, bottom) format
        
        half_bbox = bbox.xyxy[0].half().tolist()
        x1, y1, x2, y2 = half_bbox[0:4]
        bb_msg.class_name = class_names[int(bbox.cls)]
        bb_msg.confidence = bbox.conf.item()
        
        #get 3D position of the centroid of the bounding box
        x = int((x1+x2)/2)
        y = int((y1+y2)/2)
        camera_point = self.get_point_wrt_camera_frame(x, y)
        #self.centroid_pub.publish(camera_point)
        base_link_point = self.get_point_wrt_base_link(camera_point)
        map_point = self.get_point_wrt_map(base_link_point)
        
        #print(f"map point: {map_point} \n of class: {bb_msg.class_name}")
        bb_msg.centroid.header.stamp = rospy.Time.now()
        bb_msg.centroid.header.frame_id = 'map'
        if map_point is None:
            return None
        bb_msg.centroid.point.x = map_point.point.x
        bb_msg.centroid.point.y = map_point.point.y
        bb_msg.centroid.point.z = map_point.point.z
        
        
        #bounding box width and height in world coordinates
        point_TL, point_TR, point_BL, point_BR, width, length = self.get_width_length_in_world_coordinates(x1, y1, x2, y2, bb_msg.centroid.point.z)
        #bb_msg.TL = point_TL
        #bb_msg.TR = point_TR
        #bb_msg.BL = point_BL
        #bb_msg.BR = point_BR 
        
        bb_msg.width = width
        bb_msg.length = length
           
        return bb_msg
    

    def get_point_wrt_camera_frame(self, x, y):
        
        x_pix = max(min(int(x), self.raw_cv2_img.shape[1]-1), 0)
        y_pix = max(min(int(y), self.raw_cv2_img.shape[0]-1), 0)
        #get 3D position of the centeroid of the bounding box
        #the minus is to map the origin to the center of the image 
        z_pos =  self.depth_data[int(x_pix),int(y_pix)]
        x_pos = ((x_pix - self.cx_d) * z_pos) / self.fx_d
        y_pos = ((y_pix - self.cy_d) * z_pos) / self.fy_d

        #create a point in camera frame
        camera_point = PointStamped()
        camera_point.header.stamp = rospy.Time.now()
        camera_point.header.frame_id = "camera_frame"
        camera_point.point.x = x_pos
        camera_point.point.y = y_pos
        camera_point.point.z = z_pos
        
        return camera_point
         
    def get_point_wrt_base_link(self, camera_point):
        if camera_point is None:
            rospy.loginfo("Camera point is None")
            return None

        #rotation components
        #rx = -0.7372773368101609
        #ry = 2.6837016371639234e-16
        #rz = -2.463260808134392e-16
        #rw = 0.67559020761562

        #rotation_matrix = np.array([
        #    [1 - 2*(ry**2 + rz**2), 2*(rx*ry - rw*rz), 2*(rx*rz + rw*ry)],
        #    [2*(rx*ry + rw*rz), 1 - 2*(rx**2 + rz**2), 2*(ry*rz - rw*rx)],
        #    [2*(rx*rz - rw*ry), 2*(ry*rz + rw*rx), 1 - 2*(rx**2 + ry**2)]
        #])
        
        #construct rotation matrix
        #rotation around the x-axes but not exactly 90deg, can be edited from coppeliasim but leave it for now
        rotation_matrix = np.array([[ 1.00000000e+00, -6.28955030e-17,  7.25837783e-16],
                                    [-7.28557455e-16, -8.71557427e-02,  9.96194698e-01],
                                    [ 6.04764452e-19, -9.96194698e-01, -8.71557427e-02]])
        #translation components
        tx = 1.9567680809018344e-16
        ty = 0.3
        tz = 0.6639999958276729

        # Construct translation vector
        translation_vector = np.array([tx, ty, tz])

        # Construct transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector
        
        camera_point: PointStamped
        camera_pt = np.array([camera_point.point.x, camera_point.point.y, camera_point.point.z, 1])
        base_link_pt = np.dot(transformation_matrix, camera_pt)
        
        base_link_point = PointStamped()
        base_link_point.header.stamp = rospy.Time.now()
        base_link_point.header.frame_id = "base_link"
        base_link_point.point.x = base_link_pt[0]
        base_link_point.point.y = base_link_pt[1]
        base_link_point.point.z = base_link_pt[2]
        
        return base_link_point
    
    def get_point_wrt_map(self, base_link_point):
        if base_link_point is None:
            rospy.loginfo("Base link point is None")
            return None
        base_link_pt = np.array([base_link_point.point.x, base_link_point.point.y, base_link_point.point.z, 1])
        map_pt = np.dot(self.base_map_tf, base_link_pt)
        
        map_point = PointStamped()
        map_point.header.stamp = rospy.Time.now()
        map_point.header.frame_id = "map"
        map_point.point.x = map_pt[0]
        map_point.point.y = map_pt[1]
        map_point.point.z = map_pt[2]
        
        return map_point
         
        
    def get_width_length_in_world_coordinates(self, x1, y1, x2, y2, z):
        #the box coordinates in (left, top, right, bottom) format
        #get 3D line points of the bounding box
        #Top left point
        camera_point1 = self.get_point_wrt_camera_frame(x1, y1)
        map_point1 = self.get_point_wrt_map(self.get_point_wrt_base_link(camera_point1))
        #Top right point
        camera_point2 = self.get_point_wrt_camera_frame(x2, y1)
        map_point2 = self.get_point_wrt_map(self.get_point_wrt_base_link(camera_point2))
        #Bottom left point
        camera_point3 = self.get_point_wrt_camera_frame(x1, y2)
        map_point3 = self.get_point_wrt_map(self.get_point_wrt_base_link(camera_point3))
        #Bottom right point
        camera_point4 = self.get_point_wrt_camera_frame(x2, y2)
        map_point4 = self.get_point_wrt_map(self.get_point_wrt_base_link(camera_point4))
        #use depth of the center to have the bounding box in the same plane
        point_TL = np.array([map_point1.point.x, map_point1.point.y, z])
        point_TR = np.array([map_point2.point.x, map_point2.point.y, z])
        point_BL = np.array([map_point3.point.x, map_point3.point.y, z])
        point_BR = np.array([map_point4.point.x, map_point3.point.y, z])
        # Compute the Euclidean distance to get the width and length in meters
        width = np.linalg.norm(point_TL - point_TR)
        length = np.linalg.norm(point_TL - point_BL)
        
        return point_TL, point_TR, point_BL, point_BR, width, length
    
                       
    def cleanup(self):
        self.image_sub.unregister()
        self.depth_sub.unregister()
        self.image_pub.unregister()
        self.odom_sub.unregister()
        self.bb_pub.unregister()
        
if __name__ == '__main__':
    try:
        image_node = ObjectDetector()
        rospy.spin() 
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
        image_node.is_shutdown = True
        image_node.cleanup()
        rospy.signal_shutdown("Ctrl+C was pressed")
    

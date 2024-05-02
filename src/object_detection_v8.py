#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from perception_pkg.msg import bounding_box,bounding_box_array
from geometry_msgs.msg import Point
from tf2_geometry_msgs import PointStamped
from tf2_geometry_msgs import do_transform_point
from sensor_msgs.msg import CameraInfo
import tf2_ros
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
        self.camera_info_recieved = False
        self.depth_data = None
        
        self.raw_cv2_img = None
        self.output_cv2_img = None
        self.raw_ros_image = None
        self.ros_image = None
        #to make sure the same image is not detected on twice
        self.old_ros_img = None


        self.is_shutdown = False
        pkg_path = str(FILE.parents[0].parents[0].resolve())
        weights_path = pkg_path + rospy.get_param('~weights_path', default="/model/bestv8nv2.pt")
        #model_path = pkg_path + rospy.get_param('~model_path', default="/src/yolov5")
        self.model = YOLO(weights_path, verbose=True)
        self.image_sub = rospy.Subscriber('/image', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/image_depth', Image, self.depth_callback)
        #to get K and camera info in order to map to 3D points
        #self.camera_sub = rospy.Subscriber('/camera_info',CameraInfo , self.camera_callback)
        # static info topic so get the info once
        self.camera_info_msg = rospy.wait_for_message('/camera_info',CameraInfo, timeout=None)
        self.image_pub = rospy.Publisher('/detected_image', Image, queue_size=1)
        self.bb_pub = rospy.Publisher('/bounding_boxes', bounding_box_array, queue_size=1)
        
    
    def cleanup(self):
        self.image_sub.unregister()
        self.depth_sub.unregister()
        self.image_pub.unregister()
        self.bb_pub.unregister()

    def image_callback(self, msg):
        try:
            self.raw_ros_image = msg
            #convert ros image to cv2 image
            self.raw_cv2_img  = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            #cv2_img=cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            # yolov8 expects 3 channel image
            #self.raw_cv2_img  = cv2.cvtColor(self.raw_cv2_img, cv2.COLOR_GRAY2RGB)   
            #signal that new image is received
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
        #rospy.loginfo('depth resolution:', self.depth_data.shape)
        self.new_depth_received = True
        self.run()
        
    
    def handle_camera_info(self):
        # save values nedded for calculation of 3D position from pixel position
        self.camera_info_recieved = True
        self.camera_K = np.array(self.camera_info_msg.K).reshape(3,3)
        self.camera_R = np.array(self.camera_info_msg.R).reshape(3,3)#assumed to be the identity matrix
        self.camera_P = np.array(self.camera_info_msg.P).reshape(3,4)
        # Intrinsic camera matrix for the raw (distorted) images.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]
        # Dimension pixels
        self.fx_d = self.camera_K[0,0]
        self.fy_d = self.camera_K[1,1]
        self.cx_d = self.camera_K[0,2]
        self.cy_d = self.camera_K[1,2]
        self.run()
                 
    def run(self):
        #rate = rospy.Rate(20)
        #while rospy.is_shutdown() is False and self.is_shutdown is False:
        # run only if new image and depth data is received
        #rospy.loginfo(f"self.new_image_received: {self.new_image_received}, self.new_depth_received: {self.new_depth_received}, self.camera_info_recieved: {self.camera_info_recieved}")
        if self.raw_cv2_img is not None and self.new_image_received and self.new_depth_received and self.camera_info_recieved: 
            if self.old_ros_img is not None and (self.raw_ros_image.header == self.old_ros_img.header):
                rospy.loginfo(f"detected on same image!!, self.new_image_received: {self.new_image_received}, self.new_depth_received: {self.new_depth_received}")
            self.new_image_received = False
            self.new_depth_received = False
            bbs_msg = self.detect_objects()
            self.ros_image = self.bridge.cv2_to_imgmsg(self.output_cv2_img, 'bgr8')
            self.image_pub.publish(self.ros_image)
            #publish only if bounding boxes are detected
            if bbs_msg is not None:
                #rospy.loginfo(f"Publishing {len(bbs_msg.bbs_array)} bounding boxes")
                self.bb_pub.publish(bbs_msg)
            self.old_ros_img = self.raw_ros_image
        #rate.sleep()
           
     
    def detect_objects(self):
        bbs_msg = bounding_box_array()
        bbs_msg.header.stamp = rospy.Time.now()
        bbs_msg.header.frame_id = 'bounding_boxes'  
        results: Results
        bounding_boxes: Boxes
        model_results = self.model(self.raw_cv2_img, conf=self.conf, iou=self.iou, verbose=False)
        results = model_results[0]    
        bounding_boxes = results.boxes
        self.output_cv2_img = results.plot()   
        if bounding_boxes is not None:
            for bbox in bounding_boxes:
                if bbox.conf.item() < self.conf:
                    continue
                bb_msg = self.create_bounding_box_message(bbox, results.names)
                if bb_msg is not None:
                    bbs_msg.bbs_array.append(bb_msg)
        return bbs_msg if bbs_msg.bbs_array else None
    
    #WIP
    def track_objects(self):
        pass
    
    
    def create_bounding_box_message(self, bbox, class_names):
        bb_msg = bounding_box()
        bb_msg.header.stamp = rospy.Time.now()
        bb_msg.header.frame_id = 'bounding_box'
        #the box coordinates in (left, top, right, bottom) format
        half_bbox = bbox.xyxy[0].half().tolist()
        x1, y1, x2, y2 = half_bbox[0:4]    
        bb_msg.class_name = class_names[int(bbox.cls)]
        bb_msg.confidence = bbox.conf.item()     
        #get 3D position of the centroid of the bounding box
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
        camera_point = self.get_point_wrt_camera_frame(cx, cy)
        
        base_link_point = self.get_point_wrt_base_link(camera_point)
        map_point = self.get_point_wrt_map(base_link_point)
        print(f"map point: {map_point} \n of class: {bb_msg.class_name}")
        #bb_msg.centroid.header.stamp = rospy.Time.now()
        #bb_msg.centroid.header.frame_id = 'centroid_of_'+str(bb_msg.class_name)
        bb_msg.centroid.x = map_point.point.x
        bb_msg.centroid.y = map_point.point.y
        bb_msg.centroid.z = map_point.point.z
        
        #bounding box width and height in world coordinates
        point_TL, point_TR, point_BL, point_BR, width, length = self.get_width_length_in_world_coordinates(x1, y1, x2, y2, bb_msg.centroid.z)
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
            return None
        # Transform the point from camera frame to map frame
        #since it is a rigid body transformation we can use the a hardcoded transformation matrix 
        #translation components
        tx = 1.9567680809018344e-16
        ty = 0.3
        tz = 0.6639999958276729

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
        rotation_matrix = np.array([[1,         0,             0],
                                    [0,   -0.087156,     0.99619],
                                    [0,    -0.99619 ,  -0.087156]])
        
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
            return None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        try:
            #Transform the point to base_link frame first
            #delay to make sure the transform is available 
            trans = self.tf_buffer.lookup_transform('base_link',
                                              'camera_frame',
                                              rospy.Time.now(),
                                              rospy.Duration(2.0))
            # returns a PointStamped ?
            map_point : PointStamped
            map_point = self.tf_buffer.transform(base_link_point, "map")
            return map_point
        except tf2_ros.ExtrapolationException as e:
            rospy.loginfo("Extrapolation exception: %s", e)
            return None
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException) as e:
            rospy.loginfo("Failed to transform point: %s", e)
            return None
        
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
    
        
if __name__ == '__main__':
    try:
        image_node = ObjectDetector()
        image_node.handle_camera_info()
        rospy.spin() 
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
        image_node.is_shutdown = True
        image_node.cleanup()
        rospy.signal_shutdown("Ctrl+C was pressed")
    

#!/usr/bin/env python3
import rospy
import numpy as np
import cv2 
from tf.transformations import quaternion_matrix
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from perception_pkg.msg import bounding_box,bounding_box_array
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from copy import deepcopy
from pathlib import Path
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import colors
from ultralytics.utils.plotting import Annotator

#get absolute path sometimes it causes errors with the relative path when loading yolo
FILE = Path(__file__).resolve()

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)
        self.bridge = CvBridge()

        # Get parameters from the ROS parameter server
        self.conf = rospy.get_param("~confidence_threshold", default="0.5")  # NMS confidence threshold
        self.iou = rospy.get_param("~iou_threshold", default="0.6")  # NMS IoU threshold
        img_topic_name = rospy.get_param("~image_rgb_topic_name", default="/image")
        depth_topic_name = rospy.get_param("~image_depth_topic_name", default="/image_depth")
        odom_topic_name = rospy.get_param("~odom_topic_name", default="/odom")
        self.use_depth = rospy.get_param("~use_depth", default=False)
        self.use_tracking = rospy.get_param("~use_tracking", default=False)
        self.use_less_classes = rospy.get_param("~use_less_classes", default=False)
        self.perspective_angle = rospy.get_param("~prespective_angle_x", default=85.0)
        self.res_x = rospy.get_param("~resolution_x", default=960)
        self.res_y = rospy.get_param("~resolution_y", default=480)
        # Initialize variables
        self.car_pos_recieved = False
        self.new_image_received = False
        self.new_depth_received = False
        self.depth_data = None
        self.base_map_tf = None
        self.raw_cv2_img = None
        self.output_cv2_img = None
        self.raw_ros_image = None
        self.ros_image = None
        self.old_ros_img = None
        self.is_shutdown = False
        self.compact_classes_names = ["car", "person", "traffic cone"] #hardcodded for now
        pkg_path = str(FILE.parents[0].parents[0].resolve())
        weights_path = pkg_path + rospy.get_param('~weights_path', default="/model/best.pt")       
        self.model = YOLO(weights_path, verbose=True)
        
        # Calculate camera intrinsic parameters
        self.calc_intrinsic_camera_info()         
        
        # Initialize puplishers and subscribers
        self.image_pub = rospy.Publisher('/detected_image', Image, queue_size=1)
        if self.use_depth:
            self.depth_sub = rospy.Subscriber(depth_topic_name, Image, self.depth_callback)
            self.bb_pub = rospy.Publisher('/object_detection/bounding_boxes', bounding_box_array, queue_size=1)
        #self.centroid_pub = rospy.Publisher('/centroid', PointStamped, queue_size=1)
        self.image_sub = rospy.Subscriber(img_topic_name, Image, self.image_callback)
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
        #rotation components
        q = msg.pose.pose.orientation
        q = [q.x, q.y, q.z, q.w]
        rotation_matrix = quaternion_matrix(q)
        # Construct translation vector
        translation_vector = np.array([x, y, z])
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix[:3, :3]
        transformation_matrix[:3, 3] = translation_vector
        self.base_map_tf = transformation_matrix
        self.car_pos_recieved = True

        
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
        if self.new_image_received and (not self.use_depth or self.new_depth_received) and self.car_pos_recieved: 
            if self.old_ros_img is not None and (self.raw_ros_image.header == self.old_ros_img.header):
                rospy.logdebug(f"called on same image!!")   
            else:
                self.new_image_received = False
                self.new_depth_received = False
                bbs_msg = self.create_bounding_boxes()
                #save image
                #cv2.imwrite(str(FILE.parents[0].resolve())+f"/city_{rospy.Time.now()}.jpg", self.output_cv2_img)
                self.ros_image = self.bridge.cv2_to_imgmsg(self.output_cv2_img, 'bgr8')             
                #publish only if bounding boxes are detected
                if bbs_msg is not None:
                    self.bb_pub.publish(bbs_msg)          
            self.image_pub.publish(self.ros_image)
            self.old_ros_img = self.raw_ros_image
            
    #modified version of predict in the ultralytics library
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
            pil = ((pred_probs is not None and show_probs)), 
            example=names,
        )            
        if pred_boxes is not None:
            if self.use_depth:
                bbs_msg = bounding_box_array()
                bbs_msg.header.stamp = rospy.Time.now()
                bbs_msg.header.frame_id = 'bounding_boxes'   
            for bbox in reversed(pred_boxes):
                c, conf, id = int(bbox.cls), float(bbox.conf) , None if bbox.id is None else int(bbox.id.item())
                name = ("" if id is None else f"id:{id} ") + names[c] 
                if self.use_less_classes and name not in self.compact_classes_names:
                    continue
                box = bbox.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else bbox.xyxy.squeeze()
                text = (f"{name} {conf:.2f}" if conf else name) #you can change it to any other custom text
                annotator.box_label(box, text, color=colors(c, True), rotated=is_obb)
                # Draw centroid
                centroid_x = (box[0] + box[2]) / 2
                centroid_y = (box[1] + box[3]) / 2
                centroid = (int(centroid_x), int(centroid_y))
                annotator.draw_specific_points([centroid],indices=[0])
                #only create msg with data in 3d if needed
                if self.use_depth:
                    bb_msg = self.create_bounding_box_msg(bbox, results.names)
                    if bb_msg is not None:
                        bbs_msg.bbs_array.append(bb_msg)

            self.output_cv2_img = annotator.result()
                
        else:
            bbs_msg = None
            self.output_cv2_img = self.raw_cv2_img
        if self.use_depth:
            return bbs_msg if bbs_msg.bbs_array else None
        else:
            return None

    #WIP
    def track_objects(self):
        pass

    def create_bounding_box_msg(self, bbox, class_names):
        bb_msg = bounding_box()
        #the box coordinates in (left, top, right, bottom) format
        half_bbox = bbox.xyxy[0].half().tolist()
        x1, y1, x2, y2 = half_bbox[0:4]
        bb_msg.class_name = class_names[int(bbox.cls)]
        bb_msg.confidence = bbox.conf.item()
        #get 3D position of the centroid of the bounding box
        x = int((x1+x2)/2)
        y = int((y1+y2)/2)
        camera_point = self.get_point_wrt_camera_frame(x, y)
        base_link_point = self.get_point_wrt_base_link(camera_point)
        map_point = self.get_point_wrt_map(base_link_point)
        width, length = self.get_width_length_in_world_coordinates(x1, y1, x2, y2, self.depth_data[x, y])

        if map_point is None:
            return None
        map_point : Point
        bb_msg.centroid.x = map_point.y
        bb_msg.centroid.y = -map_point.x
        bb_msg.centroid.z = map_point.z
        #bounding box width and height in world coordinates
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
        camera_point = Point()
        camera_point.x = x_pos
        camera_point.y = y_pos
        camera_point.z = z_pos
        return camera_point
         
    def get_point_wrt_base_link(self, camera_point: Point):
        if camera_point is None:
            rospy.loginfo("Camera point is None")
            return None
        #construct rotation matrix
        #rotation around the x-axes but not exactly 90deg, can be edited from coppeliasim but leave it for now
        
        rotation_matrix = np.array([[ 0.9999999999993465, -1.1432298515079376e-06, 1.0256571239377271e-13],
                                    [-2.992637125762684e-08, -0.026176948307961467, 0.9996573249755546],
                                    [ -1.142838092505768e-06, -0.9996573249749013, -0.026176948307978565]])
        #translation components
        tx = 0
        ty = 0.29999999999994387
        tz = 0.6639999958276623
        # Construct translation vector
        translation_vector = np.array([tx, ty, tz])
        # Construct transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector
        
        camera_pt = np.array([camera_point.x, camera_point.y, camera_point.z, 1])
        base_link_pt = np.dot(transformation_matrix, camera_pt)
        base_link_point = Point()
        base_link_point.x = base_link_pt[0]
        base_link_point.y = base_link_pt[1]
        base_link_point.z = base_link_pt[2]
        return base_link_point
    
    def get_point_wrt_map(self, base_link_point):
        if base_link_point is None:
            rospy.loginfo("Base link point is None")
            return None
        
        base_link_pt = np.array([base_link_point.x, base_link_point.y, base_link_point.z, 1])
        map_pt = np.dot(self.base_map_tf, base_link_pt)    
        map_point = Point()
        map_point.x = map_pt[0]
        map_point.y = map_pt[1]
        map_point.z = map_pt[2]     
        return map_point
         
    def get_width_length_in_world_coordinates(self, x1, y1, x2, y2, z):
        # #the box coordinates in (left, top, right, bottom) format
        # #get 3D line points of the bounding box
        # #Top left point
        # camera_point1 = self.get_point_wrt_camera_frame(x1, y1)
        # map_point1 = self.get_point_wrt_map(self.get_point_wrt_base_link(camera_point1))
        # #Top right point
        # camera_point2 = self.get_point_wrt_camera_frame(x2, y1)
        # map_point2 = self.get_point_wrt_map(self.get_point_wrt_base_link(camera_point2))
        # #Bottom left point
        # camera_point3 = self.get_point_wrt_camera_frame(x1, y2)
        # map_point3 = self.get_point_wrt_map(self.get_point_wrt_base_link(camera_point3))
        # #use depth of the center to have the bounding box in the same plane
        # point_TL = np.array([map_point1.x, map_point1.y, map_point1.z])
        # point_TR = np.array([map_point2.x, map_point2.y, map_point2.z])
        # point_BL = np.array([map_point3.x, map_point3.y, map_point3.z])
        # rospy.loginfo(f"TL: {point_TL}, TR: {point_TR}, BL: {point_BL}")
        # #point_BR = np.array([map_point4.point.x, map_point3.point.y, z])
        # # Compute the Euclidean distance to get the width and length in meters
        # width = np.linalg.norm(point_TR - point_TL)
        # length = np.linalg.norm(point_TL - point_BL)  
        # Compute the Euclidean distance to get the width and length in meters
        width = (x2-x1) * z / self.fx_d
        length = (y2-y1) * z / self.fy_d   
        
        return width, length
                          
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
    

#!/usr/bin/env python

# Importing necessary libraries
import rospy  # ROS Python library
import cv2  # OpenCV library
import numpy as np  # NumPy library for numerical computations
from std_msgs.msg import MultiArrayDimension, Float32MultiArray # ROS message fro transferring lane coordinates
from sensor_msgs.msg import Image  # ROS message type for images
from nav_msgs.msg import Odometry # ROS message type for odometry
from cv_bridge import CvBridge, CvBridgeError  # CvBridge for converting between ROS Image messages and OpenCV images
from tf.transformations import quaternion_matrix

# Define the LaneDetection class
class LaneDetection:
    def __init__(self):
        # Initialize the ROS node with a unique name
        rospy.init_node('lane_detection')

        # Load in parameters from the ROS parameter server
        image_rgb_topic_name = rospy.get_param("/lane_detection/image_rgb_topic_name")
        image_depth_topic_name = rospy.get_param("/lane_detection/image_depth_topic_name")
        odom_topic_name = rospy.get_param("/lane_detection/odom_topic_name")
        prespective_angle_x = rospy.get_param("/lane_detection/prespective_angle_x")
        resolution_x = rospy.get_param("/lane_detection/resolution_x")
        resolution_y = rospy.get_param("/lane_detection/resolution_y")

        # Store needed camera parameters for image -> camera frame transformation
        self.tan_prespective_x = np.tan(np.deg2rad(prespective_angle_x/2))
        self.tan_prespective_y = np.tan(np.deg2rad(prespective_angle_x/2) * 
                                        np.minimum(resolution_x/resolution_y, resolution_y/resolution_x))
                
        # Initialize a NumPy array to store extracted lines
        self.lane_lines = np.zeros((1, 1, 4), dtype=np.int32) 

        # Initialize a NumPy array to store the transformation from camera -> vehicle frame
        self.cam_to_veh_transform = quaternion_matrix(quaternion=[-0.7372773368099708, 4.214387821499521e-07, 
                                                                  3.861774118818495e-07, 0.6755902076155859])
        self.cam_to_veh_transform[:3, 3] = np.array([0.0, 0.3, 0.664])
        
        # Initialize a NumPy array to store the transformation from camera -> global frame
        self.cam_to_glob_transform = np.zeros((4, 4))

        # Create a CvBridge instance for image conversion
        self.bridge = CvBridge()
        
        # Subscribe to the raw camera image and depth map topics from CoppeliaSim 
        self.image_sub = rospy.Subscriber(image_rgb_topic_name, Image, self.image_callback)
        self.depth_sub = rospy.Subscriber(image_depth_topic_name, Image, self.depth_callback)

        # Subscribe to the vehicle's odometry topic to get the vehicle's pose
        self.odom_sub = rospy.Subscriber(odom_topic_name, Odometry, self.odom_callback)
        
        # Publisher for the processed image
        self.image_pub = rospy.Publisher("/lane_detection/image_processed", Image, queue_size=1)

        # Publisher for the lane lines
        self.lane_pub = rospy.Publisher("/lane_detection/lanes", Float32MultiArray, queue_size=1)

    # Method to define the region of interest in the image
    def roi(self, image, vertices):
        # Apply binary filter to image to keep only close-to-white pixels (most likely road markers)
        threshold_min, threshold_max = 220, 255 
        _, threshold_img = cv2.threshold(image, threshold_min, threshold_max, cv2.THRESH_BINARY)

        # Create a mask image of the same size as the input image
        mask = np.zeros_like(image)
        # Set the mask color to white
        mask_color = 255
        # Fill the region of interest defined by vertices with white color
        cv2.fillPoly(mask, pts=[vertices], color=mask_color)
        # Apply the mask to the input image
        cropped_img = cv2.bitwise_and(threshold_img, mask)
        return cropped_img

    def get_line_length(self, hough_line):
        return np.sqrt((hough_line[0, 3] - hough_line[0, 1])**2 + (hough_line[0, 2] - hough_line[0, 0])**2)
    
    def get_lines_dist(self, hough_line_1, hough_line_2):
        # Return the minimum distance from all four possible point pairs
        dist_1_1 = np.sqrt((hough_line_2[0, 1] - hough_line_1[0, 1])**2 + (hough_line_2[0, 0] - hough_line_1[0, 0])**2)
        dist_1_2 = np.sqrt((hough_line_2[0, 1] - hough_line_1[0, 3])**2 + (hough_line_2[0, 0] - hough_line_1[0, 2])**2)
        dist_2_1 = np.sqrt((hough_line_2[0, 3] - hough_line_1[0, 1])**2 + (hough_line_2[0, 2] - hough_line_1[0, 0])**2)
        dist_2_2 = np.sqrt((hough_line_2[0, 3] - hough_line_1[0, 3])**2 + (hough_line_2[0, 2] - hough_line_1[0, 2])**2)

        return np.min(np.array([dist_1_1, dist_1_2, dist_2_1, dist_2_2]))
    
    def get_next_line(self, target_left, target_right, explored_lines, hough_lines):
        # Find the nearest line to either of the left or right lines
        min_dist = 50
        min_line = None
        min_is_left = True
        for line in hough_lines:
            if not any(np.array_equal(line, explored_line) for explored_line in explored_lines):
                dist_left = self.get_lines_dist(target_left, line)
                dist_right = self.get_lines_dist(target_right, line)
                if (dist_left < min_dist) and (dist_left < dist_right):
                    min_line = line
                    min_dist = dist_left
                    min_is_left = True
                elif (dist_right < min_dist) and (dist_right < dist_left):
                    min_line = line
                    min_dist = dist_right
                    min_is_left = False

        # Add nearest line and return it
        if min_line is not None:
            explored_lines.append(min_line)
            if min_is_left:
                return min_line, target_right
            else:
                return target_left, min_line
        
        return None, None
    
    def get_next_line_left(self, target_left, explored_left, hough_lines):
        # Find the nearest line to the left line
        min_dist = 50
        min_line = None
        for line in hough_lines:
            if not any(np.array_equal(line, explored_line) for explored_line in explored_left):
                dist_left = self.get_lines_dist(target_left, line)
                if (dist_left < min_dist):
                    min_line = line
                    min_dist = dist_left

        # Add nearest line and return it
        if min_line is not None:
            explored_left.append(min_line)
        
        return min_line
    
    def get_next_line_right(self, target_right, explored_right, hough_lines):
        # Find the nearest line to the right line
        min_dist = 50
        min_line = None
        for line in hough_lines:
            if not any(np.array_equal(line, explored_line) for explored_line in explored_right):
                dist_right = self.get_lines_dist(target_right, line)
                if (dist_right < min_dist):
                    min_line = line
                    min_dist = dist_right

        # Add nearest line and return it
        if min_line is not None:
            explored_right.append(min_line)
            
        return min_line

    def filter_lines(self, hough_lines):
        # Get gradient of left-most and right-most lines
        left_index = np.argmin(np.minimum(hough_lines[:, 0, 0], hough_lines[:, 0, 2]))
        right_index = np.argmax(np.maximum(hough_lines[:, 0, 0], hough_lines[:, 0, 2]))
        left_line = hough_lines[left_index]
        right_line = hough_lines[right_index]
        left_length = self.get_line_length(left_line)
        right_length = self.get_line_length(right_line)
        line_min_length = 75

        # Get neighbors of both the right and left-most line
        lane_lines = []
        if (left_length > line_min_length) and (right_length > line_min_length):
            lane_lines.append(left_line)
            lane_lines.append(right_line)
            target_left = left_line
            target_right = right_line

            while target_left is not None:
                target_left, target_right = self.get_next_line(target_left, target_right, lane_lines, hough_lines)
        
        elif (left_length > line_min_length):
            lane_lines.append(left_line)
            target_left = left_line

            while target_left is not None:
                target_left = self.get_next_line_left(target_left, lane_lines, hough_lines)

        elif (right_length > line_min_length):
            lane_lines.append(right_line)
            target_right = right_line

            while target_right is not None:
                target_right = self.get_next_line_right(target_right, lane_lines, hough_lines)

        self.lane_lines = np.array(lane_lines)
        # self.lane_lines = hough_lines

    # Method to draw detected lines on the image
    def draw_lines(self, image):
        # Iterate over the detected lines
        for line in self.lane_lines:
            # Get the coordinates of line and draw onto image
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return image

    # Method to process the input image and detect lanes
    def process(self, img):
        # Get the dimensions of the image
        height = img.shape[0]
        width = img.shape[1]
        
        # Define the vertices of the region of interest
        roi_vertices = np.array([
            [0, height//2],
            [0, height],
            [width, height],
            [width, height//2]
        ], dtype=np.int32)
        
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply dilation to enhance edges
        gray_img = cv2.dilate(gray_img, kernel=np.ones((3, 3), np.uint8))
        # Extract the region of interest
        roi_img = self.roi(gray_img, roi_vertices)
        # Apply Canny edge detection
        canny = cv2.Canny(roi_img, 130, 220)
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(canny, 1, np.pi/180, threshold=25, minLineLength=30, maxLineGap=5)
        # Process detected lines to extract potential lane lines only
        if lines is not None:
            self.filter_lines(lines)
        # Draw the detected lines on the original image
        # img = self.draw_lines(img)
        # return img
    
    def publish_lines(self, lines):
        # Initialize the msg object
        msg = Float32MultiArray()
        msg.data = lines.tolist()
        msg.layout.data_offset = 0
        msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]

        # dim[0] is the number of lines
        msg.layout.dim[0].label = "lines"
        msg.layout.dim[0].size = self.lane_lines.shape[0]
        msg.layout.dim[0].stride = self.lane_lines.shape[0] * self.lane_lines.shape[1]

        # dim[1] is the 4 coordinates of each line (x1, y1, x2, y2)
        msg.layout.dim[1].label = "points"
        msg.layout.dim[1].size = self.lane_lines.shape[1]
        msg.layout.dim[1].stride = self.lane_lines.shape[1]

        # Publish msg
        self.lane_pub.publish(msg)

    # Callback function to handle incoming images
    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # Process the image
        self.process(cv_image)
        # processed_image = self.process(cv_image)

        # Publish the processed image
        # self.image_pub.publish(self.bridge.cv2_to_imgmsg(processed_image, encoding="rgb8"))

    # Callback function to handle incoming depth maps
    def depth_callback(self, msg):
        # Convert ROS Image to OpenCV Image
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        
        # Convert from uint8 representation to float32
        depth = np.frombuffer(depth_image, dtype=np.float32).reshape((depth_image.shape[0], depth_image.shape[1]))
    
        # Get coordinates of all points in lane markers (in a flateneed array)
        if self.lane_lines.shape[0] > 0:
            lines = np.empty((self.lane_lines.shape[0] * self.lane_lines.shape[2], ), dtype=np.float32)
            for i, lane_line in enumerate(self.lane_lines):
                # Get point coordinates and transform (u, v) coordinates into (x_img, y_img) in pixels
                u1, v1, u2, v2 = lane_line[0]
                x1, y1 = u1 - depth.shape[1]/2, v1 - depth.shape[0]/2
                x2, y2 = u2 - depth.shape[1]/2, v2 - depth.shape[0]/2

                # Get coordinates in meters relative to the camera's frame
                x1 = 2 * x1 * depth[v1, u1] * self.tan_prespective_x/depth.shape[1]
                y1 = 2 * y1 * depth[v1, u1] * self.tan_prespective_y/depth.shape[0]
                x2 = 2 * x2 * depth[v2, u2] * self.tan_prespective_x/depth.shape[1]
                y2 = 2 * y2 * depth[v2, u2] * self.tan_prespective_y/depth.shape[0]

                # Transform from camera into global frame (using homogeneous coordinates)
                p1 = np.matmul(self.cam_to_glob_transform, np.array([x1, y1, depth[v1, u1], 1.0], dtype=np.float32))
                p2 = np.matmul(self.cam_to_glob_transform, np.array([x2, y2, depth[v2, u2], 1.0], dtype=np.float32))

                # Extract Cartesian coordinates back from homogeneous coordinates
                lines[4 * i] = p1[1]/p1[3]          # X = Y (in sim frame)
                lines[4 * i + 1] = -p1[0]/p1[3]     # Y = -X (in sim frame) 
                lines[4 * i + 2] = p2[1]/p2[3]
                lines[4 * i + 3] = -p2[0]/p2[3]
            
            # Publish the coordinates of the lane markers  
            self.publish_lines(lines)

    # Callback function for handling incoming odometry messages
    def odom_callback(self, msg):
        # Calculate transformation matrix (vehicle -> global frame) from pose
        trans_vect = np.array([msg.pose.pose.position.x, 
                               msg.pose.pose.position.y, 
                               msg.pose.pose.position.z])
        rot_mat = quaternion_matrix(quaternion=[msg.pose.pose.orientation.x,
                                                msg.pose.pose.orientation.y,
                                                msg.pose.pose.orientation.z,
                                                msg.pose.pose.orientation.w])
        veh_to_glob_transform = rot_mat
        veh_to_glob_transform[:3, 3] = trans_vect

        # Calculate final transformation matrix from camera -> global frame
        self.cam_to_glob_transform = np.matmul(veh_to_glob_transform, self.cam_to_veh_transform)

# Check if the script is run directly and call the main function
if __name__ == '__main__':
    # Create an instance of LaneDetectionNode class
    lane_detection_node = LaneDetection()
    try:
        # Keep the node running
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")  # Handle keyboard interrupt

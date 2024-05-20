#!/usr/bin/env python3

# Importing necessary libraries
import rospy  # ROS Python library
import cv2  # OpenCV library
import numpy as np  # NumPy library for numerical computations
from std_msgs.msg import MultiArrayDimension, Float32MultiArray # ROS message fro transferring lane coordinates
from sensor_msgs.msg import Image  # ROS message type for images
from nav_msgs.msg import Odometry # ROS message type for odometry
from cv_bridge import CvBridge  # CvBridge for converting between ROS Image messages and OpenCV images
from tf.transformations import quaternion_matrix

# Define the LaneDetection class
class LaneDetection:
    def __init__(self):
        # Initialize the ROS node with a unique name
        rospy.init_node('lane_detection')

        # Load in parameters from the ROS parameter server
        odom_topic_name = rospy.get_param("/lane_detection/odom_topic_name", default="/odom")
        image_rgb_topic_name = rospy.get_param("/lane_detection/image_rgb_topic_name", default="/image")
        image_depth_topic_name = rospy.get_param("/lane_detection/image_depth_topic_name", default="/image_depth")
        
        prespective_angle_x = rospy.get_param("/lane_detection/prespective_angle_x", default=60.0)
        resolution_x = rospy.get_param("/lane_detection/resolution_x", default=960)
        resolution_y = rospy.get_param("/lane_detection/resolution_y", default=480)
        cam_to_veh_quat = rospy.get_param("/lane_detection/cam_to_veh_quat", default=[0.0, 0.0, 0.0, 1.0])
        cam_to_veh_offset = rospy.get_param("/lane_detection/cam_to_veh_offset", default=[0.0, 0.0, 0.0])

        self.image_color_threshold_min = rospy.get_param("/lane_detection/image_color_threshold_min", default=220)
        self.lane_line_min_length = rospy.get_param("/lane_detection/lane_line_min_length", default=75)
        self.lane_line_max_dist = rospy.get_param("/lane_detection/lane_line_max_dist", default=50)
        self.hough_threshold = rospy.get_param("/lane_detection/hough_threshold", default=25)
        self.hough_min_length = rospy.get_param("/lane_detection/hough_min_length", default=30)
        self.hough_max_gap = rospy.get_param("/lane_detection/hough_max_gap", default=5)
        
        self.publish_lane_image = rospy.get_param("/lane_detection/publish_lane_image", default=False)

        # Store needed camera parameters for image -> camera frame transformation
        self.tan_prespective_x = np.tan(np.deg2rad(prespective_angle_x/2))
        self.tan_prespective_y = np.tan(np.deg2rad(prespective_angle_x/2) * 
                                        np.minimum(resolution_x/resolution_y, resolution_y/resolution_x))
                
        # Initialize a NumPy array to store extracted lines
        self.lane_lines = np.ones((1, 4), dtype=np.int32) 

        # Initialize a NumPy array to store the transformation from camera -> vehicle frame
        self.cam_to_veh_transform = quaternion_matrix(quaternion=cam_to_veh_quat)
        self.cam_to_veh_transform[:3, 3] = np.array(cam_to_veh_offset)
        
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
        self.image_pub = rospy.Publisher("/lane_detection/image_processed", Image, queue_size=0)

        # Publisher for the lane lines
        self.lane_pub = rospy.Publisher("/lane_detection/lanes", Float32MultiArray, queue_size=0)

    # Method to define the region of interest in the image
    def roi(self, image, vertices):
        # Apply binary filter to image to keep only close-to-white pixels (most likely road markers)
        threshold_min, threshold_max = self.image_color_threshold_min, 255 
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
        return np.sqrt((hough_line[3] - hough_line[1])**2 + (hough_line[2] - hough_line[0])**2)
    
    def get_lines_dist(self, hough_line_1, hough_line_2):
        # Return the minimum distance from all four possible point pairs
        dist_1_1 = np.sqrt((hough_line_2[1] - hough_line_1[1])**2 + (hough_line_2[0] - hough_line_1[0])**2)
        dist_1_2 = np.sqrt((hough_line_2[1] - hough_line_1[3])**2 + (hough_line_2[0] - hough_line_1[2])**2)
        dist_2_1 = np.sqrt((hough_line_2[3] - hough_line_1[1])**2 + (hough_line_2[2] - hough_line_1[0])**2)
        dist_2_2 = np.sqrt((hough_line_2[3] - hough_line_1[3])**2 + (hough_line_2[2] - hough_line_1[2])**2)

        return np.min(np.array([dist_1_1, dist_1_2, dist_2_1, dist_2_2]))
    
    def pairwise_line_distances(self, hough_lines):
        # Reshape lines to (n, 2, 2) where each line is represented by two endpoints (x1, y1), (x2, y2)
        line_points = hough_lines.reshape(-1, 2, 2)
        
        # Expand dimensions to compute pairwise distances
        expanded_lines_1 = line_points[:, None, :, :]  # shape (n, 1, 2, 2)
        expanded_lines_2 = line_points[None, :, :, :]  # shape (1, n, 2, 2)
        
        # Compute pairwise distances between all endpoints of each pair of lines
        dist_squared = np.sum((expanded_lines_1[:, :, :, None, :] - expanded_lines_2[:, :, None, :, :]) ** 2, axis=-1)
        
        # Compute the minimum distance for each pair of lines
        min_distances = np.sqrt(np.min(dist_squared, axis=(2, 3)))
        np.fill_diagonal(min_distances, np.inf)
        
        return min_distances


    def filter_lines(self, hough_lines):
        # Get gradient of left-most and right-most lines
        left_index = np.argmin(np.minimum(hough_lines[:, 0], hough_lines[:, 2]))
        right_index = np.argmax(np.maximum(hough_lines[:, 0], hough_lines[:, 2]))
        left_line = hough_lines[left_index]
        right_line = hough_lines[right_index]
        left_length = self.get_line_length(left_line)
        right_length = self.get_line_length(right_line)
        line_min_length = self.lane_line_min_length
        
        # Get the distances between all pairs of lines
        min_distances = np.ma.masked_array(self.pairwise_line_distances(hough_lines), mask=False)
        lane_lines = np.zeros_like(hough_lines)
        
        # Get neighbors of both the right and left-most line
        if (left_length > line_min_length) and (right_length > line_min_length):
            lane_lines[0], lane_lines[1] = left_line, right_line
            num_lines, min_distance = 2, 0

            while (num_lines < lane_lines.shape[0]) and (min_distance < self.lane_line_max_dist):
                left_neighbor = min_distances[left_index, :].argmin(axis=0, fill_value=np.inf)
                right_neighbor = min_distances[right_index, :].argmin(axis=0, fill_value=np.inf)
                if min_distances[left_index, left_neighbor] < min_distances[right_index, right_neighbor]:
                    min_distance = min_distances[left_index, left_neighbor]
                    min_distances.mask[:, left_index] = True
                    lane_lines[num_lines] = hough_lines[left_neighbor]
                    left_index = left_neighbor
                    num_lines += 1
                else:
                    min_distance = min_distances[right_index, right_neighbor]
                    min_distances.mask[:, right_index] = True
                    lane_lines[num_lines] = hough_lines[right_neighbor]
                    right_index = right_neighbor
                    num_lines += 1

            self.lane_lines = lane_lines[:num_lines-1]

        # Get neighbors of left-most line only
        elif (left_length > line_min_length):
            lane_lines[0] = left_line
            num_lines, min_distance = 1, 0

            while (num_lines < lane_lines.shape[0]) and (min_distance < self.lane_line_max_dist):
                left_neighbor = min_distances[left_index, :].argmin(axis=0, fill_value=np.inf)
                min_distance = min_distances[left_index, left_neighbor]
                min_distances.mask[:, left_index] = True
                lane_lines[num_lines] = hough_lines[left_neighbor]
                left_index = left_neighbor
                num_lines += 1

            self.lane_lines = lane_lines[:num_lines-1]
        
        # Get neighbors of right-most line only
        elif (right_length > line_min_length):
            lane_lines[0] = right_line
            num_lines, min_distance = 1, 0

            while (num_lines < lane_lines.shape[0]) and (min_distance < self.lane_line_max_dist):
                right_neighbor = min_distances[right_index, :].argmin(axis=0, fill_value=np.inf)
                min_distance = min_distances[right_index, right_neighbor]
                min_distances.mask[:, right_index] = True
                lane_lines[num_lines] = hough_lines[right_neighbor]
                right_index = right_neighbor
                num_lines += 1

            self.lane_lines = lane_lines[:num_lines-1]


    # Method to draw detected lines on the image
    def draw_lines(self, image):
        # Iterate over the detected lines
        for line in self.lane_lines:
            # Get the coordinates of line and draw onto image
            x1, y1, x2, y2 = line
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return image

    # Method to process the input image and detect lanes
    def process(self, img):
        # Get the dimensions of the image
        height = img.shape[0]
        width = img.shape[1]
        
        # Define the vertices of the region of interest
        roi_vertices = np.array([
            [0, 3*height//5],
            [0, 9*height//10],
            [width, 9*height//10],
            [width, 3*height//5]
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
        lines = cv2.HoughLinesP(canny, 1, np.pi/180, threshold=self.hough_threshold, 
                                minLineLength=self.hough_min_length, maxLineGap=self.hough_max_gap)
        lines = np.squeeze(lines) # Remove unnecessary 2nd dimension of length 1

        # Process detected lines to extract potential lane lines only
        # if (lines is not None) and (lines.ndim > 1):
        #     self.filter_lines(lines)
        self.lane_lines = lines

        # Draw the detected lines on the original image if processed image is required
        if self.publish_lane_image:
            img = self.draw_lines(img)
            return img
        
        return None
    
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
        processed_image = self.process(cv_image)

        # Publish the processed image if required
        if self.publish_lane_image:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(processed_image, encoding="rgb8"))

    # Callback function to handle incoming depth maps
    def depth_callback(self, msg):
        # Convert ROS Image to OpenCV Image
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        
        # Convert from uint8 representation to float32
        depth = np.frombuffer(depth_image, dtype=np.float32).reshape((depth_image.shape[0], depth_image.shape[1]))
    
        # Get coordinates of all points in lane markers (in a flateneed array)
        if self.lane_lines.ndim > 1 and self.lane_lines.shape[0] > 0:
            lines = np.empty((self.lane_lines.shape[0] * self.lane_lines.shape[1], ), dtype=np.float32)
            for i, lane_line in enumerate(self.lane_lines):
                # Get point coordinates and transform (u, v) coordinates into (x_img, y_img) in pixels
                u1, v1, u2, v2 = lane_line
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

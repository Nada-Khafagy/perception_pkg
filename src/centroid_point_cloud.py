#!/usr/bin/env python3
import rospy 
from geometry_msgs.msg import PointStamped 
from sensor_msgs.msg import PointCloud2, PointField 
import numpy as np 
from perception_pkg.msg import bounding_box,bounding_box_array 

class PointCloudPublisher:
    def __init__(self):
        rospy.init_node('point_cloud_publisher', anonymous=True)
        self.point_cloud_publisher = rospy.Publisher('/point_cloud', PointCloud2, queue_size=10)
        self.bounding_boxes_subscriber = rospy.Subscriber('/bounding_boxes', bounding_box_array, self.bounding_boxes_callback)
        self.centroid_points = []  # List to store centroid points
    
    def bounding_boxes_callback(self, msg):
        # Clear previous centroid points
        self.centroid_points = []
        # Extract centroid points from bounding boxes
        for bb in msg.bbs_array:
            bb: bounding_box
            centroid_point = bb.centeroid
            self.centroid_points.append(centroid_point)
        # Publish point cloud
        self.publish_point_cloud()

    def publish_point_cloud(self):
        if not self.centroid_points:
            rospy.loginfo("No centroid points to publish.")
            return
        
        # Create PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header.stamp = rospy.Time.now()
        cloud_msg.header.frame_id = 'map'  # Set appropriate frame ID
        
        # Define fields for the point cloud data (x, y, z)
        cloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        # Flatten list of PointStamped messages into arrays of x, y, z coordinates
        points = np.array([(point.x, point.y, point.z) for point in self.centroid_points], dtype=np.float32)

        # Convert points to a binary string
        cloud_msg.data = points.tostring()
        rospy.loginfo("Point cloud data: %s" % cloud_msg.data)

        # Set width, height, and point step (number of bytes per point)
        cloud_msg.width = len(self.centroid_points)
        cloud_msg.height = 1
        cloud_msg.point_step = 12  # 3 fields (x, y, z) * 4 bytes/field

        # Set is_dense flag to indicate if the data contains NaN values
        cloud_msg.is_dense = True

        self.point_cloud_publisher.publish(cloud_msg)

        rospy.loginfo("Published point cloud with %d points." % len(self.centroid_points))

if __name__== '__main__':
    try:
        point_cloud_publisher = PointCloudPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted")

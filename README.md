# Perception Package

## Package Summary

The **perception_pkg** contains all nodes related to allowing the vehicle to perceive and interpret its surrounding environment by processing data from sensors such cameras, LiDARs or radar. However, at its current state only a front stereo camera is relied upon. The package is made up of two nodes, namely:
    
* `lane_detection`: A node responsible for detecting solid lane markers on the road and extracting the coordinates of their endpoints. The node relies on older techniques for CV such as: Canny edge detection and the Hough transform, which require manual tuning, for lane detection.

* `object_detection`: A node responsible for detecting objects in the environment such as other cars, pedestrians, cones, barriers, etc. It relies on the YOLOv8 nano model, trained on data from the simulator, to pefrom 2D object detection and is capable of classifying objects into one of 14 classes or as background objects.  

## Nodes Summary

### Lane Detection:

#### Node Parameters:
* `odom_topic_name`: Name of the topic where the vehicle's current pose and twist are published.
* `image_rgb_topic_name`: Name of the topic where the front camera's RGB image is published from the simulator.
* `image_depth_topic_name`: Name of the topic where the front camera's depth map is published from the simulator.
* `prespective_angle_x`: Camera's horizontal prespective angle in degrees. Parameter from the simulator which affects the camera's FOV.
* `resolution_x`: The camera's resolution in pixels along its horizontal axis.
* `resolution_y`: The camera's resolution in pixels along its vertical axis. 
* `cam_to_veh_quat`: Quaternion which defines the relative orientation of the camera's frame of reference to that of the vehicle.
* `cam_to_veh_offset`: Translation vector which defines the offset from the origin of the camera's frame of reference to that of the vehicle.
* `image_color_threshold_min`: The minimum pixel intensity below which all pixels are filtered to black during the thresholding operation before lane detection.
* `hough_threshold`: The minimum number of intersection points in the Hough space needed to consider a line as valid.
* `hough_min_length`: The minimum length a line has to be in order to be detected by the Hough Transform.
* `hough_max_gap`: The maximum gap between two lines such that they can be considered parts of the same line, else they'll be treated as separate lines.
* `publish_lane_image`: Controls whether the input image with detected lane lines overlayed on top of it should be published or not.

#### Known Issues and Possible Solutions:

* **Issue**: False detections of dashed lanes and failing to accurately detect curved solid lane lines.
* **Solution**: Unfortunatley, its hard to solve both as solving one makes the other worse. By increasing the minimum line length, the former is solved but the latter is worsened and vice versa. The best solution is to adopt a more modern deep learning-based approach for lane detection such as *LaneATT*.

### Object Detection

#### Node Parameters:
* `odom_topic_name` :  Name of the topic where the vehicle's current pose and twist are published.
* `image_rgb_topic_name` : Name of the topic where the front camera's RGB image is published from the simulator.
* `image_depth_topic_name` : Name of the topic where the front camera's depth map is published from the simulator.
* `prespective_angle_x` : Camera's horizontal prespective angle in degrees. Parameter from the simulator which affects the camera's FOV.
* `resolution_x` : The camera's resolution in pixels along its horizontal axis.
* `resolution_y` : The camera's resolution in pixels along its vertical axis. 
* `use_depth` : Boolean parameter that should be set to true to get the bounding box message published.
* `use_less_classes` : Boolean parameter that limits the classes to the classes needed by EVER (car, person, traffic cone).
* `use_tracking` : Boolean parameter that allows for object tracking (not implemented).
* `confidence_threshold` : The minimum confidence value above which the model detects the object.
* `iou_threshold` : The minimum intersection over union value above which two detected overlabbing bounding boxes are considered to be the same object
* `weights_path` : The path to the weights file which is the ".pt" output file from training the yolo model.

#### Known Issues and Possible Solutions:
* **Issue**: low mAP 
* **Solution**: increase the number of annotation for certain classes (increase dataset).
* **Issue**: high inference time
* **Solution**: for now i think using a better Nvidia gpu would be a good solution :) 


<br></br>
## Requirements
- Make sure you have CV2 and download and the other dependancies, run this:
```bash
pip install -r requirements.txt
```
- Then Build the package (put an empty folder called include if building caused an issue).

## Camera specification in Coppeliasim
- frequency = 20 Hz
- near/far plane = [0.01, 30] 
- perspective angle = 85.0deg
- resolution 960x480

## Dataset
- Roboflow: https://universe.roboflow.com/autocomp/ccc123iii123ttt123yyy
- It is recommended to have 1.5k> annotation per class
- The current classes are 14 : Left hand curve sign, Pedestrian Crossing sign, Right Hand Curve sign, Yield sign, car, person, plastic lane barrier, roundabout sign, speed limit (10) sign, speed limit (20) sign, speed limit (40) sign, speed limit (50) sign, stop sign, traffic cone




## Refrences:
- http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
- https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/
- https://github.com/hank-ai/darknet
- https://www.ccoderun.ca/programming/yolo_faq/
- https://github.com/ultralytics/yolov5
- https://github.com/leggedrobotics/darknet_ros
- https://github.com/mats-robotics/yolov5_ros
- https://karpathy.github.io/2019/04/25/recipe/
- https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
- https://cs.wellesley.edu/~cs307/lectures/Camera.html
- https://github.com/openvinotoolkit/openvino

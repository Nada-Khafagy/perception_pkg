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

#### Known Issues and Possible Solutions:

## Milestone III - 16 April:
This milestone will be testing mainly! and seeing how the integration of image processing with CoppeliaSim will be done.
- ~~Exctract info from Camera~~ 
- ~~Test image processing operations~~

- ~~Get X,Y of the object and send on the topic~~
- ~~to facilitate the lane detection you can get the depth of the lines using hough transform~~
- Check if the aruco markers can be detected--> i added them to the YOLO classes still didn't test cv on them 
- ~~look for already made packages that can be used~~

- ~~create a lane detection using canny edge detector~~


## Milestones IX -  23 April:
- ~~Create needed topics to integrate with navigation team~~
- ~~invistigate YOLO~~

- ~~investigate other lane detection protocols such as morphological operatins~~

## Milestone X - 3 May:
- Modify Yolo
- Check how to make the operation faster since there might be a lot of delay in the processing (avoid drawbacks)

- ~~create a ROS node and test it in the environment to get the desired final output and test the curves adaptability~~

## Milestone XI - 7 May:
- System integration

<br></br>

# Notes on Object detection
## Using CV
- straight forward just use the  CvBridge, look at the ROS tut in the refrences

## Camera specification in Coppeliasim
- frequency = 20 Hz
- near/far plane = [0.01, 1000] 
- perspective angle = 85.0deg
- resolution 960x480

## Screenshots for Milestone2
- https://drive.google.com/drive/folders/1LXjcMTeBrHbOERKU5WTjVh3MWxG7T_q3?usp=sharing

## Yolo
### Data annotation
- I am using Roboflow, https://universe.roboflow.com/autocomp/ccc123iii123ttt123yyy
- it is recommended to have 1.5k> per class, rn its 1014
- the current classes are 14 : Left hand curve sign, Pedestrian Crossing sign, Right Hand Curve sign, Yield sign, car, person, plastic lane barrier, roundabout sign, speed limit (10) sign, speed limit (20) sign, speed limit (40) sign, speed limit (50) sign, stop sign, traffic cone

### training
- Yolov8n
    - curr mAP (0.5-0.95) =  0.74099 --> lower than last time due to the lower number of imgs (less agumentation) 
    but i was testing to see if that will improve the recall, it didn't :) will increse the dataset and train again with higher augmentation...

### integration with ros
- made a simple wrapper lots of refrences are listed

## Testing the package
- make sure you have CV2 and download and the other dependancies, run this:
```bash
pip install -r requirements.txt
```
- to run with the filtered data
```bash
roslaunch perception_pkg object_detector.launch use_encoder:=true
```
- to send bounding boxes msgs
```bash
roslaunch perception_pkg object_detector.launch  use_depth:=true
```
- to visualize only percsion of an object (enter how many in the scene)
```bash
roslaunch perception_pkg object_detector.launch  car_num:=2 
```
```bash
roslaunch perception_pkg object_detector.launch  person_num:=2 
```
```bash
roslaunch perception_pkg object_detector.launch  cone_num:=2 
```
- then don't forget to build the package
- also put an empty folder called include if building caused an issue


## Next Steps:
- ~~get new data with different scene settings and lightings and anotate them. (usless cuz of greyscale but might try diff lightings)~~
- ~~analayze the YOLO archetecture --> i found a tutorial where i can build it from scratch i might try that.~~
- ~~test using cv2 to detect aruco markers and estimate distance (neglect this for now)~~
- ~~test yolov7 built on the Darknet framewotrk (meh might try that later not now)~~
- optimize inference time using openvino and look for other techniques
- use the velodyne and try to estimate distance with obtacles better 


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

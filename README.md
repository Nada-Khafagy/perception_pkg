# Perception
The perception of the vehicle will be divided into two parts: 
- lane detection (Mina)
- Object detection (Nada)


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


```diff
- Warning: Shiko you are not allowed to edit anything in this repo focus on your tasks thank you.
```

# Perception

The perception of the vehicle will be divided into two parts: 
- lane detection (Mina)
- Object detection (Nada)


## Milestone III - 16 April:
This milestone will be testing mainly! and seeing how the integration of image processing with CoppeliaSim will be done.
- ~~Exctract info from Camera~~ 
- ~~Test image processing operations~~
- Create needed topics to integrate with navigation team --> we should discuss this
- Get X,Y of the object and send on the topic
- to facilitate the lane detection you can get the depth of the lines using Half Transform
- Check if the aruco markers can be detected--> i added them to the YOLO classes still didn't test cv on them
- ~~look for already made packages that can be used~~


## Milestones IX -  23 April:
- Commnunicate with the planning team
- Take an action based on the processed frames 
- ~~invistigate YOLO~~

## Milestone X - 3 May:
- Modify Yolo
- Check how to make the operation faster since there might be a lot of delay in the processing (avoid drawbacks) 

## Milestone XI - 7 May:
- System integration

<br></br>

# Notes on Object detection
## Using CV
- straight forward just use the  CvBridge, look at the ROS tut in the refrences

## Camera specification in Coppeliasim
- frequency = 20 Hz
- near/far plane = [0.1, 100] (I wanted to capture the sky)
- perspective angle = 60deg
- resolution 1080x720

## Environment 
- https://drive.google.com/file/d/1Fx4pnt02th4oUcjqRUbAMUMitahHYjym/view?usp=sharing


## Yolo
### Data annotation
- I am using Roboflow, https://universe.roboflow.com/autocomp/city-h2uz7
- it is recommended to have 1.5k> per class, rn it is around(800) total
- the current classes are 9 : aruco marker sign, bike, building, car, person, plastic lane barrier, traffic cone, traffic sign, tree

### training
- used Yolov5m 
    - mAP (0.5-0.95) = 0.68185 
    - confusion matrix: acceptable but not the best
- Yolov8n
    - mAP (0.5-0.95) =  0.76426
    - has a better confusion matrix
    - trained it but haven't tested it yet
### integration with ros
- made a simple wrapper lots of refrences are listed

## Testing the package
- make sure you have CV2 and download and the other dependancies
- you can run the following if you want the confidence to be 0.8 (default is 0.5 but then you will see how bad the model is T-T )
```bash
roslaunch perception_pkg object_detection.launch confidence_threshold:=0.8
```
- for the submodule to run you need to clone this repo then add the submodule
```bash
git clone --recurse-submodules https://github.com/ultralytics/yolov5
cd yolov5_ros/src/yolov5
pip install -r requirements.txt
```
- then don't forget to build the package (I hope it works)
- also put like an empty folder called include if building caused an issue

## Next Steps:
- get new data with different scene settings and lightings and anotate them.
- analayze the YOLO archetecture --> i found a tutorial where i can build it from scratch i might try that.
- test using cv2 to detect aruco markers.
- test yolov7 built on the Darknet framewotrk


## Refrences:
- http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
- https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/
- https://github.com/hank-ai/darknet
- https://www.ccoderun.ca/programming/yolo_faq/
- https://github.com/ultralytics/yolov5
- https://github.com/leggedrobotics/darknet_ros
- https://github.com/mats-robotics/yolov5_ros
- https://karpathy.github.io/2019/04/25/recipe/





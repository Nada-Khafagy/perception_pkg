
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

## Camera specification in Coppeliasim
- frequency = 20 Hz
- resolution 1080x720
- near/far plane = [0.1, 100] (I wanted to capture the sky)

## Environment 
- https://drive.google.com/file/d/1Fx4pnt02th4oUcjqRUbAMUMitahHYjym/view?usp=sharing

## Next Steps:
- get new data with different scene settings and lightings and anotate them.
- analayze the YOLO archetecture --> i found a tutorial where i can build it from scratch i might try that.
- test using cv2 to detect aruco markers.


## Refrences:
- http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
- https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/
- https://github.com/hank-ai/darknet
- https://www.ccoderun.ca/programming/yolo_faq/
- https://github.com/ultralytics/yolov5
- https://github.com/leggedrobotics/darknet_ros
- https://github.com/mats-robotics/yolov5_ros




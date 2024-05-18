# Message for object detection, containing information about detected objects.
string class_name               # Name of the calss of the detected 
float32 confidence              # Confidence score (level of certainty of the model that a particular object detection is accurate)
geometry_msgs/Point centroid    # 3D coordinates of the centroid of the detected object in the scene (with respect to world coordinates)
float64 width                   # Width of the detected object in the secne
float64 length                  # Length of the detected object in the scene

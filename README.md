
# Hand Tracker

## Basic Info
Should be python based hand tracker using cv2 library for camera intergration and mediapipe for its hand tracker model.
Currently the hand positsion is calculated to be twice as far from the middle as the cameras perspective. This is to induces a "safe zone",
this allows users to overshoot the screen, without the tracking losing the hand. The cordinates are always fixed to specifc cordinates, so losses of tracking are resolved when a hand is found again (sort of like a trackpad). 

## Model 
Project uses googles LM [hand_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).

## Prerequisites to running programm
Required libraries are in the requirements.txt file, recommended to be installed with latest version of pip.

## To run
Program can be activated with python (for example python3) and running the program name.
Has flags to configure settings:
--mode : preview mode, can be p for preview and l live.
--udp-ip: default="127.0.0.1")
--udp-port: port for destination, type=int, default=5055)

default command from directory containing the file:
python3 tracket.py --mode p

## To do
- Test performance improvments
- Research packaging the file to be runnable without python (.exe file)
- UI for user friendly usability  

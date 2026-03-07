
# Hand Tracker
>[!CAUTION]
>!NB! IN DEVELOPMENT

## Basic Info
Should be python based hand tracker using cv2 library for camera intergration and mediapipe for its hand tracker model.
Currently the hand positsion is calculated to be twice as far from the middle as the cameras perspective. This is to induces a "safe zone",
this allows users to overshoot the screen, without the tracking losing the hand. The cordinates are always fixed to specifc cordinates, so losses of tracking are resolved when a hand is found again (sort of like a trackpad). 

## Model 
Project uses googles LM [hand_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).

## Prerequisites to running programm
Required libraries are in the requirements.txt file, recommended to be installed with latest version of pip.


## To do
- Test performance improvments
- Test impact of hand vs point of hand tracking (currently only one point tracked)
- Reserch packaging the file to be runnable without python (.exe file)
- Research why hand detection is slow (hand tracking is fine)
  

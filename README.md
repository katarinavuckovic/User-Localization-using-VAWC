# User-Localization-using-VAWC

## PART 1: YOLO Object Detection 
This implementes the YOLO object detection algorithm on the ViWi Dataset for single user colocated camera scenarios. 
The YOLO model is trained to detect a single user in the image and output a bounding box around the user.
Training and testing dataset generated using image-labeling tool in MATLAB. 
The dataset can be downloaded from  [ViWi Dataset](https://viwi-dataset.net/scenarios.html). (scenrio 2 and 3)

Training Dataset: 
-Dataset contains images from all three cameras
-300 images (mix of all three cameras)

Testing Dataset:  TestDataset.mat
-101 images (mix of all three cameras)

YOLO Training: yoloTrainViWi.m
Trained Model: viWiYOLODetector_V2

YOLO Testing: yoloTestViWi.m

## PART 2: MLP Neural Network
This network is trained to map the location of the bounding box in the image to the (x,y) locations on the map. 
The input to the network are 4 values that define the bound box.
The output are two values that (x,y) geolocation coordinates on the map.
Since there are 3 cameras, there are three MLP networks (one for each camera). 

Training Dataset: TrainMLPDatasetCam1_V2.mat,TrainMLPDatasetCam1_V2.mat,TrainMLPDatasetCam1_V2.mat

Testing Dataset: TestMLPDataset.mat corresponds to the TestDataset.mat

MLP Training: mlpTrainViWi.m, buildNetfc_H3.m

Trained Models: TrainedNetCam1, TrainedNetCam2, TrainedNetCam3

MLP Testing: mlpTestViwi.m 

## PART 3: User Localization using Object Detection (End-to-End System)
This part integrates 1 and 2 into an end-to-end system and calculates the localization accuracy. 
The input to the system is a image from one of the camera and the output is the estimated user location.

Single Sample Test: SingleSampleTest.m
- Tests a single sample from the dataset. 
- Outputs the input image overlayed with the bounding box and the localization error. 

System Testing: TestSystem.m
- Outputs the localization error over the entire training dataset (mean error, cdf...etc)


## Questions 
If you have any question regarding the codes, dont hesitate to contact me at kvuckovic993@gmail.com or kvuckovic@knights.ucf.edu

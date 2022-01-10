% Katarina Vuckovic, UCF ECE5415, Dec 2021

% Description:
% This code test is a quick test of the YOLO detection algorithm. The code
% loads a single image and performs YOLO object detection. The code
% displays the image with the bounding box surrounding the object (user)
% and the score that represents the confidence of the recognizer. 
s
close all
clear all 
clc
%load detector
load ViWiYOLODetector_V2
% add location of the test image
I = imread('D:\MATLAB\YOLO\rgb\cam_3_15.4439_10.jpg');
figure (1)
[bboxes,scores] = detect(viWiDetector_V2,I);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
imshow(I)

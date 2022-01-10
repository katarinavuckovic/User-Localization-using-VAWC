% Katarina Vuckovic, UCF ECE5415, Dec 2021

% Description: 
% This code test a single sample. The input is the ith sample (image) from the
% dataset and the output is the image with the YOLO bounding box and the
% estimated location error after MLP.

%----------------------------------------------------------------------------
% Initialization and loading of dataset
clear all
close all
clc
load viWiYOLODetector_V2 %Load YOLO detector 
%load('ViWiYOLODetector_V2.mat')
load('TestDataset_v2.mat') %Load input testing dataset
imageFilename = gTruth.DataSource.Source;
%imageFilename(85,:) = []; %Delete Duplicate sample
numAnchors = cell2mat( table2cell(gTruth.LabelData));
not_detected=0;
not_det_index = [];
j = 0;

i = 90; % sample of interest change number to select a different from the test dataset
%---------------------------------------------------------------------------
%Part 1 YOLO estimation
I = imread(imageFilename{i,1});
[bboxes,scores] = detect(viWiDetector_V2,I);
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
imshow(I)
%[anchorBoxes, meanIoU] = estimateAnchorBoxes(bboxes, );
[val,argmax] = max(scores);
if(~isempty(bboxes))
    bboxes(i,:) = bboxes(argmax,:);
    score(i) = val;
    overlapRatio(i) = bboxOverlapRatio(bboxes(argmax,:),numAnchors(i,:));
else
    not_detected(j) = i;
    j = j+1;
    verlapRatio(i) = 0;
    bboxes(i,:) = [0,0,0,0];
    not_det_index = [not_det_index,i];
end

%----------------------------------------------------------------------------
% Part 2 MLP uses the bound box parameters to estimate the x-y location on
% the map

% load testing dataset
load('TestMLPDataset_v2.mat')
trueLoc  = OutputLoc(1:end,:);
input(1,:,1,:)  = bboxes(1:end,:)';

% load MLP networks
load('trainedNetCam1_V3.mat')
load('trainedNetCam2_V3.mat')
load('trainedNetCam3_V3.mat')

% predict and calculate error 
if (i<30) %CAM 1
    Lconstest(i,:) = predict(trainedNetCam1_V3,input(:,:,:,i));
    e = norm(Lconstest(i,:)- trueLoc(i,:))^2;
elseif (i<70) %CAM 2
    Lconstest(i,:) = predict(trainedNetCam2_V3,input(:,:,:,i));
    e = norm(Lconstest(i,:)- trueLoc(i,:))^2;
else %CAM 3
    Lconstest(i,:) = predict(trainedNetCam3_V3,input(:,:,:,i));
    e = norm(Lconstest(i,:)- trueLoc(i,:))^2;
end

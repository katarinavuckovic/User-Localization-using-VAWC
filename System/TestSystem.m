% Katarina Vuckovic, UCF ECE5415, Dec 2021

% Description: 
% This code performs the test over the entire testing dataset. 
% The input is the ith sample (image) from the
% dataset and the output is the image with the YOLO bounding box and the
% estimated location error after MLP. This is computed for all samples
% (images) in the testing dataset. 


clear all 
close all 
clc
%----------------------------------------------------------------------------
%Part 0 -  Initialization, loading model and dataset

% load yolo detector 
load viWiYOLODetector_V2
% load testing dataset
load('TestDataset_v2.mat')
load('TestMLPDataset_v2.mat') 
imageFilename = gTruth.DataSource.Source;
trueLoc  = OutputLoc(1:end,:);
%input  = inputBBox;
numAnchors = cell2mat( table2cell(gTruth.LabelData));
not_detected=0;
j = 0;

%---------------------------------------------------------------------------
%Part 1 YOLO estimation
for i = 1:length(imageFilename)
  %  I = imread(imageFilename(i,:))
    I = imread(imageFilename{i,1});
   
    [bboxes,scores] = detect(viWiDetector_V2,I);
    %[anchorBoxes, meanIoU] = estimateAnchorBoxes(bboxes, );
    [val,argmax] = max(scores);
    if(~isempty(bboxes))
        estimate_bboxes(i,:) = bboxes(argmax,:);
        score(i) = val;
        overlapRatio(i) = bboxOverlapRatio(estimate_bboxes(argmax,:),numAnchors(i,:));
    else
        not_detected(j) = i;
        j = j+1;
        verlapRatio(i) = 0;
        estimate_bboxes(i,:) = [0,0,0,0];
        not_det_index = [not_det_index,i];
    end

end
% Part 2 MLP
%CAM1
load('trainedNetCam1_V3.mat')
for i = 1:29
     Lconstest(i,:) = predict(trainedNetCam1_V3,estimate_bboxes(i,:));
     e1(i) = norm(Lconstest(i,:)- trueLoc(i,:))^2;
end

%CAM2
load('trainedNetCam2_V3.mat')
for i = 30:70
    Lconstest(i,:) = predict(trainedNetCam2_V3,estimate_bboxes(i,:));
    e2(i-29) = norm(Lconstest(i,:)- trueLoc(i,:))^2;
end
% %CAM3
load('trainedNetCam3_V3.mat')
for i = 71:101
     Lconstest(i,:) = predict(trainedNetCam3_V3,estimate_bboxes(i,:));
     e3(i-70) = norm(Lconstest(i,:)- trueLoc(i,:))^2;
end

e_tot = [e1 e2 e3];

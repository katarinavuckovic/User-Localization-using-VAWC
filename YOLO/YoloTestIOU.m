% Katarina Vuckovic, UCF ECE5415, Dec 2021

% Description:
% This code test is a quick test of the YOLO detection algorithm. The code
% loads a single image and performs YOLO object detection. The code
% displays the image with the bounding box surrounding the object (user)
% and the score that represents the confidence of the recognizer. 


clear all
close all 
clc

load viWiYOLODetector_V2 %Load YOLO detector
load('TestDataset.mat') %Load test dataset
truebox = cell2mat(table2array(gTruth.LabelData));
filenames = gTruth.DataSource.Source;
filenames(85,:) = []; % this sample needs to be removed in the dataset because it is corrupted
j = 1;
k = 1;
for i=1:1:(length(filenames))
    I = imread(char(filenames(i)));
    [bboxes,scores] = detect(viWiDetector_V2,I);
    [val,argmax] = max(scores); % if multiple bounding boxes are selected, this picks the one with highest confidence score
    % check if the bounding box is detected
    if(~isempty(bboxes))
        bbox(k,:) = bboxes(argmax,:);
        score(k) = val;
        %the overlap ratio calcualtes the IOU
        overlapRatio(k) = bboxOverlapRatio(bbox(k,:),truebox(i,:));
        k = k+1;
    else
        % counts the number of samples for which no user was detected
        % this never happens in the test samples but it is a check 
        not_detected(j) = i;
        j = j+1;
    end

end
%plot the cdf of IOU
cdfplot(overlapRatio)
title('IOU CDF')
xlim([0 1])
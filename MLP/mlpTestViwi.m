
% Katarina Vuckovic, UCF ECE5415, Dec 2021

% Description:
% This code tests the three MLP models. It caculates the error MSE for each
% sameple and the creates a CDF flot for each camera

clear all
close all
clc

%load testining dataset this dataset consists of 101 samples.
% Samples 1-29 are from camera 1 30-70 are for camera 2 and 71-101 are for
% camera 3. 
load('TestMLPDataset_V2.mat')

trueLoc  = OutputLoc(1:end,:);
%input(1,:,1)  = inputBBox;%(1:end,:);
input = inputBBox;

%CAM1
load('trainedNetCam1_V3.mat')
for i = 1:29
     Lconstest(i,:) = predict(trainedNetCam1_V3,input(i,:));
     e1(i) = norm(Lconstest(i,:)- trueLoc(i,:))^2;
end

%CAM2
load('trainedNetCam2_V3.mat')
for i = 30:70
    Lconstest(i,:) = predict(trainedNetCam2_V3,input(i,:));
    e2(i) = norm(Lconstest(i,:)- trueLoc(i,:))^2;
end
% %CAM3
load('trainedNetCam3_V3.mat')
for i = 71:101
     Lconstest(i,:) = predict(trainedNetCam3_V3,input(i,:));
     e3(i-70) = norm(Lconstest(i,:)- trueLoc(i,:))^2;
end
e_tot = [e1 e2 e3];

%plot CDF
cdfplot(e1)
hold on
cdfplot(e2)
hold on 
cdfplot(e3)
legend('camera 1' ,'camera 2', 'camera 3')

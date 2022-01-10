
% Katarina Vuckovic, UCF ECE5415, Dec 2021

% Description:
% This code rains a MLP network to output (x,y) location of user given the
% parameters of the object detected bounding box (from YOLO)
% Input: 4 parameters defining bounding box
% Output: 2 parameters (x,y location)
% Same architecture is used to train 3 MLP networks (one for each camera).
% Thefore, this code trains on three different datasets (cam1, cam2, cam3)
% and creates 3 different MLP networks
% (trainedNetCam1_V3.mat,trainedNetCam2_V3.mat',trainedNetCam2_V3.mat')


clear all
close all
clc

tic % start time keeping to know hold long the code will take to complete

%load data and prepocess it (reshape, split test/train...)
% CAMERA 1 Dataset
load('TrainMLPDatasetCam1_V2.mat')
OutputLoc = OutputLoc1;
inputBBox =inputBBox1;

% CAMERA 2 Dataset
% load('TrainMLPDatasetCam2.mat')
% OutputLoc = OutputLoc2;
% inputBBox =inputBBox2;

% CAMERA 3 Dataset 
% load('TrainMLPDatasetCam3.mat')
% OutputLoc = OutputLoc3;
% inputBBox =inputBBox3;

% Split data into train and test (validate)
Ttot = 0.8; %Train-Test ratio
len = length(OutputLoc)
shuffledInd = randperm(len);
output = OutputLoc(shuffledInd,:);
input =inputBBox(shuffledInd,:);
n = round(length(input)*.9);
inputTrain(1,:,1,:)  = input(1:n,:)'; % note reshape is required because input is 4D lxwxcxN
outputTrain = output(1:n,:);
inputTest(1,:,1,:)  = input(n+1:end,:)';
outputTest = output(n+1:end,:);

% Setup training network parameters
options.type = 'MLP1';
options.solver = 'adam';
options.learningRate = 0.0001; %1e-5;
options.schedule = 10;
options.dropLR = 'piecewise';
options.dropFactor = 0.1;
options.maxEpoch = 50;
options.batchSize = 15;
options.verbose = 1;
options.verboseFrequency = 10;
options.valFreq = 10;
options.shuffle = 'every-epoch';
options.weightDecay = 1e-6;
options.progPlot = 'none';
options.inputSize = [1,4];
options.transPower = 1;

% 3 hidden layers MLP network 
net1 =  buildNetfc_H3(options);
trainingOpt = trainingOptions(options.solver, ...
    'InitialLearnRate',options.learningRate,...
    'LearnRateSchedule',options.dropLR, ...
    'LearnRateDropFactor',options.dropFactor, ...
    'LearnRateDropPeriod',options.schedule, ...
    'MaxEpochs',options.maxEpoch, ...
    'L2Regularization',options.weightDecay,...
    'Shuffle', options.shuffle,...
    'MiniBatchSize',options.batchSize, ...
    'ValidationData', {inputTest,outputTest},...
    'ValidationFrequency', options.valFreq,...
    'Verbose', options.verbose,...
    'verboseFrequency', options.verboseFrequency,...
    'Plots',options.progPlot);


% Train network
[trainedNetCam1_V3,  trainInfo] = trainNetwork(inputTrain,outputTrain,net1,trainingOpt);
toc
%save network
save('trainedNetCam1_V3.mat')
%save('trainedNetCam2_V3.mat')
%save('trainedNetCam3_V3.mat')


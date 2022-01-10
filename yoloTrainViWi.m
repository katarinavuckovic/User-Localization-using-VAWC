% Katarina Vuckovic, UCF ECE5415, Dec 2021

% Description:
% This code test a single sample. Input is ith sample sample from the
% dataset. The Output is the image with the YOLO bounding box and the
% estimated location error after MLP.

% References:
% Debug refences if code does not work: https://www.mathworks.com/matlabcentral/answers/5200-undefined-function-or-method-for-input-arguments-of-type-double

% Initialize and load training dataset
clear all
close all
clc
tic
%training dataset generate using imageLabeler and images from the viwi
%dataset. 
%viwi: https://viwi-dataset.net/
load('Cam_TrainingDataset_300.mat') 
imageFilename = gTruth.DataSource.Source;
vehicle = gTruth.LabelData.user(:,:);
whos vehicle
vehicleDataset = table(imageFilename, vehicle);

% split dataset into testing, validation and training
rng(0);
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * length(shuffledIndices) );
trainingIdx = 1:idx;
trainingDataTbl = vehicleDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) );
validationDataTbl = vehicleDataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = vehicleDataset(shuffledIndices(testIdx),:);

%Use imageDatastore and boxLabelDatastore to create datastores for loading the image and label data during training and evaluation.
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'vehicle'));

imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'vehicle'));

imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'vehicle'));

%Combine image and box label datastores.
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

%Display one of the training images and box labels.
data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure (1)
imshow(annotatedImage)


%Create a YOLO v2 Object Detection Network
inputSize = [224 224 3]; %this is the actual siz of the image
numClasses = width(vehicleDataset)-1;
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)


%use resnet50 to load a pretrained ResNet-50 model.
featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';

%Create the YOLO v2 object detection network.
lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,
);

%Data Augmentation - increase the dataset by augmenting the existing images
augmentedTrainingData = transform(trainingData,@augmentData);

% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
figure (2)
montage(augmentedData,'BorderSize',10) %works now


%Preprocessing
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
preprocessedValidationData = transform(validationData,@(data)preprocessData(data,inputSize));
preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));
data = read(preprocessedTrainingData);

I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure (3)
imshow(annotatedImage)

%Train YOLO v2 Object Detector
options = trainingOptions('sgdm',...
          'InitialLearnRate',0.001,...
          'Verbose',true,...
          'MiniBatchSize',16,...
          'MaxEpochs',30,...
          'Shuffle','never',...
          'VerboseFrequency',30,...
          'CheckpointPath',tempdir);
[detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
viWiDetector_V2 = detector
save viWiDetector_V2 %save detector 

%Testing
load viWiDetector_V2
%quick test
I = imread('D:\MATLAB\YOLO\rgb\cam_3_15.4439_10.jpg');
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(viWiDetector,I);

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)

toc

%Evaluate Detector Using Test Set
preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));
detectionResults = detect(viWiDetector_V2, preprocessedValidationData);
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedValidationData);

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))
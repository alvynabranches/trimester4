% step 1
[imgDataTrain, labelsTrain, imgDataTest, labelsTest] = prepareData;

%step 2
warning off images:imshow:magnificationMustBeFitForDockedFigure
perm = randperm(numel(labelsTrain), 25);
subset = imgDataTrain(:,:,1,perm);
montage(subset)

%step 3
load MNISTModel
% Predict the class of an image
randIndx = randi(numel(labelsTest));
img = imgDataTest(:,:,1,randIndx);
actualLabel = labelsTest(randIndx);
predictedLabel = net.classify(img);
imshow(img);
title(['Predicted: ' char(predictedLabel) ', Actual: ' char(actualLabel)])

% step 4 -> CNN
layers = [  imageInputLayer([28 28 1])
           convolution2dLayer(5,20)
           reluLayer
           maxPooling2dLayer(2, 'Stride', 2)
           fullyConnectedLayer(10)
           softmaxLayer
           classificationLayer()   ]
       
miniBatchSize = 8192;
options = trainingOptions( 'sgdm',...
   'MiniBatchSize', miniBatchSize,...
   'Plots', 'training-progress');
net = trainNetwork(imgDataTrain, labelsTrain, layers, options);
options = trainingOptions( 'sgdm',...
   'MiniBatchSize', miniBatchSize,...
   'Plots', 'training-progress',...
   'InitialLearnRate', 0.0001);
net = trainNetwork(imgDataTrain, labelsTrain, layers, options);

layers = [
   imageInputLayer([28 28 1])
   convolution2dLayer(3,16,'Padding',1)
   batchNormalizationLayer
   reluLayer
   maxPooling2dLayer(2,'Stride',2)
   convolution2dLayer(3,32,'Padding',1)
   batchNormalizationLayer
   reluLayer
   maxPooling2dLayer(2,'Stride',2)
   convolution2dLayer(3,64,'Padding',1)
   batchNormalizationLayer
   reluLayer
   fullyConnectedLayer(10)
   softmaxLayer
   classificationLayer];
options = trainingOptions( 'sgdm',...
   'MiniBatchSize', miniBatchSize,...
   'Plots', 'training-progress');
net = trainNetwork(imgDataTrain, labelsTrain, layers, options);

% step 5
predLabelsTest = net.classify(imgDataTest);
testAccuracy = sum(predLabelsTest == labelsTest) / numel(labelsTest)

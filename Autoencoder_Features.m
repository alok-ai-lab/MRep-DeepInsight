close all;
clear all;
cd Data
dset=load('dataset4.mat');
dset=dset.dset;
cd ../

%[xTrainImages,tTrain] = digitTrainCellArrayData;
%xTrainImages=dset.Xtrain;
tTrain = zeros(dset.class,size(dset.Xtrain,2));
for j=1:dset.class
    rng = sum(dset.num_tr(1:j-1))+1:sum(dset.num_tr(1:j));
    tTrain(j,rng)=1;
end

rng('default');
hiddenSize1 = 500;
hiddenSize2 = 250;
MaxEpochs = 400;

autoenc1 = trainAutoencoder(dset.Xtrain,hiddenSize1, ...
    'MaxEpochs',MaxEpochs, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false,...
    'UseGPU',true);

feat1 = encode(autoenc1,dset.Xtrain);


autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',MaxEpochs, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false,...
    'UseGPU',true);

feat2 = encode(autoenc2,feat1);

softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',MaxEpochs);
stackednet = stack(autoenc1,autoenc2,softnet);

inputSize = size(dset.Xtrain);%imageWidth*imageHeight;

% Load the test images
%[xTestImages,tTest] = digitTestCellArrayData;
tTest = zeros(dset.class,size(dset.Xtest,2));
for j=1:dset.class
    rng = sum(dset.num_tst(1:j-1))+1:sum(dset.num_tst(1:j));
    tTest(j,rng)=1;
end

% Turn the test images into vectors and put them in a matrix
% xTest = zeros(inputSize,numel(xTestImages));
% for i = 1:numel(xTestImages)
%     xTest(:,i) = xTestImages{i}(:);
% end

y = stackednet(dset.Xtest);
plotconfusion(tTest,y);


% Turn the training images into vectors and put them in a matrix
% xTrain = zeros(inputSize,numel(xTrainImages));
% for i = 1:numel(xTrainImages)
%     xTrain(:,i) = xTrainImages{i}(:);
% end

% Perform fine tuning
stackednet = train(stackednet,dset.Xtrain,tTrain,'useGPU','yes');


y = stackednet(dset.Xtest);
plotconfusion(tTest,y);

% extracting features
tfeat1=encode(autoenc1,dset.Xtest);
tfeat2=encode(autoenc2,tfeat1);
dset.Xtrain = feat1;
dset.Xtest=tfeat1;
dset.dim=size(feat1,1);
save('dataset60.mat','dset','-v7.3');

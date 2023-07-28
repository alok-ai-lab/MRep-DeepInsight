function [XTr,YTr,XVal,YVal] = augmentGAN(XTrain,YTrain,XValidation,YValidation,xp,yp,aug_tr,aug_val)
% augment data using conditional GAN

%Ratio = (double(YTrain)/double(YValidation));
addpath('/home/aloks/MatWorks/Unsup/GANexample/');
XTr=[];
YTr=[];
XVal=[];
YVal=[];

inputSize = [64 64 3];%[Out2.A Out2.B Out2.C];%size(Out2.XTrain);
Factor = 2;
inputSize(1:2) = Factor*inputSize(1:2); %128x128x3
numObservationsNew = aug_tr+aug_val;

%augmenter = imageDataAugmenter(RandXReflection=true);
%augimds = augmentedImageDatastore([64 64],imds,DataAugmentation=augmenter);

numClasses = numel(unique(YTrain));
augimds = augmentedImageDatastore(inputSize(1:2),cat(4,XTrain, XValidation),[YTrain;YValidation]);
%augimdsValidation = augmentedImageDatastore(inputSize(1:2),Out2.XValidation,Out2.YValidation);

numLatentInputs = 100;%100
embeddingDimension = 50;
numFilters = Factor*64;

filterSize = 4;%4;%5;
projectionSize = Factor*[4 4 1024];

layersGenerator = [
    featureInputLayer(numLatentInputs)
    fullyConnectedLayer(prod(projectionSize))
    functionLayer(@(X) feature2image(X,projectionSize),Formattable=true)
    concatenationLayer(3,2,Name="cat");
    transposedConv2dLayer(filterSize,4*numFilters,Stride=2,Cropping="same")
    batchNormalizationLayer
    reluLayer
    transposedConv2dLayer(filterSize,2*numFilters,Stride=2,Cropping="same")
    batchNormalizationLayer
    reluLayer
    transposedConv2dLayer(filterSize,numFilters,Stride=2,Cropping="same")
    batchNormalizationLayer
    reluLayer
    transposedConv2dLayer(filterSize,3,Stride=2,Cropping="same")
    tanhLayer];

lgraphGenerator = layerGraph(layersGenerator);

layers = [
    featureInputLayer(1)
    embeddingLayer(embeddingDimension,numClasses)
    fullyConnectedLayer(prod(projectionSize(1:2)))
    functionLayer(@(X) feature2image(X,[projectionSize(1:2) 1]),Formattable=true,Name="emb_reshape")];

lgraphGenerator = addLayers(lgraphGenerator,layers);
lgraphGenerator = connectLayers(lgraphGenerator,"emb_reshape","cat/in2");

netG = dlnetwork(lgraphGenerator);

dropoutProb = 0.75;
scale = 0.2;

layersDiscriminator = [
    imageInputLayer(inputSize,Normalization="none")
    dropoutLayer(dropoutProb)
    concatenationLayer(3,2,Name="cat")
    convolution2dLayer(filterSize,numFilters,Stride=2,Padding="same")
    leakyReluLayer(scale)
    convolution2dLayer(filterSize,2*numFilters,Stride=2,Padding="same")
    batchNormalizationLayer
    leakyReluLayer(scale)
    convolution2dLayer(filterSize,4*numFilters,Stride=2,Padding="same")
    batchNormalizationLayer
    leakyReluLayer(scale)
    convolution2dLayer(filterSize,8*numFilters,Stride=2,Padding="same")
    batchNormalizationLayer
    leakyReluLayer(scale)
    convolution2dLayer(Factor*4,1)];

lgraphDiscriminator = layerGraph(layersDiscriminator);

layers = [
    featureInputLayer(1)
    embeddingLayer(embeddingDimension,numClasses)
    fullyConnectedLayer(prod(inputSize(1:2)))
    functionLayer(@(X) feature2image(X,[inputSize(1:2) 1]),Formattable=true,Name="emb_reshape")];

lgraphDiscriminator = addLayers(lgraphDiscriminator,layers);
lgraphDiscriminator = connectLayers(lgraphDiscriminator,"emb_reshape","cat/in2");

netD = dlnetwork(lgraphDiscriminator);

numEpochs = 500;
miniBatchSize = 256;%128;%256;%128;
%Specify the options for Adam optimization. For both networks, specify:

%A learning rate of 0.0002

%A gradient decay factor of 0.5

%A squared gradient decay factor of 0.999

learnRate = 0.0002;
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;
%If the discriminator learns to discriminate between real and generated images too quickly, then the generator can fail to train. To better balance the learning of the discriminator and the generator, add noise to the real data by randomly flipping the labels assigned to the real images.

%Specify to flip the real labels with probability 0.35. Note that this does not impair the generator as all the generated images are still labeled correctly.

flipFactor = 0.5;
%Display the generated validation images every 100 iterations.

validationFrequency = 100;

%Train on a GPU if one is available. By default, the minibatchqueue object converts each output to a gpuArray if a GPU is available. Using a GPU requires Parallel Computing Toolbox™ and a supported GPU device. For information on supported devices, see GPU Support by Release (Parallel Computing Toolbox).

augimds.MiniBatchSize = miniBatchSize;
executionEnvironment = "auto";%"gpu";

mbq = minibatchqueue(augimds, ...
    'MiniBatchSize',miniBatchSize, ...
    'PartialMiniBatch',"discard", ...
    'MiniBatchFormat',["SSCB","BC"],...
    'OutputEnvironment',executionEnvironment);
%Train the model using a custom training loop. Loop over the training data and update the network parameters at each iteration. To monitor the training progress, display a batch of generated images using a held-out array of random values to input to the generator as well as a plot of the scores.

%Initialize the parameters for Adam optimization.
velocityD = [];
trailingAvgG = [];
trailingAvgSqG = [];
trailingAvgD = [];
trailingAvgSqD = [];

%To monitor the training progress, display a batch of generated images using a held-out batch of fixed random vectors fed into the generator and plot the network scores.


f = figure;
f.Position(3) = 2*f.Position(3);
%Create a subplot for the generated images and the network scores.

imageAxes = subplot(1,2,1);
scoreAxes = subplot(1,2,2);
%Initialize the animated lines for the scores plot.
lineScoreGenerator = animatedline(scoreAxes,Color=[0 0.447 0.741]);
lineScoreDiscriminator = animatedline(scoreAxes,Color=[0.85 0.325 0.098]);filterSize

legend("Generator","Discriminator");
ylim([0 1])
xlabel("Iteration")
ylabel("Score")
grid on

numValidationImagesPerClass = 5;
ZValidation = randn(numLatentInputs,numValidationImagesPerClass*numClasses,"single");

TValidation = single(repmat(1:numClasses,[1 numValidationImagesPerClass]));

ZValidation = dlarray(ZValidation,"CB");
TValidation = dlarray(TValidation,"CB");

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    ZValidation = gpuArray(ZValidation);
    TValidation = gpuArray(TValidation);
end

iteration = 0;
start = tic;

% Loop over epochs.
for epoch = 1:numEpochs
    
    % Reset and shuffle data.
    shuffle(mbq);
    
    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        [X,T] = next(mbq);
        
        % Generate latent inputs for the generator network. Convert to
        % dlarray and specify the dimension labels "CB" (channel, batch).
        % If training on a GPU, then convert latent inputs to gpuArray.
        Z = randn(numLatentInputs,miniBatchSize,"single");
        Z = dlarray(Z,"CB");
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            Z = gpuArray(Z);
        end

        % Evaluate the gradients of the loss with respect to the learnable
        % parameters, the generator state, and the network scores using
        % dlfeval and the modelLoss function.
        [~,~,gradientsG,gradientsD,stateG,scoreG,scoreD] = ...
            dlfeval(@modelLoss2,netG,netD,X,T,Z,flipFactor);
        netG.State = stateG;

        % Update the discriminator network parameters.
        [netD,trailingAvgD,trailingAvgSqD] = adamupdate(netD, gradientsD, ...
            trailingAvgD, trailingAvgSqD, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Update the generator network parameters.
        [netG,trailingAvgG,trailingAvgSqG] = ...
            adamupdate(netG, gradientsG, ...
            trailingAvgG, trailingAvgSqG, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Every validationFrequency iterations, display batch of generated images using the
        % held-out generator input.
        if mod(iteration,validationFrequency) == 0 || iteration == 1
            
            % Generate images using the held-out generator input.
            XGeneratedValidation = predict(netG,ZValidation,TValidation);
            
            % Tile and rescale the images in the range [0 1].
            I = imtile(extractdata(XGeneratedValidation), ...
                GridSize=[numValidationImagesPerClass numClasses]);
            I = rescale(I);
            
            % Display the images.
            subplot(1,2,1);
            image(imageAxes,I)
            xticklabels([]);
            yticklabels([]);
            title("Generated Images");
        end
        
        % Update the scores plot.
        subplot(1,2,2)
        addpoints(lineScoreGenerator,iteration,double(scoreG));
        
        addpoints(lineScoreDiscriminator,iteration,double(scoreD));
        
        % Update the title with training progress information.
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        title(...
            "Epoch: " + epoch + ", " + ...
            "Iteration: " + iteration + ", " + ...
            "Elapsed: " + string(D))
        
        drawnow
    end
end


%numObservationsNew = 36;
%idxClass = 1;
for idxClass=1:numClasses
ZNew = randn(numLatentInputs,numObservationsNew,"single");
TNew = repmat(single(idxClass),[1 numObservationsNew]);
%Convert the data to dlarray objects with the dimension labels "SSCB" (spatial, spatial, channels, batch).

ZNew = dlarray(ZNew,"CB");
TNew = dlarray(TNew,"CB");
%To generate images using the GPU, also convert the data to gpuArray objects.

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    ZNew = gpuArray(ZNew);
    TNew = gpuArray(TNew);
end
%Generate images using the predict function with the generator network.

XGeneratedNew = predict(netG,ZNew,TNew);
%Display the generated images in a plot.

%convert to uint8
I = extractdata(XGeneratedNew);
J = im2uint8(I);
J =imresize(J,[size(XTrain,1) size(XTrain,2)]);
IND=sub2ind(size(J,1:2),xp,yp);
for j=1:size(J,4)
    K1 = uint8(ones(size(XTrain,1), size(XTrain,2))*255);
    K2 = K1; K3 = K1;
    J1 = J(:,:,1,j);
    J2 = J(:,:,2,j);
    J3 = J(:,:,3,j);
    K1(IND) = J1(IND); K2(IND) = J2(IND); K3(IND) = J3(IND);
    S = cat(3,K1,K2,K3);
    J(:,:,:,j) = S;
end
%R = round(size(J,4)*Ratio);

XTr=cat(4,XTr,J(:,:,:,1:aug_tr));
YTr=[YTr;categorical(idxClass*ones(aug_tr,1))];
if aug_val>0
XVal=cat(4,XVal,J(:,:,:,aug_tr+1:end));
YVal=[YVal;categorical(idxClass*ones(aug_val,1))];
else
    XVal=[];
    YVal=[];
end
%P = imtile(J);
%P = rescale(P);
%figure; imshow(P);
%title(["Class: " + num2str(idxClass)]);

% matlab code before from the CGAN example
% figure
% I = imtile(extractdata(XGeneratedNew));
% I = rescale(I);
% imshow(I)
% title("Class: " + Out2.YTrain(idxClass))
end
XTr=gather(cat(4,XTrain,XTr));
YTr=[categorical(YTrain);YTr];
XVal=gather(cat(4,XValidation,XVal));
YVal=[categorical(YValidation);YVal];
end




function K = merge(K1,K2)
% K = merge(K1,K2)
% merge two Out files produced by DeepInsight method

K=K1;
K.XTrain = cat(4,K1.XTrain,K2.XTrain);
K.YTrain = [K1.YTrain;K2.YTrain];

if any(isfield(K1,'XValidation'))==1
    if isempty(K1.XValidation)~=1
        K.XValidation = cat(4,K1.XValidation,K2.XValidation);
        K.YValidation = [K1.YValidation;K2.YValidation];
    end
end

if any(isfield(K1,'XTest'))==1
    if isempty(K1.XTest)~=1
        K.XTest = cat(4,K1.XTest,K2.XTest);
        K.YTest = [K1.YTest;K2.YTest];
    end
end
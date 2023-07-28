function [Acc,AUC] = LogReg_FindAccuracy(dset)
% Find Accuracy and/or AUC (for 2-class problem) using Logistic Resgression

current_dir=pwd;
YTrain=[]; YTest=[];
for c=1:dset.class
    YTrain = [YTrain;ones(dset.num_tr(c),1)*c];
    YTest = [YTest;ones(dset.num_tst(c),1)*c];
end

if size(dset.Xtrain,3)==2
    dset.Xtrain = [dset.Xtrain(:,:,1);dset.Xtrain(:,:,2)];
    dset.Xtest  = [dset.Xtest(:,:,1);dset.Xtest(:,:,2)];
    dset.dim = size(dset.Xtrain,1);
elseif size(dset.Xtrain,3)==3
    dset.Xtrain = [dset.Xtrain(:,:,1);dset.Xtrain(:,:,2);dset.Xtrain(:,:,3)];
    dset.Xtest  = [dset.Xtest(:,:,1);dset.Xtest(:,:,2);dset.Xtest(:,:,3)];
    dset.dim = size(dset.Xtrain,1);
end

cd ~/MatWorks/Unsup/liblinear-2.11/matlab/
model=train(double(YTrain),sparse(double(dset.Xtrain')),['-s 0','liblinear_options',]);
[predicted_label,acc,probs]=predict(double(YTest),sparse(double(dset.Xtest')),model,['-b 1']);
cd(current_dir);
Acc = acc(1);
if max(double(YTest))==2
    [a,b,c,AUC] = perfcurve(YTest,probs(:,2),'2');
else
    AUC=[];
end

end
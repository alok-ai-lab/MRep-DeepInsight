function PrepareDataForLopez
Out2 = load('Out2.mat');
Out2.YTest=single(Out2.YTest);
Out2.YTrain=single(Out2.YTrain);
Out2.YValidation=single(Out2.YValidation);
if any(strcmp('orgYTrain',fieldnames(Out2)))
    Out2=rmfield(Out2,'orgYTrain');
end
if any(strcmp('orgYValidation',fieldnames(Out2)))
    Out2=rmfield(Out2,'orgYValidation');
end

if size(Out2.XTrain,3)==1
    Out2.XTrain = cat(3,Out2.XTrain,Out2.XTrain,Out2.XTrain);
    Out2.XTest  = cat(3,Out2.XTest,Out2.XTest,Out2.XTest);
    Out2.XValidation = cat(3,Out2.XValidation,Out2.XValidation,Out2.XValidation);
elseif size(Out2.XTrain,3)==2
        Out2.XTrain = cat(3,Out2.XTrain(:,:,1,:),Out2.XTrain(:,:,2,:),Out2.XTrain(:,:,1,:));
    Out2.XTest  = cat(3,Out2.XTest(:,:,1,:),Out2.XTest(:,:,2,:),Out2.XTest(:,:,1,:));
    Out2.XValidation = cat(3,Out2.XValidation(:,:,1,:),Out2.XValidation(:,:,2,:),Out2.XValidation(:,:,1,:));
end

save('Out2.mat','-struct','Out2','-v7.3');

function Out = Map_Assignment(Out,Parm)
% Out = Map_Assignment(Out,Parm)
% function to map assignment

if length(Out.xp)>3000
    disp(['Warning! assignment may take longer ' ...
        'than usual time for dim > 3000.']);
end

fprintf('\nAssignment started..\n');	
zp_tmp = assignment([Out.xp;Out.yp],'pixel',max([Out.A Out.B]));
Out.xp = zp_tmp(:,1)';
Out.yp = zp_tmp(:,2)';
clear zp_tmp   
fprintf('Assignment completed\n');


fprintf('\n Pixels: %d x %d\n',Out.A,Out.B);

dset=load('NormalizedData.mat');
dset=dset.dset;

Out.XTrain=[];
Out.XValidation=[];
if Parm.Augment==1
    Out.YTrain=Out.orgYTrain;
    Out.YValidation=Out.orgYValidation;
end

fprintf('\nMapping of assignment begins\n');

for dsz = 1:size(dset.Xtrain,3)
    for j=1:length(Out.YTrain)
        Out.XTrain(:,:,dsz,j) = ConvPixel(dset.Xtrain(:,j,dsz),Out.xp,Out.yp,Out.A,Out.B,Out.Base,0);
    end
end
dset.Xtrain=[];

for dsz = 1:size(dset.Xtest,3)
    for j=1:length(Out.YTest)
        Out.XTest(:,:,dsz,j) = ConvPixel(dset.Xtest(:,j,dsz),Out.xp,Out.yp,Out.A,Out.B,Out.Base,0);
    end
end
dset.Xtest=[];

for dsz=1:size(dset.XValidation,3)
    for j=1:length(Out.YValidation)
        Out.XValidation(:,:,dsz,j) = ConvPixel(dset.XValidation(:,j,dsz),Out.xp,Out.yp,Out.A,Out.B,Out.Base,0);
    end
end
dset.XValidation=[];
Out.C = size(Out.XTrain,3);

if Parm.Augment==1
    if Parm.AugMeth==1
        [Out.XTrain,Out.YTrain] = augmentDeepInsight(Out.XTrain,Out.YTrain);
        if Parm.ValidRatio>0
            [Out.XValidation,Out.YValidation] = augmentDeepInsight(Out.XValidation,Out.YValidation);
        end
    elseif Parm.AugMeth==2
        [Out.XTrain,Out.YTrain] = augmentDeepInsight2(Out.XTrain,Out.YTrain,Parm.aug_tr);
        if Parm.ValidRatio>0
            [Out.XValidation,Out.YValidation] = augmentDeepInsight2(Out.XValidation,Out.YValidation,Parm.aug_val);
        end
    end
end

fprintf('\nMapping of assignment completed.\n');
close all;
end
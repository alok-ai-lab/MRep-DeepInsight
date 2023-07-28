function dset = Norm3(dset)

Layers = size(dset.Xtrain,3);
dset.Xtrain = pagetranspose(dset.Xtrain);
dset.Xtest = pagetranspose(dset.Xtest);

for j=1:Layers
    %dset.Xtrain = dset.Xtrain';	
    M = mean(dset.Xtrain(:,:,j));
    dset.Xtrain(:,:,j) = (dset.Xtrain(:,:,j)-M);
    %Std = std(dset.Xtrain(:,:,j));
    %Std(Std==0)=1;
    %Std = max(Std);
    %dset.Xtrain(:,:,j) = dset.Xtrain(:,:,j)./Std;
    dset.Xtrain(:,:,j) = dset.Xtrain(:,:,j)./vecnorm(dset.Xtrain(:,:,j)')';
    %dset.Xtrain = dset.Xtrain';

    %dset.Xtest = dset.Xtest';
    dset.Xtest(:,:,j) = (dset.Xtest(:,:,j)-M);
    %dset.Xtest(:,:,j) = dset.Xtest(:,:,j)./Std;
    dset.Xtest(:,:,j) = dset.Xtest(:,:,j)./vecnorm(dset.Xtest(:,:,j)')';
    %dset.Xtest = dset.Xtest';
end
if any(isfield(dset,'Set'))==0
    dset.Set = 'unknown';
end
dset.Set = [dset.Set,'_norm3'];
dset.Xtrain = pagetranspose(dset.Xtrain);
dset.Xtest = pagetranspose(dset.Xtest);

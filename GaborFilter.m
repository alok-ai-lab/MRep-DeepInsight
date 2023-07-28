function Out = GaborFilter(Out,Workers)
% Out = GaborFilter(Out,Workers)
if nargin<2
    Workers=10;
end
disp('Gabor filter is used');
% set up parallel pool
poolobj = gcp('nocreate');
delete(poolobj);
parpool('local',Workers);
poolobj = gcp('nocreate');

wavelength=2;
orientation=0;
g = gabor(wavelength,orientation);

Layers = size(Out.XTrain,3);

clear GB
for lyr=1:Layers
    parfor j=1:size(Out.XTrain,4)
        GB(:,:,1,j) = im2uint8(imgaborfilt(im2double(Out.XTrain(:,:,lyr,j)), g));
    end
    Out.XTrain(:,:,lyr,:)=GB;
    clear GB
end

if any(strcmp('XValidation',fieldnames(Out)))==1
    if isempty(Out.XValidation)~=1
        for lyr=1:Layers
            parfor j=1:size(Out.XValidation,4)
                GB(:,:,1,j) = im2uint8(imgaborfilt(im2double(Out.XValidation(:,:,lyr,j)), g));
            end
            Out.XValidation(:,:,lyr,:)=GB;
            clear GB
        end
    end
end

if any(strcmp('XTest',fieldnames(Out)))==1
    if isempty(Out.XTest)~=1
        for lyr=1:Layers
            parfor j=1:size(Out.XTest,4)
                GB(:,:,1,j) = im2uint8(imgaborfilt(im2double(Out.XTest(:,:,lyr,j)), g));
            end
            Out.XTest(:,:,lyr,:)=GB;
            clear GB
        end
    end
end

delete(poolobj);

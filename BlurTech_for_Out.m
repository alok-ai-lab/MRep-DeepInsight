function Out = BlurTech_for_Out(Out,Workers)
% Input: Out1 or Out2 file
%        
% Output: Out using blurring of nearby pixels
% if Out.OverWrite = 1 then empty places will be filled with 'blur' values even
% if these points are [xp,yp]. Basically, same as BlurTech(M) % default Step=4

if nargin < 2
   Workers = 20;
end

if any(strcmp('OverWrite',fieldnames(Out)))~=1
   Out.OverWrite = 'no';
end

disp('Blur technique has been used.')

% set up parallel pool
poolobj = gcp('nocreate');
delete(poolobj);
parpool('local',Workers);
poolobj = gcp('nocreate');

if any(strcmp('xp',fieldnames(Out)))==1
    Exist=1;
else
    Exist=0;
end
if strcmp(Out.OverWrite,'yes')==1
    Exist=0;
end

clear MB
parfor j=1:size(Out.XTrain,4)
    if Exist==1
        MB(:,:,:,j)=BlurTech(Out.XTrain(:,:,:,j),Out.xp,Out.yp);
    else
        MB(:,:,:,j)=BlurTech(Out.XTrain(:,:,:,j));
    end
end
Out.XTrain=MB;
clear MB
parfor j=1:size(Out.XTest,4)
    if Exist==1
        MB(:,:,:,j)=BlurTech(Out.XTest(:,:,:,j),Out.xp,Out.yp);
    else
        MB(:,:,:,j)=BlurTech(Out.XTest(:,:,:,j));
    end
end
Out.XTest=MB;
clear MB
if any(strcmp('XValidation',fieldnames(Out)))==1
    if isempty(Out.XValidation)~=1
        parfor j=1:size(Out.XValidation,4)
            if Exist==1
                MB(:,:,:,j)=BlurTech(Out.XValidation(:,:,:,j),Out.xp,Out.yp);
            else
                MB(:,:,:,j)=BlurTech(Out.XValidation(:,:,:,j));
            end
        end
        Out.XValidation=MB;
        clear MB
    end
end
delete(poolobj);


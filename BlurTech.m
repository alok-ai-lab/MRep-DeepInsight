function MB = BlurTech(M,xp,yp,STEP)
% blurring nearby pixels
% MB = BlurTech(M,xp,yp,STEP)
% maximum of 5 steps
%
% usgae 1)
% MB = BlurTech(M); it finds xp and yp; default STEP=4;
%
% usage 2)
% MB = BlurTech(M,xp,yp); it uses characteristics pixel locations; STEP=4
%
% usgae 3)
% MB = BlurTech(M,xp,yp,STEP); provide all values STEP values between [1,5]

%determine the input type
%isa(M,'double'); %double
TypeUint8= isa(M,'uint8'); %unsigned int8
if TypeUint8==1
    M=im2double(M);
end

if nargin<2
    ind=1:size(M,1)*size(M,2);
    if size(M,3)==1
        [xp,yp]=ind2sub(size(M),ind(reshape(M<1,1,size(M,1)*size(M,2))));
    elseif size(M,3)==3
        [xp,yp]=ind2sub(size(M(:,:,1)),ind(reshape(M(:,:,1)<1,1,size(M,1)*size(M,2))));
    end
    STEP=4;
elseif nargin<4
    STEP=4;
end
% Step=1...5 (maximum can be set to five)
for step=1:STEP
    M = BlurTechStep(M,xp,yp,step);
end
if TypeUint8==1
    MB=im2uint8(M);
else
    MB=M;
end
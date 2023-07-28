function Y = overlap_reduction(data,dist)
% ovrelap_reduction as per Eqs(5) & (6) of Manuel's paper
% Only for tsne
r=size(data,1);
z1=[];z2=[];
data=[data;data;data];
if r<5000
    rng("default");
    Y=tsne(data,'Algorithm','exact','Distance',dist);
else
    disp('this is an expensive procedure if dimensionality is large');
    disp('it requires a lot of cpu time');
    rng("default");
    Y=tsne(data,'Algorithm','barneshut','Distance',dist);
end
clear data
z1=[];z2=[];
for j=1:3
    z1=[z1,Y(1+(j-1)*r:r*j,1)];
    z2=[z2,Y(1+(j-1)*r:r*j,2)];
end
z1=mean(z1')';
z2=mean(z2')';
Y = [z1,z2];

if r<5000
else
end
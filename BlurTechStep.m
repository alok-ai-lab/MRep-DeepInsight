function MB = BlurTechStep(M,xp,yp,step)
% blurring nearby pixels
% B = BlurTechStep(M,xp,yp)
[A,B]=size(M,1:2);
M = 1-M;

ZP=[xp;yp];

%step=1;
if step==1
    z1=[xp+1,xp,  xp,  xp-1];
    z2=[yp  ,yp+1,yp-1,yp  ];
end

%step2
if step==2
    z1=[xp+2,xp+1,xp  ,xp-1,xp-2,xp-1,xp  ,xp+1];
    z2=[yp  ,yp+1,yp+2,yp+1,yp  ,yp-1,yp-2,yp-1];
end

%step3
if step==3
    z1=[xp+3,xp+2,xp+2,xp+2,xp+2,xp+1,xp+1,xp  ,xp  ,xp-1,xp-1,xp-2,xp-2,xp-2,xp-2,xp-3];
    z2=[yp  ,yp+2,yp+1,yp-1,yp-2,yp+2,yp-2,yp+3,yp-3,yp+2,yp-2,yp+2,yp+1,yp-1,yp-2,yp  ];
end

%step4
if step==4
    z1=[xp-4,xp-4,xp-4,xp-3,xp-3,xp-3,xp-3,xp-3,xp-3,xp-2,xp-2,xp-1,xp-1,xp-1,xp-1,xp  ,xp  ];
    z2=[yp-1,yp  ,yp+1,yp-3,yp-2,yp-1,yp+1,yp+2,yp+3,yp-3,yp+3,yp-4,yp-3,yp+3,yp+4,yp-4,yp+4];
    z1=[z1,xp+1,xp+1,xp+1,xp+1,xp+2,xp+2,xp+3,xp+3,xp+3,xp+3,xp+3,xp+3,xp+4,xp+4,xp+4];
    z2=[z2,yp-4,yp-3,yp+3,yp+4,yp-3,yp+3,yp-3,yp-2,yp-1,yp+1,yp+2,yp+3,yp-1,yp  ,yp+1];
end

if step==5
    z1=[xp-5,xp-5,xp-5,xp-4,xp-4,xp-4,xp-4,xp-4,xp-4,xp-3,xp-3,xp-2,xp-2,xp-1,xp-1,xp  ,xp  ];
    z2=[yp-1,yp  ,yp+1,yp-4,yp-3,yp-2,yp+2,yp+3,yp+4,yp-4,yp+4,yp-4,yp+4,yp-5,yp+5,yp-5,yp+5];
    z1=[z1,xp+1,xp+1,xp+2,xp+2,xp+3,xp+3,xp+4,xp+4,xp+4,xp+4,xp+4,xp+4,xp+5,xp+5,xp+5];
    z2=[z2,yp-5,yp+5,yp-4,yp+4,yp-4,yp+4,yp-4,yp-3,yp-2,yp+2,yp+3,yp+4,yp-1,yp  ,yp+1];
end

% check if created pixels are not out of the frame
%cond1 = (z1>0);
%cond2 = (z2>0);
%cond3 = (z1<=A);
%cond4 = (z2<=B);

% check if newly created position is not overlapping with the
% characteristic pixels (i.e. [xp,yp])
z = [z1;z2];
%cond5 = ~ismember(z',ZP','rows')';

%cond = cond1 & cond2 & cond3 & cond4 & cond5;

INX = 1:length(z1);
%INX = INX(cond);
INX = INX((z1>0)&(z2>0)&(z1<=A)&(z2<=B)&(~ismember(z',ZP','rows')')); % all 5 conditions at once
if size(M,3)==1
  for k=INX
    z=[z1(k);z2(k)];
    if mod(k,length(xp))==0
        n=length(xp);
    else
        n=mod(k,length(xp));    
    end
    %if M(xp(n),yp(n))<1
        blur = (M(xp(n),yp(n)))/((sqrt(exp(1)))^(step-1));
        % blurring with maximum value
        if blur > M(z(1),z(2))
            M(z(1),z(2)) = blur;    
        end 
    %end
  end
elseif size(M,3)==3
  for k=INX
    z=[z1(k);z2(k)];
    if mod(k,length(xp))==0
        n=length(xp);
    else
        n=mod(k,length(xp));    
    end

    blur = (M(xp(n),yp(n),:))/((sqrt(exp(1)))^(step-1));
    blur = reshape(blur,[3,1]);
    % blurring with maximum value
    for layers=1:3
        %if M(xp(n),yp(n),layers)<1
            if blur(layers) > M(z(1),z(2),layers)
                M(z(1),z(2),layers) = blur(layers);    
            end 
        %end
    end
  end
end
MB=1-M;
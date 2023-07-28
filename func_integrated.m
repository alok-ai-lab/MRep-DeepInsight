function [Duplicate,InputSz1,InputSz2,InputSz3,Init_dim,SET] = func_integrated(Parm);
% [Duplicate,InputSz1,InputSz2,InputSz3,Init_dim,SET] = func_integrated(Parm);
% Prepare data in an integrated way; i.e., by combining various methods
% e.g.
% Parm.UseIntegrate = 'yes';
% Parm.integrate = {'tsne','euclidean','tsne','hamming',...
%                      'umap','assignment','blur'};

Parm.Assignment='no';
Parm.Blur='no';

Tsne = strcmpi(Parm.integrate,'tsne');
Assign = strcmpi(Parm.integrate,'assignment');
Blur = strcmpi(Parm.integrate,'blur');
Gb = strcmpi(Parm.integrate,'gabor');

MethodFile_generated = 'no';

if sum(Tsne)>0
    cnt=1;
    for j=1:length(Parm.integrate)
        if Tsne(j)==1
            if j<length(Parm.integrate)
                Dist{cnt} = Parm.integrate(j+1);
                cnt=cnt+1;
            else
                disp('Error: No distance of tsne is assigned!');
            end
        end
    end
    Parm.Method='tsne';
    for j=1:length(Dist)
        Parm.Dist = cell2mat(Dist{j});
        [InputSz1,InputSz2,InputSz3,Init_dim,SET] = func_Prepare_Data(Parm);
        if Parm.Norm==1
            MethodOut=load('Out1.mat');
        elseif Parm.Norm==2
            MethodOut=load('Out2.mat');
        end
        if sum(Gb)>0
            GaborOut = GaborFilter(MethodOut,20);
            save('GaborOut.mat','-struct','GaborOut','-v7.3');
            clear GaborOut
        end
        if sum(Blur)>0
            BlurOut = BlurTech_for_Out(MethodOut,20);
            save('BlurOut.mat','-struct','BlurOut','-v7.3');
            clear BlurOut
        end
        if sum(Assign)>0
            AssignOut = Map_Assignment(MethodOut,Parm);
            save('AssignOut.mat','-struct','AssignOut','-v7.3');
            clear AssignOut
        end
        if sum(Blur)>0 & sum(Assign)>0
            AssignOut=load('AssignOut.mat');
            BlurOutAssign = BlurTech_for_Out(AssignOut,20);
            save('BlurOutAssign.mat','-struct','BlurOutAssign','-v7.3');
            clear BlurOutAssign AssignOut
        end
        %merging (xp yp values will not longer be correct)
        if sum(Blur)>0
            BlurOut=load('BlurOut.mat');
            MethodOut = merge(MethodOut,BlurOut);
            clear BlurOut
            unix(['rm BlurOut.mat']);
        end
        if sum(Assign)>0
            AssignOut=load('AssignOut.mat');
            MethodOut = merge(MethodOut,AssignOut);
            clear AssignOut
            unix(['rm AssignOut.mat']);
        end
        if sum(Blur)>0 & sum(Assign)>0
            BlurOutAssign=load('BlurOutAssign.mat');
            MethodOut = merge(MethodOut,BlurOutAssign);
            clear BlurOutAssign
            unix(['rm BlurOutAssign.mat']);
        end
        if sum(Gb)>0
            GaborOut=load('GaborOut.mat');
            MethodOut = merge(MethodOut,GaborOut);
            clear GaborOut
            unix(['rm GaborOut.mat']); 
        end
        if j==1
            save('MethodOut.mat','-struct','MethodOut','-v7.3');
            MethodFile_generated = 'yes';
            clear MethodOut
        else
            MethodOutprev = load('MethodOut.mat');
            MethodOut = merge(MethodOut,MethodOutprev);
            clear MethodOutprev
            save('MethodOut.mat','-struct','MethodOut','-v7.3');
            clear MethodOut
        end
    end
end

Methods={'umap','kpca','pca'};
if strcmp(Parm.TightRep.Perm,'yes')==1
   Methods={'umap','kpca','pca','direct'}; 
end

for k=1:length(Methods)
  other=strcmpi(Parm.integrate,Methods{k});

  if sum(other)>0
    Parm.Method=Methods{k};
    [InputSz1,InputSz2,InputSz3,Init_dim,SET] = func_Prepare_Data(Parm);
    if Parm.Norm==1
        MethodOut=load('Out1.mat');
    elseif Parm.Norm==2
        MethodOut=load('Out2.mat');    
    end
    if sum(Gb)>0
       GaborOut = GaborFilter(MethodOut,20);
       save('GaborOut.mat','-struct','GaborOut','-v7.3');
       clear GaborOut
    end
    if sum(Blur)>0
        BlurOut = BlurTech_for_Out(MethodOut,20);
        save('BlurOut.mat','-struct','BlurOut','-v7.3');
        clear BlurOut
    end
    if sum(Assign)>0
        AssignOut = Map_Assignment(MethodOut,Parm);
        save('AssignOut.mat','-struct','AssignOut','-v7.3');
        clear AssignOut
    end
    if sum(Blur)>0 & sum(Assign)>0
        AssignOut=load('AssignOut.mat');
        BlurOutAssign = BlurTech_for_Out(AssignOut,20);
        save('BlurOutAssign.mat','-struct','BlurOutAssign','-v7.3');
        clear BlurOutAssign AssignOut
    end
    %merging (xp yp values will not longer be correct)
    if sum(Blur)>0
        BlurOut=load('BlurOut.mat');
        MethodOut = merge(MethodOut,BlurOut);
        clear BlurOut
        unix(['rm BlurOut.mat']);
    end
    if sum(Assign)>0
        AssignOut=load('AssignOut.mat');
        MethodOut = merge(MethodOut,AssignOut);
        clear AssignOut
        unix(['rm AssignOut.mat']);
    end
    if sum(Blur)>0 & sum(Assign)>0
        BlurOutAssign=load('BlurOutAssign.mat');
        MethodOut = merge(MethodOut,BlurOutAssign);
        clear BlurOutAssign
        unix(['rm BlurOutAssign.mat']);
    end
    if sum(Gb)>0
        GaborOut=load('GaborOut.mat');
        MethodOut = merge(MethodOut,GaborOut);
        clear GaborOut
        unix(['rm GaborOut.mat']); 
    end
    
    if strcmp(MethodFile_generated,'no')
        MethodOut.Method=define_integrated_method(Parm);
        save('MethodOut.mat','-struct','MethodOut','-v7.3');
        clear MethodOut
        MethodFile_generated='yes';
    else
        MethodOutprev = load('MethodOut.mat');
        MethodOut = merge(MethodOut,MethodOutprev);
        clear MethodOutprev
        MethodOut.Method=define_integrated_method(Parm);
        save('MethodOut.mat','-struct','MethodOut','-v7.3');
        clear MethodOut
    end
  end
end

if Parm.Norm==1
    unix(['mv MethodOut.mat Out1.mat']);
elseif Parm.Norm==2
    unix(['mv MethodOut.mat Out2.mat']);
end


Umap=strcmpi(Parm.integrate,'umap');
Kpca=strcmpi(Parm.integrate,'kpca');
Pca=strcmpi(Parm.integrate,'pca');
if strcmp(Parm.TightRep.Perm,'yes')==1
    Direct=strcmpi(Parm.integrate,'direct');
end

Duplicate = sum(Tsne)+any(Umap)+any(Kpca)+any(Pca);
if strcmp(Parm.TightRep.Perm,'yes')==1
    Duplicate = sum(Tsne)+any(Umap)+any(Kpca)+any(Pca)+any(Direct);
end
Duplicate = any(Assign)*Duplicate + Duplicate;
Duplicate = any(Blur)*Duplicate + Duplicate;
if strcmp(Parm.TightRep.Perm,'yes')==0
    Duplicate = Duplicate + any(Gb)*(sum(Tsne) + sum(Umap) + ...
    sum(Kpca) + sum(Pca));
else
    Duplicate = Duplicate + any(Gb)*(sum(Tsne) + sum(Umap) + ...
    sum(Kpca) + sum(Pca) + sum(Direct));
end


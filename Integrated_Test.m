function [Accuracy,AUC,C,prob,...
    Accuracy_wgt,AUC_wgt,C_wgt,prob_wgt] = Integrated_Test(prob,Duplicate)
% test results using integrated method
% prob is a probability function produced by func_TrainModel.m.
%
% Accuracy, AUC, C and prob are measures of Equal Weighted method.
%
% Accuracy_Wgt, AUC_Wgt, C_Wgt and prob_Wgt are measure of LogReg weighted
% method.

Len=size(prob.test,1);
%Duplicate = 4;
Len=Len/Duplicate;

% Equal weights
prob_ew=zeros(Len,size(prob.test,2));
for j=1:Duplicate
    prob_ew=prob.test((j-1)*Len+1:j*Len,:)+prob_ew;
end
prob_ew=prob_ew/Duplicate;

for j=1:Len
    [r,YPred(j)]=max(prob_ew(j,:));
end
YPred=YPred';
Accuracy = 100*mean(double(prob.YTest(1:Len))==YPred);
fprintf('\nIntegrated accuracy: %6.2f\n',Accuracy);
YPred=categorical(YPred);

C = confusionmat(prob.YTest(1:Len),YPred);
if size(C,1)==2
    [a,b,YPred,AUC] = perfcurve(prob.YTest(1:Len),prob_ew(:,2),'2');
    fprintf('\nIntegrated AUC: %6.4f\n',AUC);
else
    AUC=[];
end

% weighted with log reg
% Validation data is needed!
if isfield(prob,'val')==0
    prob.val=[];
end
if size(prob.val,1)>0
    Xtr=[];Xts=[];
    LenV = size(prob.val,1)/Duplicate;
    for j=1:Duplicate
        Xtr=[Xtr,prob.val((j-1)*LenV+1:j*LenV,:)];
        Xts=[Xts,prob.test((j-1)*Len+1:j*Len,:)];
    end
    Ytr=prob.YValidation(1:LenV);
    Yts=prob.YTest(1:Len);

    current_dir=pwd;
    cd ~/MatWorks/Unsup/liblinear-2.11/matlab/
    model=train(double(Ytr),sparse(double(Xtr)),['-s 0','liblinear_options',]);
    [yhat,acc,prob_wgt]=predict(double(Yts),sparse(double(Xts)),model,['-b 1']);
    C_wgt=confusionmat(Yts,categorical(round(yhat)));
    cd(current_dir);
    Accuracy_wgt = acc(1);
    fprintf('\nWeighted integrated accuracy: %6.2f\n',Accuracy_wgt);

    if size(C_wgt,1)==2
        prob_wgt=prob_wgt(:,2);
        [a,b,c,AUC_wgt] = perfcurve(Yts,prob_wgt,'2');
        fprintf('\nWeighted integrate AUC: %6.4f\n',AUC_wgt);
    else
        AUC_wgt=[];
    end

else
    Accuracy_wgt=[];
    AUC_wgt=[];
    C_wgt=[];
    prob_wgt=[];
end
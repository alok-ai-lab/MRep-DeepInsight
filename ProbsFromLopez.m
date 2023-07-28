function Accuracy = ProbsFromLopez
test = load('probs.mat');
data = load('RNA_ATAC3.mat');
prob.test=test.probs;
prob.YTest=categorical(data.YTest);
addpath("/home/aloks/MatWorks/Unsup/DeepInsight3D_pkg/");
Duplicate=findDuplicate(prob.YTest);
clear data
[Accuracy,AUC,C,prob,...
        Accuracy_wgt,AUC_wgt,C_wgt,prob_wgt] = Integrated_Test(prob,Duplicate);
end
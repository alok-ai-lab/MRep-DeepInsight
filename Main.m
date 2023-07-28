close all
clear all

for j=1:1%7
    fprintf('\n dataset %d\n',j);
    [AUC{j},C{j},Accuracy{j},ValErr{j},Genes{j}] = DeepInsight3D(j); % DeepInsight3D model
    %[Genes{j},GenesPerClass{j}]=func_GeneSel_SavedModels(j);
    close all;
    close all hidden;
    AUC{j}
end

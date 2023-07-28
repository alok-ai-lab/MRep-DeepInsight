function [Acc,AUC] = RandomForest(dset)

nTrees=500;
labels=[]; labels_tst=[];
for j=1:dset.class
    labels=[labels;ones(dset.num_tr(j),1)*j];
    labels_tst=[labels_tst;ones(dset.num_tst(j),1)*j];
end
labels=categorical(labels);
B = TreeBagger(nTrees,dset.Xtrain',labels,'Method','classification');
[pred,probs] = B.predict(dset.Xtest');
pred = str2double(pred);
Acc = sum(pred==labels_tst)/length(labels_tst);
if dset.class==2
    [a,b,c,AUC] = perfcurve(labels_tst,probs(:,2),'2');
else
    AUC=[];
end
end


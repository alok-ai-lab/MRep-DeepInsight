function Duplicate = findDuplicate(labels)
labels=double(labels);
q=labels>=max(labels);
for j=1:length(q)-1
    P(j)=and(q(j),~q(j+1));
end
[r,c]=max(P);
Duplicate = length(q)/c(1);
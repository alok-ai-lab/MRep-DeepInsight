function Duplicate = findDuplicate_Example(integrate)

%	integrate = {'tsne','hamming','umap','assignment','gabor','blur'};%{'tsne','hamming','assignment','umap','gabor'}; % add as required

Tsne=strcmpi(integrate,'tsne');
Umap=strcmpi(integrate,'umap');
Kpca=strcmpi(integrate,'kpca');
Pca=strcmpi(integrate,'pca');
Gb=strcmpi(integrate,'gabor');
Assgn=strcmpi(integrate,'assignment');
Blurry=strcmpi(integrate,'blur');
Duplicate = sum(Tsne)+any(Umap)+any(Kpca)+any(Pca);
Duplicate = any(Assgn)*Duplicate + Duplicate;
Duplicate = any(Blurry)*Duplicate + Duplicate;
Duplicate = Duplicate + any(Gb)*(sum(Tsne) + sum(Umap) + ...
    sum(Kpca) + sum(Pca));

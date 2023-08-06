# MRep-DeepInsight
Multiple Representation DeepInsight package

A step-by-step guide to run MRep-DeepInsight codes.
Language: MATLAB

MRep-DeepInsight is built upon the previous package [DeepInsight3D](https://github.com/alok-ai-lab/DeepInsight3D_pkg). Therefore, all preceding packages be executed using `MRep-DeepInsight` package, which includes [DeepInsight](https://github.com/alok-ai-lab/DeepInsight), [DeepFeature](https://github.com/alok-ai-lab/deepfeature) and [DeepInsight3D](https://github.com/alok-ai-lab/DeepInsight3D_pkg).

Setting up the `Parameters.m` file enables one to run the package in various ways including all the previously developed packages.

The MRep-DeepInsight3D package has 2 main components. 1) conversion of tabular data to image samples, and 2) processing images to the convolutional neural network (CNN). 

Figure 1 depicts the MRep-DeepInsight approach, where part a shows the transformation phase, part b shows the model estimation phase, and part c illustrates the model analysis phase.

<img src="https://github.com/alok-ai-lab/Supplementary/blob/main/Fig1_A4.png?raw=true" width="600" height="900">

**Figure 1:** An overview of the MRep-DeepInsight approach

### MRep-DeepInsight tested on:
OS: Linux Ubuntu 20.04;
Matlab version: 2022b;
GPU A100 (4 parallel);

# Reference 
Sharma A, Lopez Y, Jia S, Lysenko A, Boroevich KA, Tsunoda T, Multi-representation DeepInsight: an improvement on tabular data analysis, 2023 [Paper link](https://www.biorxiv.org/content/10.1101/2023.08.02.551620v1) 

## Download and Install

1. Download the Matlab package MRep-DeepInsight.tar.gz or the entire directory from the link above. Store it in your working directory. Gunzip and untar as follows:

    ```Matlab
    >> gunzip MRep-DeepInsight.tar.gz
    >> tar -xvf MRep-DeepInsight.tar
    ```

    **Note**:
   1) Install R/Python software to use UMAP (see `umap_Rmatlab.m`).  
   2) This package also uses liblinear tools. Therefore, load liblinear package and set the path correctly in Line 54, [Integrated_Test.m](https://github.com/alok-ai-lab/MRep-DeepInsight/blob/main/Integrated_Test.m).
      Alternatively, comment out call for weighted integrated accuracy & AUC (Lines 38-75, [Integrated_Test.m](https://github.com/alok-ai-lab/MRep-DeepInsight/blob/main/Integrated_Test.m)).

3. Download the example dataset from the following link (caution: data size is 17MB):

    [dataset4.mat](https://github.com/alok-ai-lab/MRep-DeepInsight/blob/main/Data/dataset4.mat)
   
   Move the dataset4.mat to the folder `MRep-DeepInsight/Data/`. 

   The dataset is given in the struct format of Matlab. Use any other data (binary class or multi-class) in a similar struct format for MRep-DeepInsight.

4. Download and Install example CNN net such as ResNet-50 in Matlab, see details about ResNet-50 from MathWorks [link](https://www.mathworks.com/help/deeplearning/ref/resnet50.html). You may use different nets as desired.

5. Executing the MRep-DeepInsight: all the codes should be run in the folder ../MRep-DeepInsight/, if you want to run in a different folder then addpath to appropriate directories in Matlab

## Manifold and supplement techniques for MRep-DeepInsight
The following mapping techniques can be used for MRep-DeepInsight

$\textcolor{red}{\textsf{1) tSNE:}}$ with 11 distances- {Euclidean, Correlation, Minkowski, Standard Euclidean, Spearman, Jaccard, Cityblock, Hamming, Chebychev, Mahalanobis, Cosine};

$\textcolor{green}{\textsf{2) UMAP}}$ 

$\textcolor{blue}{\textsf{3) Kernel PCA (KPCA)}}$ 

$\textcolor{purple}{\textsf{4) PCA}}$ 

The supplement techniques modify the mappings of manifold techniques. These techniques can't be run independently and therefore at least one manifold technique is required to use. The supplement techniques are:
```diff
-1) Gabor filtering
+2) Blurring technique
-3) Assignment distribution algorithm
```

### Parameter settings for MRep-DeepInsight
In order to use one or a combination of the above techniques, please set the following parameter correctly.
1) Open `Parameters.m` file.
2) Change (Line 5) `Parm.UseIntegrate='yes';`.
   Options are either `yes` or `no`. This will trigger the `MRep-DeepInsight` methodology, and override `Parm.Method` option.
   
4) Change (Line 133) **$\textcolor{red}{\textsf{Parm.integrate}}$** as required. Some examples are given here under:
   
   **ex-1)**    Use both tSNE with hamming distance and tSNE with Euclidean distance:

            `Parm.integrate={'tsne','hamming','tsne','euclidean'};`

      i.e. Define distance after tSNE technique {tsne, distance,...}      

   **ex-2)**    Use tSNE with hamming distance and UMAP technique:
   
            `Parm.integrate={'tsne','hamming','umap'};`

     i.e. umap does not require to define any distance. Same is true for KPCA and PCA.
   
   **ex-3)**    Use UMAP. Kernel PCA and PCA:
   
            `Parm.integrate={'umap','kpca','pca'};`

   **ex-4)**    Use tSNE with cosine, UMAP, Gabor, Blurring, Assignment and tsne with Chebychev:
   
            `Parm.integrate={'tsne','cosine','umap','gabor','blur','assignment','tsne','chebychev'};`

    Please note the term `blur` is used for `Blurring technique`; and `'assignment` is used for Assignment distribution technique.

### Example 1: classification of tabular data using the MRep-DeepInsight model
In this example, tabular data with 2539 dimensions is used. It has 1178 training samples and 131 test samples. It is divided into two classes, namely Alzheimer's Disease (AD) and Normal Control (NC). First, the dataset is converted to images by the MRep-DeepInsight converter. Then the CNN net (resnet50) has been trained. The performance evaluation, in terms of accuracy, is done on the test set of the data.

1. File: open the Example1.m file in the Matlab Editor.

2. In order to activate MRep-DeepInsight pipeline, set true the variable `Parm.UseIntegrate=yes` in the `Parameters.m` file.
   
3. Depending upon how many representations are required, setup `Parm.integrate` in the `Parameters.m` file.
   For e.g. define `Parm.integrate={'tsne','hamming','tsne','cosine'}`, i.e., two representations (m=2).

4. For a quick test of codes, use 1 objective function; i.e., `Parm.MaxObj=1`. The recommended MaxObj value is 25 or over.

5. Set up other parameters as required by changing the `Parameters.m` file, otherwise leave all as default.
   However, based on your hardware requirements, change `Parm.miniBatchSize` to lower value if encountering memory problems (we use the default value as 1024) and also `Parm.ExecutionEnvironment` (default is multi-gpu). If you don't want to see the training progress plot produced by CNN training, then set `Parm.trainingPlot=none`. 

6. Dataset calling: since the dataset name is `dataset4.mat`, set the variable `DSETnum=4` (at Line 17 of Example1.m) has been used. If the name of the dataset is `datasetX.m` then variable `DSETnum` should be set as `X`.

7. Example1.m file uses updated function DeepInsight3D.m. This function has two parts: 1) tabular data to image conversion using `func_Prepare_Data.m` (supports previously developed converters) and `func_integrate.m` (supports MRep-DeepInsight), and 2) CNN training using resent50 (default or change as required) using `func_TrainModel.m`.

8. The output is AUC (for 2-class problem only), C (confusion matrix) and Accuracy of the test set (at Line 28). It also gives ValErr which is the validation error.

9. By default, trained CNN models (such as model.mat, 0*.mat) and converted tabular data to images (either Out1.mat or Out2.mat) will be saved in folder /Models/Run4/ (since DSETnum=4; if DSETnum=N then saved in ../RunN/) and figures will be stored in folder /FIGS/Run4/ (since DSETnum=4). The saving of files is done by calling the functions `func_SaveModels.m` and `func_SaveFigs.m`

10. The execution results are stored in the file `DeepInsight3D_Results.txt` which is stored in the folder /MRep-DeepInsight/.

11. A few messages will be displayed by running Example1.m on the Command Window of Matlab, such as

    ```
    Dataset: Alzheimer 1 and 5
    
    NORM-2
    tSNE with exact algorithm is used
    Distance: hamming

    Pixels: 224 x 224

    Dataset: Alzheimer 1 and 5

    NORM-2
    tsne with exact algorithm is used
    Distance: cosine
    
    Pixels: 224 x 224
    Integrated conversion finished and saved as Out1.mat or Out2.mat!
    Training model begins: Net1
    ...
    |Iter | Eval result | Objective | ...
    |1    |  Best       |  0.18345  | ...
    ....
    Optimization completed
    MaxObjectiveEvaluations of 1 reached.
    Total function evaluations: 1
    Total elapsed time: 1785.2313 seconds
    Total objective function evaluation time: 1784.7876

    Best observed feasible point:
    InitialLearnRate       Momentum     L2Regularization
       4.9866e-05           0.80103       0.012516
    Training model ends
    
    weighted integrated accuracy: 84.73
    weighted integrated AUC: 0.8702
      
    model =
      struct with fields:
      bestIdx: 1
      fileName: "0.18343.mat"
          prob: [1x1 struct]
      valError: 0.1834

    Model Files Saved ...
    Figures Saved in the FIGS folder...
    End of script Example1.
    ```

    *Note that the above values might differ.*

    The following training plot (Figure 2) can be seen if the `Parm.trainingPlot` option is set to `training-progress`.

    <img src="https://github.com/alok-ai-lab/Supplementary/blob/main/TrainingPlot.png?raw=true" width="900" height="450">   
    
    **Figure 2** Training progress plot 

    The objective function figure will be shown for the Bayesian Optimization Technique (BOT). By default 'no BOT' will be applied; i.e. `Parm.MaxObj=1`. However, if BOT is required then change parameter `Parm.MaxObj' to a value higher than 1. If it is set as 'Parm.MaxObj=25' then 25 objective functions will be searched for tuning hyperparameters and the best one (with the minimum validation error) will be selected.
    
    Results file: check `DeepInsight3D_Results.txt` for more information, such as
    ```
    AUC: 0.8692
    ConfusionMatrix
    99  3
    18  11
    ```
 
### Note:

* All the results will be stored in the current stage folder
 `~/DeepInsight3D_pkg/Models/Run4/StageX`  where X is the current stage;

* Similarly, all the figures will be stored in a folder
`~/DeepInsight3D_pkg/FIGS/Run4/StageX` where X is the current stage.

* For feature selection: If the loop continues then the value of X will increment to 1, 2, 3, …; i.e., repeating the model to find a smaller subset of features/genes.

### Bayesian Optimization vs NO Bayesian Optimization
For hyperparameter tuning, Bayesian Optimization Technique (BOT) can be used. If `Parm.MaxObj=1` then NO BOT will be applied. If it is N>1 (i.e. greater than 1) then N objectives functions will be created and the best hyperparameters (for which the validation error is the minimum) will be selected.

Therefore, for BOT, use,

`Parm.MaxObj=N`

where N is any number greater than 1, e.g. N=10 gives 10 objective functions.

For, NO BOT, use,

`Parm.MaxObj=1`

## Description of files and folders

1. `MRep-DeepInsight` has 4 folders: Data, DeepResults, FIGS, and Models. It has several .m files. However, the main file is `Deepinsight3D.m`, which performs tabular data to image conversion and CNN modelling. The codes of MRep-DeepInsight is developed on the DeepInsight3D package and therefore it can perform all tasks of previously developed models such as DeepInsight, DeepFeature and DeepInsight3D. All the parameter settings can be done in the `Parameters.m` file.

2. DeepInsight3D.m has following functions:

    * `func_integrated`: This function supports transforming tabular data to image data using MRep-DeepInsight methodology. It loads the data, splits the training data into the Train and Validation sets, normalizes all the 3 sets (including the Test set), and converts samples to images form using the Training set. The Test and Validation sets are not used to find pixel locations. The image datasets are stored as Out1.mat or Out2.mat depending on whether norm1 or norm2 was selected.

    * `Integrated_Test`: This function computes the integrated performance (as shown in Figure 1c: model analysis phase). 

    * `func_Prepare_Data`: This function supports previous models (DeepFeature, DeepInsight and DeepInsight3D). It loads the data, splits the training data into the Train and Validation sets, normalizes all the 3 sets (including the Test set), and converts multi-layered non-image samples to 3D image form using the Training set. The Test and Validation sets are not used to find pixel locations. Once the pixel locations are obtained, all the non-image samples are converted to 3D image samples. The image datasets are stored as Out1.mat or Out2.mat depending on whether norm1 or norm2 was selected.

    * `func_TrainModel`: This function executes the convolution neural network (CNN) using many pretrained and custom nets. The user may change the net as required. The default values of hyperparameters for CNN are used. However, if `Parm.MaxObj` is greater than 1 then it optimizes hyper-parameters using the Bayesian Optimization Technique. It uses a Training set and Validation set to tune and evaluate the model hyper-parameters.

        Note: To tune hyperparameters of CNN automatically, use a higher value of `Parm.MaxObj`.

        The best model (in case Parm.MaxObj>1) is stored in the DeepResults folder as .mat files, where the file name depicts the best validation error achieved. For example, file 0.32624.mat in the DeepResults folder tells the hyper-parameters at validation error 0.32624. Also, the model file `model.mat` details the weights file and other relevant information to be stored.

4. Feature selection functions
    * `func_FeatureSelection`: This will find activation maps at the ReLu layer, perform Region Accumulation (RA) step and Element Decoder step to find the element/gene subset. The input is model.mat (from `func_TrainModel`) and related .mat file from the folder DeepResults. This function finds CAM for each sample and provides the union of all maps.
    * `func_FS_class_basedCAM`: This function performs class-based CAM, i.e., each class will have a distinct CAM.
    * `func_FeatureSelection_avgCAM`: This function finds the common CAM across all the samples.

5. Non-image to image conversion: two core sub-functions of `func_Prepare_Data` and `func_integrated` are used to convert samples from non-image to image. These are described below.

    * `Cart2Pixel`: The input to this function is the entire Training set. The output is the feature or gene locations Z in the pixel frame. The size of the pixel frame is pre-defined by the user.

    * `ConvPixel`: The input is a non-image sample or feature vector and Z (from above). The output is an image sample corresponding to the input sample.

4. Compression Snow-fall algorithm (SnowFall.m): Not used in this package. However, this compression algorithm is used to provide more space for features in the given pixel frame. Since the conversion from Cartesian coordinates system to the pixel frame depends on the pixel resolution, it becomes difficult to fit all the features without overlapping each other. This algorithm tries to create more space such that the overlapping of feature or gene locations can be minimized. The input is the locations of genes or features with the pixel size information. The output is the readjusted image. It is up to the user to use Snow-fall compression or not by setting `Parm.SnowFall` to either `0` (not use) or `1` (use).

5. Extraction of Gene Names (optional): This option is useful for enrichment analysis. Two files for the extraction of genes are GeneNames_Extract.m and GeneNames.m. The list of names of genes is stored in `~/DeepInsight3D_pkg/Models/RunY/StageX/` folder.

    After running the feature selection function, the results will be stored in the corresponding RunY and StageX folders (where X and Y are integers 1,2,3…). If it is required to find the gene IDs/names of the obtained subset for each cancer type, then execute `GeneNames_Extract` function. Go to Line 4, and set the `Out_Stages` variable. For e.g., if Stage 2 has been saved inside Run1 after executing `func_FS_class_basedCAM`, use `Out_Stages = 2`. Then go to Line 6 and define `FileRun`. For MRep-DeepInsight, we have not used feature selection.

    The gene list per class will be generated. If there are 10 cancer types, then 10 files will be generated. In addition, one file with all genes listed will be generated (e.g. GeneList_UnCmprss.txt). The results will be stored in `~/Models/RunY/StageX` as RunYStageX.tar.gz and a folder with the same results will also be created as RunYStageX. In this example, it will be stored in the folder `Run1Stage2` and Run1Stage2.tar.gz.


## Parameter settings to run the package

A number of parameters/variables are used to control the DeepFeature_pkg. The details are given hereunder

1. `Parm.Method` (select dimensionality reduction technique)

    Dimensionality reduction technique (DRT) can be considered as one of the following methods; 1) tSNE 2) Principal component analysis (PCA) 3) kernel PCA, 4) uniform manifold approximation and projection (umap). For umap you can use python or R scripts (please see umapa_Rmatlab.m). Please note that these DRTs are not used in the conventional manner. Only the element locations are obtained by DRTs, and the reduction of features or dimensions is NOT performed.

    Select this variable in Parameter.m file or after calling `Parm = Parameter(DSETnum)` change

    Parm.Method = ‘tSNE’, ‘kpca’, ‘pca’ or ‘umap’

    Default is tSNE.

2. `Parm.UseIntegrate`: can  be `'yes'` or `'no'`. If 'yes' then the following `Parm.integrate` variable will be used, otherwise `Parm.Method` will be used.

3. `Parm.integrate`: This will support for than one representation (which is not possible with `Parm.Method`). Various manifold techniques (with respective distances esp. for tSNE) and supplement methods can be listed here to integrate the performance of these techniques. See Line 133 in `Parameters.m` file. The usage is:

   ```
   Parm.integrate = {Manifold1,Distance1,Manifold1,Distance2,...
            Manifold3,Manifold4,...,
            Supplement1,Supplement2,Supplement3}
   ```

   where Manifold1 is tSNE and Distance1...Distance11 are tSNE distances. Manifold3, Manifold4,... are other manifold techniques such as KPCA, UMAP and PCA. Supplement1, Supplement2,.. are Blur, Gabor and Assignment. All or any of these combinations can be used for `Parm.integrate` as long as more than 1 technique is selected to render multiple representation strategy.

4. `Parm.Dist` (Distance selection only for tSNE) - This parameter is NOT used when `Parm.UseInteregrate=yes`.

    If tSNE is used, then one of the following distances can be used. The default distance is ‘euclidean’.

    Parm.Dist = ‘cosine’, ‘hamming’, ‘mahalanobis’, ‘educidean’, ‘chebychev’, ‘correlation’, ‘minkowski’, ‘jaccard’, or ‘seuclidean’ (standardized Eucliden distance).

5. `Parm.Max_Px_Size` (maximum pixel frame either row or column)

    The default value is 224 as required by ResNet-50 architecture.

6. `Parm.ValidRatio` (ratio of validation data and training data)

    The amount of training data required to be used as a validation set. Default is 0.1; i.e., 10% of training data is kept aside as a validation set. The new training set will be 90% of the original size.
   **Note**: If `Parm.ValidRatio=0` then not validation set will be kept aside. In this case, entire training set will be used for model estimaton.

7. `Parm.Seed`

    Random parameter seed to split the data.

8.  `Parm.NetName`: use pre-trained nets such as `resnet50`, `inceptionresnetv2`, `nasnetlarge`, `efficientnetb0`, `googlenet` and so on. See a list of pre-trained nets from Matlab link [here](https://www.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html)

9.  `Parm.ExecutionEnvironment`: execution environment based on your hardware. Options are `cpu`, `gpu`, `multi-gpu`, `parallel`, and `auto`. Please check trainingOptions (Matlab) for further details.

10.  `Parm.ParallelNet`: if '1' then this option overrides `Parm.NetName`. The custom made net from `makeObjFcn2.m` will be used.

11.  `Parm.miniBatchSize`: define miniBatchSize, default is 1024 (for 4 parallel A100 GPUs of 40GB each).

12.  `Parm.Augment`: augment samples during training progress, select '1' for yes and '0' for no.

13.  `Parm.AugMeth`: select method '1' or '2'. Method 1 automatically augments samples whereas Method 2 is done by the user

14.  `Parm.aug_tr`: if `Parm.AugMeth=2` then `Parm.aug_tr=500` will augment 500 samples of training set if the number of samples in a class is less than 500.

15.  `Parm.aug_val`: if `Parm.Aug=2` then `Parm.aug_val=50` will augment 50 samples of validation set if the number of samples in a class is less than 50.

16.  `Parm.ApplyFS`: if '1' it applies a feature selection process using Logistic Regression before applying DeepInsight transformation.

17.  `Parm.FeatureMap`: has following options. `0` means use 'all' omics or multi-layered data for conversion.
                            '1' means use the 1st layer for conversion (e.g. expression)
                            '2' means use the 2nd layer for conversion (e.g. methylation)
                            '3' means use the 3rd layer for conversion (e.g. mutation)
                            
18.  `Parm.TransLearn`: if '1' then learn CNN from previously trained nets on your different datasets. Please save `model.mat` and pretrained model `0*.mat` files generated from the previous run to `Models/Run32/Stage1` folder. The current execution of CNN will train on the pretrained model `Models/Run32/Stage1/0*.mat`. This will render transfer learning from `0*.mat` and `model.mat` files.

19. `Parm.FileRun`

    Change the name as RunX, where X is an integer defining the run of DeepFeature on your data.

    Change the value X for new runs.

20. `Parm.SnowFall` (compression algorithm)

    Suppose SnowFall compression algorithm is used then set the value as 1, otherwise 0. Default is set as 1.

21. `Parm.Threshold` (for Class Activation Maps)

    Set the threshold of class activation maps (CAMs) by changing the value between 0 and 1. If the value is high (towards 1), then the region of activation maps will be very fine. On the other hand, the region will be broader towards value 0. Default is 0.3. 

22. `Parm.DesiredGenes`

    Expected number of genes to be selected. Default is set as 1200. However, change as required.

23. `Parm.UsePrevModel`

    The iterative way runs in multiple stages. If you want to avoid running CNN multiple times then set these values as ‘y’ (yes); i.e., the previous weights of CNN will be used for the current model. This way, the processing time is shorter, however, performance (in terms of selection and accuracy) would be lower. The default setting is ‘n’ (no).

24. `Parm.SaveModels`

    For saving models type ‘y’, otherwise ‘n’. Default is set as yes ‘y’.

25. `Parm.Stage`

    Define the stage of execution. The default value is set as `Parm.Stage=1`. All the results will be saved in RunXStage1. If iterative process is executed then results will be stored in Stage2, Stage3… and so on.


26. `Parm.PATH`

    Default paths for FIGS, Models and Data are `~/MRep-DeepInsight/FIGS/`, `~/MRep-DeepInsight/Models/` and `~/MRep-DeepInsight/Data/`, respectively. Runtime parameters will be stored in `~/MRep-DeepInsight/` folder (such as model.mat, Out1.mat or Out2.mat).

27. Log and performance file (including an overview of parameter information)

    The runtime results will be stored in `~/MRep-DeepInsight/DeepInsight3D_Results.txt` with complete information about the run.

## Related materials

### DeepInsight YouTube

A YouTube video about the original DeepInsight method is available [here](https://www.youtube.com/watch?v=411iwaptk24&feature=youtu.be).
A Matlab page on DeepInsight can be viewed from [here](https://www.mathworks.com/company/user_stories/case-studies/riken-develops-a-method-to-apply-cnn-to-non-image-data.html).

### Previous papers
*Sharma A, Vans E, Shigemizu D, Boroevich KA, Tsunoda T, DeepInsight: A methodology to transform a non-image data to an image for convolution neural network architecture, Scientific Reports, 9(1), 1-7, 2019.*  
*Sharma A, Lysenko A, Boroevich K, Vans E, Tsunoda T, DeepFeature: feature selection in nonimage data using convolutional neural network, Briefings in Bioinformatics, 22(6), 2021.*  
*Sharma A, Lysenko A, Boroevich K, Tsunoda T, DeepInsight-3D architecture for anti-cancer drug response prediction with deep-learning on multi-omics, Scientific Reports, 13(2483), 2023.*  
*Jia S, Lysenko A, Boroevich K, Sharma A, Tsunoda T, scDeepInsight: a supervised cell-type identification method for scRNA-seq data with deep learning, Briefings in Bioinformatics, 2023. https://doi.org/10.1093/bib/bbad266*

### GitHub weblink of DeepInsight (Python and Matlab)
Overall weblink [here](https://alok-ai-lab.github.io/DeepInsight/)

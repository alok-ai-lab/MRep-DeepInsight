% Example 1: Classification of tabular data using the MRep-DeepInsight model
%
% In this Example, an Alzheimer's disease data (saved in Data folder as 
% dataset4.mat) is first converted to images using MRep-DeepInsight converter.
% Then CNN net (resnet-50) is applied for training the model. The 
% performance evaluation (accuracy) is done on the test set of the
% data. 
%
% Example data is "Alzheimer's disease" data (dataset4.mat)

clear all;
close all hidden;

% 1. Set up parameters by changing Parameter.m file, otherwise leave it with default values.
% 2. Provide the path of dataset in Parameter.m file by chaning the "Data_path" variable.

DSETnum = 4; %This means the stored data in your defined path is dataset1.mat
	     % dataset(DSETnum)
Parm = Parameters(DSETnum); % Define parameters for MRep-DeepInsight and CNN


% NOTE: 1) Set "Parm.miniBatchSize" based on your GPU requirements. 
%       by default Parm.miniBatchSize = 1024.
%   
%       2) Set execution environment (for trainingOptions). By default it
%       is set to 'multi-gpu'.

[AUC,C,Accuracy,ValErr] = DeepInsight3D(DSETnum,Parm); % This will perform MRep-DeepInsight
% NOTE: 1) You may use separately image conversion function using the file
% func_Prepare_Data.m
%
%       2) func_TrainModel performs CNN modeling. Some pretrained nets are
%       given and 1 custom made are given. However, please prepare your own
%       nets as per required.

% Define the folder where the model files and figures to be stored.
% By default Parm.FileRun = 'Run1' and Parm.Stage=1 (change as required)
% Then execute the following commands.


% Save model files
func_SaveModels(Parm); % model files will be stored in e.g. ~/DeepInsight3D/Models/Run1/Stage1/ (if DSETnum=1)

% Save all figures
func_SaveFigs(Parm); % all figures will be stored in e.g. ~/DeepInsight3D_pkg/FIGS/Run1/Stage1/ (if DSETnum=1)

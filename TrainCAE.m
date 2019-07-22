%load in all data as inputs
imds = imageDatastore('Data\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');


%based off of trainNetwork help example for img class
options = trainingOptions('adam', ...
    'MaxEpochs',50,...
    'InitialLearnRate',1e-3, ...
    'Verbose',false, ...
    'Plots','training-progress');






%trainingDS = imds;
%trainingDS.Labels = categorical(trainingDS.Labels);
%trainingDS.ReadFcn = @readFcn1;

trainNetwork(imds, layers, options)
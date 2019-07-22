layers = [
    imageInputLayer([inputSize 1], 'Normalization','none')
    fullyConnectedLayer(inputSize)
    %reluLayer
    fullyConnectedLayer(16*L^2)
    reluLayer
    convert1d2dLayer
    convolution2dLayer(3, inputSize, 'Padding', 1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 2*inputSize, 'Padding', 1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 1, 'Padding', 1)
    sigmoidLayer('sigmoid') %custom sigmoid layer
    % ---end of encoder--- %
    transposedConv2dLayer(16, inputSize, 'Stride', 16) % upsampling
    % ---start of decoder--- %
    convolution2dLayer(5, 2*inputSize, 'Padding', 2)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, inputSize, 'Padding', 1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(inputSize)
    %reluLayer
    fullyConnectedLayer(inputSize)
    softmaxLayer
    classificationLayer];

%analyzeNetwork(layers)
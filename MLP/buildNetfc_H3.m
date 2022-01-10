function net = buildNetfc_H3(options)
%=========================================================================%
% Build the neural network.
% Multi Perception Network with 3 hidden layers of 1024 neurons in each
% layer and result actioanion function between each layer
%=========================================================================%

switch options.type
    case 'MLP1'
        input = imageInputLayer(options.inputSize, 'Name', 'input');
        fc1 = fullyConnectedLayer(512*2, 'Name', 'fc1');
        relu1 = reluLayer('Name','relu1');
       % drop1 = dropoutLayer(0.2, 'Name', 'drop1');
        fc2 = fullyConnectedLayer(512*2, 'Name', 'fc2');
        relu2 = reluLayer('Name','relu2');
       % drop2 = dropoutLayer(0.2, 'Name', 'drop2');
        fc3 = fullyConnectedLayer(512*2, 'Name', 'fc3');
        relu3 = reluLayer('Name','relu3');
        fc9 = fullyConnectedLayer(2, 'Name', 'fc9');
        Olayer = regressionLayer('Name','Olayer');
        sfm = softmaxLayer('Name','sfm');
        classifier = classificationLayer('Name','classifier');

        layers = [
                  input
                  fc1
                  relu1
                  %drop1
                  fc2
                  relu2
                  fc9
                  Olayer

                 ];
        net = layerGraph(layers);

end

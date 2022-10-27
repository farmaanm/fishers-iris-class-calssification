clear clc

% Loading fisheriris data
load fisheriris.mat

% Convert species to an index vector where 1, 2, 3 correspond to setosa, versicolor, virginica
species = grp2idx(species);

% Shuffling a Vector, seperated in to unique values
dataset_1 = randperm(150,(150*0.6))'; % Seperating 60% of the unique values
dataset_2 = setdiff((1:150)',dataset_1); % Seperating the reamining 40%

% Getting 60% for the training dataset
trainData = meas(dataset_1,:);
trainTarget = species(dataset_1,:);

% Getting 40% for the training dataset
testData = meas(dataset_2,:);
testTarget = species(dataset_2,:);

% Construct a feedforward network with 5, 10, 15, 20 hidden layers
net = feedforwardnet([5,10,15,20]);

% Training the data
net = train(net,trainData.',trainTarget.');
view(net)



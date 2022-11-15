clear; close all; clc;

% Task 2.1
% 1 - Loading Fisher Iris dataset
load fisheriris.mat

% 2 - Shuffling and separating the dataset
% Convert species to an index vector where 1, 2, 3 correspond to setosa, versicolor, virginica
species = grp2idx(species);

% Shuffling a Vector, separated in to unique values
dataset_1 = randperm(150, (150*0.6))'; % Separating 60% of the unique values
dataset_2 = setdiff((1:150)', dataset_1); % Separating the remaining 40%

% Getting 60% for the training dataset
trainData = meas(dataset_1,:);
trainTarget = species(dataset_1,:);

% Getting 40% for the training dataset
testData = meas(dataset_2,:);
testTarget = species(dataset_2,:);

% Task 2.2
% 1, 2, 3 - Construct a feedforward network with 5, 10, 15, 20 hidden layers
for n = [5, 10, 15, 20]

    net = feedforwardnet(n);
    accuracy_array = [];

    for i = 1:10

        % Training the data
        net = train(net, trainData.', trainTarget.');

        % 4 - Testing the data
        predicted_output = net(testData.');
        
        accuracy_check = (sum(round(predicted_output) == testTarget(:,end).')) * (100/size(testTarget,1));

        accuracy_array(end+1) = accuracy_check;

    end

    average_accuracy = mean(accuracy_array);

end

view(net)

% % 4 - Testing the data
% predicted_output = net(testData.');
% perf = perform(net, predicted_output, testTarget.');
% 
% accuracy_check = (sum(round(predicted_output) == testTarget(:,end).')) * (100/size(testTarget,1));




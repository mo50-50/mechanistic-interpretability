'''
This file is for creating the machine learning part of the simulation.
'''
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim

# create dataset class to that will take as input the data points list which is a list of 5 np arrays
# each array is a list of points belonging to the class of its idx, and num classes, which is teh number
# of classes (less than or equal to the len of the data points list)
# there is also another array which is ood, this array contains points for an additional class
# there is a variable ood, which if false, this new array should not be ussed at all. If true, then
# the ood array should be used as the last class.
class Data(Dataset):
    def __init__(self, data_points, num_classes, ood_array, ood):
        self.data_points = data_points
        self.num_classes = num_classes
        self.ood_array = ood_array
        self.ood = ood

    def __len__(self):
        total = 0
        for i in range(self.num_classes):
            total += len(self.data_points[i])
        if self.ood:
            # we should not oversample the ood class so we will add only the part of it thats equal to all others combined
            if len(self.ood_array) > total:
                total *= 2
            else:
                total += len(self.ood_array)
        return total

    def __getitem__(self, idx):
        # find the class of the idx
        class_idx = 0
        while class_idx < self.num_classes and idx >= len(self.data_points[class_idx]):
            idx -= len(self.data_points[class_idx])
            class_idx += 1

        if class_idx >= self.num_classes: # means out point is in ood class
            # return self.ood_array[idx], class_idx
            return torch.tensor(self.ood_array[idx], dtype=torch.float32), class_idx
        # return convert to double
        # return self.data_points[class_idx][idx], class_idx
        return torch.tensor(self.data_points[class_idx][idx], dtype=torch.float32), class_idx

# create a neural network class that will take as input the number of classes and the number of features,
# and the number of hidden layers, and the number of neurons in each hidden layer,
# and the activation function, which is a string that can be either "relu", "sigmoid", "tanh"
# there is one more boolean input, ood, which id true if there is one more output class for out-distributon

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes, num_features, num_hidden_layers, num_neurons, activation_function, ood):
        super(NeuralNetwork, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = num_neurons
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_features, num_neurons))
        for i in range(num_hidden_layers):
            self.layers.append(nn.Linear(num_neurons, num_neurons))
        if ood:
            out_size = num_classes + 1
        else:
            out_size = num_classes
        self.layers.append(nn.Linear(num_neurons, out_size))
        self.activation_function = getattr(nn.functional, activation_function)

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.activation_function(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# create a training class that will take as input the neural network, the dataset, the batch size
# the learning rate, and the loss function, which is a string that can be either "cross_entropy" or "mse"
# and the optimizer, which is a string that can be either "adam" or "sgd" or "rmsprop" or "adagrad" or "adadelta"

class Training:
    def __init__(self, model, dataset, batch_size, learning_rate, loss_function, optimizer):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # create loss function based on the string input
        if loss_function == "CrossEntropy":
            self.loss_function = nn.CrossEntropyLoss()
        elif loss_function == "MSELoss":
            self.loss_function = nn.MSELoss()
        # create optimizer based on the string input
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif optimizer == "Adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif optimizer == "Adadelta":
            self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate)

        # create a data loader for the dataset
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    # define a function that will train for only one epoch
    def train_one_epoch(self):
        # set the model to training mode
        self.model.train()

        loss = 0 # initial assignment to avoid error
        # iterate over the data loader
        for batch_idx, (data, target) in enumerate(self.data_loader):
            # zero the gradients
            self.optimizer.zero_grad()
            # forward pass
            output = self.model(data)
            # calculate the loss
            loss = self.loss_function(output, target)
            # backward pass
            loss.backward()
            # update the weights
            self.optimizer.step()

        # print the loss at the end
        print(f"Loss: {loss.item()}")

    # define a funtion to predict. This will take a number that represents the number of points in each
    # direction, it should create a 2D grid of points between 0, 1 and then predict the values for each point.
    def predict(self, n_points):
        # create a grid of points
        x = torch.linspace(0, 1, n_points)
        y = torch.linspace(0, 1, n_points)
        x_grid, y_grid = torch.meshgrid(x, y)
        # create a tensor of points
        points = torch.stack((x_grid, y_grid), dim=2)
        # reshape the tensor to be 2D
        points = points.reshape(-1, 2)
        # predict the values
        with torch.no_grad():
            predictions = self.model(points)
        # reshape the predictions to be 2D
        predictions = predictions.reshape(n_points, n_points, -1)
        # return the predictions
        return predictions

    # define a function called gradient ascent, that will pick random points as initial points
    # for all classes then it will run gradient ascent algorithm adam, to obtain points that will
    # maximize the probability of each class. It will return the points and history (path) of all points.
    def gradient_ascent(self, iterations, lr, num_classes):
        # create a tensor of random points
        points = 5*torch.rand(num_classes, 2) - 2.5
        # make the points require gradients
        points.requires_grad = True
        # points = torch.empty(num_classes, 2, requires_grad=True).uniform_(-2, 2)

        # pass points through the sigmoid to ensure it is in the range [0, 1]
        points_sig = torch.sigmoid(points)
        # create a tensor of history
        history = torch.zeros(iterations, num_classes, 2)
        # save the initial points
        history[0] = points_sig.detach().clone()
        # create optimizer
        optimizer = torch.optim.Adam([points], lr=lr)
        # run gradient ascent
        for i in range(iterations - 1):
            # zero the gradients
            optimizer.zero_grad()
            # predict the values
            predictions = self.model(torch.sigmoid(points))
            # print(predictions.shape, 'predictions.shape')

            # calculate the loss each point should belong to one class,
            # there might be one additional output class for the ood class
            n_rows, n_cols = predictions.shape
            # take only a square portion if there are extra columns
            n = min(n_rows, n_cols)
            trace = torch.sum(torch.diagonal(predictions[:, :n]))
            # for gradient ascent, negate it
            loss = -trace

            # calculate the gradients
            loss.backward()
            # update the points
            optimizer.step()
            # save the points after applying the sigmoid
            history[i + 1] = torch.sigmoid(points).detach().clone()
            # history[i] = points.detach().clone()
        # return the history as numpy array
        # print(history.shape)
        return history.detach().numpy()


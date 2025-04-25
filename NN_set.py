'''

This file contains the implementation of several neural network models with different numbers of hidden layers.


    Initialize the weights of the neural network model.

    Args:
        m: The module to initialize the weights for.

    Neural Network Model with k hidden layers.

    Args:
        input_size: The size of the input layer.

        Forward pass of the neural network model.

        Args:
            t_input: The input tensor.

        Returns:
            The output tensor.

'''

import torch as pt


# initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        pt.nn.init.xavier_normal(m.weight.data)
        pt.nn.init.constant(m.bias.data, 0.0)


# Neural Network Model (1 hidden layer)
class NN1(pt.nn.Module):
    def __init__(self, input_size):
        super(NN1, self).__init__()
        self.fc1 = pt.nn.Linear(input_size, 32)
        self.bn1 = pt.nn.BatchNorm1d(num_features=32)
        self.output = pt.nn.Linear(32, 1)

    def forward(self, t_input):
        t_output = self.bn1(self.fc1(t_input))
        t_output = pt.nn.functional.relu(t_output)
        return self.output(t_output)


# Neural Network Model (2 hidden layers)
class NN2(pt.nn.Module):
    def __init__(self, input_size):
        super(NN2, self).__init__()
        self.fc1 = pt.nn.Linear(input_size, 32)
        self.bn1 = pt.nn.BatchNorm1d(num_features=32)
        self.fc2 = pt.nn.Linear(32, 16)
        self.bn2 = pt.nn.BatchNorm1d(num_features=16)
        self.output = pt.nn.Linear(16, 1)

    def forward(self, t_input):
        t_output = self.bn1(self.fc1(t_input))
        t_output = pt.nn.functional.relu(t_output)
        t_output = self.bn2(self.fc2(t_output))
        t_output = pt.nn.functional.relu(t_output)
        return self.output(t_output)


# Neural Network Model (3 hidden layers)
class NN3(pt.nn.Module):
    def __init__(self, input_size):
        super(NN3, self).__init__()
        self.fc1 = pt.nn.Linear(input_size, 32)
        self.bn1 = pt.nn.BatchNorm1d(num_features=32)
        self.fc2 = pt.nn.Linear(32, 16)
        self.bn2 = pt.nn.BatchNorm1d(num_features=16)
        self.fc3 = pt.nn.Linear(16, 8)
        self.bn3 = pt.nn.BatchNorm1d(num_features=8)
        self.output = pt.nn.Linear(8, 1)

    def forward(self, t_input):
        t_output = self.bn1(self.fc1(t_input))
        t_output = pt.nn.functional.relu(t_output)
        t_output = self.bn2(self.fc2(t_output))
        t_output = pt.nn.functional.relu(t_output)
        t_output = self.bn3(self.fc3(t_output))
        t_output = pt.nn.functional.relu(t_output)
        return self.output(t_output)


# Neural Network Model (4 hidden layers)
class NN4(pt.nn.Module):
    def __init__(self, input_size):
        super(NN4, self).__init__()
        self.fc1 = pt.nn.Linear(input_size, 32)
        self.bn1 = pt.nn.BatchNorm1d(num_features=32)
        self.fc2 = pt.nn.Linear(32, 16)
        self.bn2 = pt.nn.BatchNorm1d(num_features=16)
        self.fc3 = pt.nn.Linear(16, 8)
        self.bn3 = pt.nn.BatchNorm1d(num_features=8)
        self.fc4 = pt.nn.Linear(8, 4)
        self.bn4 = pt.nn.BatchNorm1d(num_features=4)
        self.output = pt.nn.Linear(4, 1)

    def forward(self, t_input):
        t_output = self.bn1(self.fc1(t_input))
        t_output = pt.nn.functional.relu(t_output)
        t_output = self.bn2(self.fc2(t_output))
        t_output = pt.nn.functional.relu(t_output)
        t_output = self.bn3(self.fc3(t_output))
        t_output = pt.nn.functional.relu(t_output)
        t_output = self.bn4(self.fc4(t_output))
        t_output = pt.nn.functional.relu(t_output)
        return self.output(t_output)


# Neural Network Model (5 hidden layers)
class NN5(pt.nn.Module):
    def __init__(self, input_size):
        super(NN5, self).__init__()
        self.fc1 = pt.nn.Linear(input_size, 32)
        self.bn1 = pt.nn.BatchNorm1d(num_features=32)
        self.fc2 = pt.nn.Linear(32, 16)
        self.bn2 = pt.nn.BatchNorm1d(num_features=16)
        self.fc3 = pt.nn.Linear(16, 8)
        self.bn3 = pt.nn.BatchNorm1d(num_features=8)
        self.fc4 = pt.nn.Linear(8, 4)
        self.bn4 = pt.nn.BatchNorm1d(num_features=4)
        self.fc5 = pt.nn.Linear(4, 2)
        self.bn5 = pt.nn.BatchNorm1d(num_features=2)
        self.output = pt.nn.Linear(2, 1)

    def forward(self, t_input):
        t_output = self.bn1(self.fc1(t_input))
        t_output = pt.nn.functional.relu(t_output)
        t_output = self.bn2(self.fc2(t_output))
        t_output = pt.nn.functional.relu(t_output)
        t_output = self.bn3(self.fc3(t_output))
        t_output = pt.nn.functional.relu(t_output)
        t_output = self.bn4(self.fc4(t_output))
        t_output = pt.nn.functional.relu(t_output)
        t_output = self.bn5(self.fc5(t_output))
        t_output = pt.nn.functional.relu(t_output)
        return self.output(t_output)

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()

        # Part 1

        # Input Layer
        self.fc1 = nn.Linear(784, 100)
        # Output Layer
        self.fc2 = nn.Linear(100, 10)

        # Part 2

        # 40 Kernels 5x5 kernel with a Stride = 1
        self.conv1 = nn.Conv2d(1, 40, 5, stride = 1)
        self.conv2 = nn.Conv2d(40, 40, 5, stride = 1)
        # Input layer
        self.fc3 = nn.Linear(640, 100)
        # Output Layer
        self.fc4 = nn.Linear(100, 10)

        # Part 4

        self.fc5 = nn.Linear(100, 100)

        # FC Layers with 1000 neurons

        self.fc6 = nn.Linear(640, 1000)
        self.fc7 = nn.Linear(1000, 1000)
        self.fc8 = nn.Linear(1000, 10)

        # Dropout Layer
        self.dropout = nn.Dropout(0.5)

        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else:
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)

    # Part 1
    def model_1(self, X):
        # ===========================
        # One fully connected layer.
        # ===========================

        # Flattens the input without editing the batch size
        x = torch.flatten(X, 1)
        # Input layer
        x = torch.sigmoid(self.fc1(x))
        # Output Layer
        x = torch.sigmoid(self.fc2(x))

        return x

    # Part 2
    def model_2(self, X):
        # ======================================================
        # Two convolutional layers + one fully connnected layer.
        # ======================================================

        # Two convolution layers with max pooling
        x = F.max_pool2d(self.conv1(X), (2,2))
        x = F.max_pool2d(self.conv2(x), (2,2))

        # Flatten Input
        x = torch.flatten(x, 1)

        # Input Layer
        x = torch.sigmoid(self.fc3(x))
        # Hidden Layer with Sigmoid
        x = torch.sigmoid(self.fc4(x))
        return x

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        x = F.max_pool2d(self.conv1(X), (2,2))
        x = F.max_pool2d(self.conv2(x), (2,2))

        x = torch.flatten(x, 1)

        # Input Layer
        x = torch.relu(self.fc3(x))
        # Hidden Layer with Sigmoid
        x = torch.relu(self.fc4(x))

        return x

    # Add one extra fully connected layer.
    def model_4(self, X):

        x = F.max_pool2d(self.conv1(X), (2,2))
        x = F.max_pool2d(self.conv2(x), (2,2))

        x = torch.flatten(x, 1)

        # Generates the FC Layers
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc5(x))

        return x

    # Use Dropout now.
    def model_5(self, X):
        x = F.max_pool2d(self.conv1(X), (2,2))
        x = F.max_pool2d(self.conv2(x), (2,2))

        x = torch.flatten(x, 1)

        # FC and Dropout Layers
        x = torch.relu(self.fc6(x))
        x = self.dropout(x)
        x = torch.relu(self.fc7(x))
        x = self.dropout(x)
        x = torch.relu(self.fc8(x))

        return x

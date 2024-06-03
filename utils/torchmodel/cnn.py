from torch import nn
import torch.nn.functional as F


class CNN1(nn.Module):
    def __init__(self, in_channels, output_dims):
        super(CNN1, self).__init__()
        # conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        conv2 = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.convs = [conv1, conv2]
        for layer in self.convs:
            self.add_module(str(len(self._modules)), layer)
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 50),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(50, output_dims)

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNN2(nn.Module):
    def __init__(self, in_channels, output_dims):
        super(CNN2, self).__init__()
        conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        conv2 = nn.Sequential(
            nn.Conv2d(10, 64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.convs = [conv1, conv2]
        for layer in self.convs:
            self.add_module(str(len(self._modules)), layer)
        self.fc1 = nn.Sequential(
            nn.Linear(1600, 50),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(50, output_dims)

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNN3(nn.Module):
    def __init__(self, in_channels, output_dims):
        super(CNN3, self).__init__()
        # conv1 = nn.Conv2d(3, 6, 5)
        conv1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        # self.pool = nn.MaxPool2d(2, 2)
        conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU()
        )
        conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU()
        )
        conv3 = nn.Conv2d(16, 32, 5)
        conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # conv4 = nn.Conv2d(32, 64, 5)
        conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU()
        )
        self.convs = [conv1, conv2, conv3, conv4, conv5]
        for layer in self.convs:
            self.add_module(str(len(self._modules)), layer)
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(64, output_dims)

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class CNN4(nn.Module):
    def __init__(self, in_channels, output_dims):
        super(CNN4, self).__init__()
        self.num_classes = output_dims
        self.output_dim = 512
        # Define a series of convolutional layers named conv1, conv2, etc.
        conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05)
        )
        conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.convs = [conv1, conv2, conv3, conv4, conv5, conv6]
        for layer in self.convs:
            self.add_module(str(len(self._modules)), layer)
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(512, output_dims)

    def forward(self, x):
        # Pass the input tensor through the series of convolutional layers
        for layer in self.convs:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        # Apply log softmax activation to the output
        return self.fc3(x)

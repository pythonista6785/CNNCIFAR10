import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # input is color image with 3 channles 
        # 6 is the number of feature maps and the kernel size is 5x5

        self.pool = nn.MaxPool2d(2,2)
        # maxpool will be used multiple times, but defined once 

        # in_channels = 6 because self.conv1 output is 6 channels 
        self.conv2 = nn.Conv2d(6,16,5)
        # 5*5 comes from the dimension of the last convent layer 
        self.fc1 = nn.Linear(16*5*5, 128)

        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sm(self.fc3(x))     # softmax activation on final layer
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PetNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 12, 5) #128 - 5 = 123 + 1 = 124. (12, 124, 124)
        self.pool = nn.MaxPool2d(2, 2) #(12, 62, 62)
        self.conv2 = nn.Conv2d(12, 24, 5) # 62 - 5 = 57 + 1 = 58. (24, 58, 58) -> (24, 29, 29) 
        self.fc1 = nn.Linear(24 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 37)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x , 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
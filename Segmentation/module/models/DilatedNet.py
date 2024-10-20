import torch.nn as nn
import torch.nn.functional as F

class DilatedNet(nn.Module):
    #just replace pools with dilation
    #use dilation 1, 2, 4, 8, 16, 1 (a mixture of the paper and the description given here)

    def __init__(self):
        super().__init__()

        # context block
        self.conv0 = nn.Conv2d(3, 64, 3, padding=1, dilation = 1) #receptive field: 3x3
        self.conv1 = nn.Conv2d(64, 64, 3, padding=2, dilation = 2) #receptive field: 7x7
        self.conv2 = nn.Conv2d(64, 64, 3, padding=4, dilation = 4) #receptive field: 15x15
        self.conv3 = nn.Conv2d(64, 64, 3, padding=8, dilation = 8) #receptive field: 31x31
        self.conv4 = nn.Conv2d(64, 64, 3, padding=16, dilation = 16) #receptive field: 63x63
        self.conv5 = nn.Conv2d(64, 1, 3, padding=1, dilation = 1) #receptive field: 65x65

    def forward(self, x):
        c0 = F.relu(self.conv0(x))
        c1 = F.relu(self.conv1(c0))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(c4))
        
        return c5

import torch 
import torch.nn as nn 
import torch.functional as F
from torch.nn.modules.activation import ELU
from torch.nn.modules.linear import Identity
import torchvision 
import numpy as np 
"""
Things to do change relu ---> ELU 
"""
class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__
        
        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=1, padding =1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,kernel_size=1, stride = 1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels,eps = 0.001, momentum=0.99)
        self.batch_norm2 = nn.BatchNorm2d(out_channels, eps = 0.001, momentum= 0.99)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1))
        x = self.relu(self.batch_norm2(self.conv2))
        x+= identity

        return x


class BottleneckBlock1_3_1(nn.Module):
    def __init__(self, in_channels, out_channels, expansion):
        super(BottleneckBlock1_3_1).__init__
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride =1, padding =1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride =1, padding =1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride =1, padding =1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(out_channels, eps = 0.001, momemtum = 0.99)
        self.batch_norm2 = nn.BatchNorm2d(out_channels, eps = 0.001, momemtum = 0.99)
        self.batch_norm3 = nn.BatchNorm2d(out_channels, eps = 0.001, momemtum = 0.99)



    def forward(self, x):
        
        pass 


class BottleneckBlock1_5_1(nn.Module):
    def __init(self):
        super(BottleneckBlock1_5_1, self).__init__

    def forward(self,x):
        pass 

class AggregationBlock(nn.Module):
    def __init__(self):
        super(AggregationBlock, self).__init__


    def forward(self,x):
        pass

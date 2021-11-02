import torch 
import torch.nn as nn 
import torch.functional as F
from torch.nn.modules.activation import ELU
from torch.nn.modules.linear import Identity
import torchvision 
import numpy as np 

"""
Things to do change relu ---> ELU

I have figured it out that there is expansion factor of 6 in all the next residual layers
[16,24,48,88,120,208,352] * 6 == num of filters in the respective residual layers

"""

#starting residual block
class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=1, padding =1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,kernel_size=1, stride = 1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels,eps = 0.001, momentum=0.99)
        self.batch_norm2 = nn.BatchNorm2d(out_channels, eps = 0.001, momentum= 0.99)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x+= identity

        return x

#repititive residual block with (1x1) - (3x3) - (1x1) kernel
class BottleneckBlock1_3_1(nn.Module):
    def __init__(self, in_channels, out_channels, d_stride, d_pad,expansion= 6):
        super(BottleneckBlock1_3_1).__init__
        self.expansion = expansion
        self.d_stride = d_stride
        self.d_pad = d_pad
        self.conv1 = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride =1, padding =0)
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, stride =1, padding =self.d_pad)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride =1, padding =0)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(out_channels, eps = 0.001, momentum = 0.99)
        self.batch_norm2 = nn.BatchNorm2d(out_channels, eps = 0.001, momentum = 0.99)
        self.batch_norm3 = nn.BatchNorm2d(out_channels, eps = 0.001, momentum = 0.99)

    def forward(self, x):

        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.relu(self.batch_norm3(self.conv3(x)))
        x+=identity
        return x

# reptitive block with (1x1) - (5x5) - (1x1) kernel
class BottleneckBlock1_5_1(nn.Module):
    def __init(self,in_channels, out_channels, d_stride, d_pad,expansion= 6):
        super(BottleneckBlock1_5_1, self).__init__()
        self.d_pad = d_pad 
        self.d_stride = d_stride
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride =1, padding =0)
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, stride =1, padding =self.d_pad)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride =1, padding =0)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(out_channels, eps = 0.001, momentum = 0.99)
        self.batch_norm2 = nn.BatchNorm2d(out_channels, eps = 0.001, momentum = 0.99)
        self.batch_norm3 = nn.BatchNorm2d(out_channels, eps = 0.001, momentum = 0.99)
    def forward(self,x):

        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.relu(self.batch_norm3(self.conv3(x)))
        x+=identity
        return x 
 

class AggregationBlock(nn.Module):
    def __init__(self,in_channels, out_channels, expansion, pad_stridepairs= []):
        super(AggregationBlock, self).__init__()
        self.pad_stridepairs =pad_stridepairs  
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, in_channels*self.expansion, kernel_size=1, stride =self.pad_stridepairs[0][0], padding = self.pad_stridepairs[0][1])
        self.conv2 = nn.Conv2d(in_channels*self.expansion, in_channels*self.expansion, kernel_size=5, stride =self.pad_stridepairs[1][0], padding =self.pad_stridepairs[1][1])
        self.conv3 = nn.Conv2d(in_channels*self.expansion, out_channels, kernel_size=1, stride =self.pad_stridepairs[2][0], padding =self.pad_stridepairs[2][1])
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(out_channels, eps = 0.001, momentum=0.99)
        self.batch_norm2 = nn.BatchNorm2d(out_channels, eps = 0.001, momentum = 0.99)
        self.batch_norm3 = nn.BatchNorm2d(out_channels, eps = 0.001, momentum = 0.99)          

    def forward(self,x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv1(x))

        return x 


# Combined Resnet for conv features extraction

class ConvFeatureExtractor(nn.Module):
    def __init__(self, filter_list, num_in_channels =12):
        super(ConvFeatureExtractor,self).__init__()

        self.block1 = InitialBlock
        self.block2 = AggregationBlock
        self.block3 = BottleneckBlock1_3_1
        self.block4 = BottleneckBlock1_5_1
        self.filter_list = filter_list
        self.num_in_channels = num_in_channels 
        self.initial_output_channels = self.filter_list[0]

        self.layer1 = self.make_layers()
        # self.layer2 = 
        # self.layer3 = 
        # self.layer4 = 
        # self.layer5 = 
        # self.layer6 = 


    def make_layers(self, filter_size, blocks):
        
        layers = []


        return nn.Sequential(*layers)

    def feed_forward_network(self,x):


        return x 

"""

filters_list = [16,24,48,88,120,208,352]
model = AggregationBlock(filters_list[4],filters_list[5],6,[(1,0),(1,2),(1,0)])
this is an example how to make the aggregation layers
"""

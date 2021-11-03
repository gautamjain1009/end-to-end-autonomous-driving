import torch 
import torch.nn as nn 
import torch.functional as F
from torch.nn.modules.activation import ELU
from torch.nn.modules.linear import Identity
from torch.nn.modules.rnn import LSTM
import torchvision 
import numpy as np 
from torchsummary import summary

"""
To do:: code refactoring. 

expansion factor of 6 for depthwise conv. in all the next residual layers
[16,24,48,88,120,208,352] * 6 == num of filters in the respective residual layers

Pytorch 1.7.1 has no efficient net implemented. 
"""

## starting aggregation block
class intialAggregationBlock(nn.Module):
    def __init__(self, in_channels):
        super(intialAggregationBlock, self).__init__()
        
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(self.in_channels,32,kernel_size=3, padding=(1,1), stride=(2,2))
        self.conv2 = nn.Conv2d(32,32, kernel_size=3, padding=(1,1), stride=(1,1))
        self.conv3 = nn.Conv2d(32,16,kernel_size=1, stride=(1,1))
        self.elu = ELU()
        self.batchnorm1 = nn.BatchNorm2d(32, eps = 0.001, momentum=0.99)
        self.batchnorm2 = nn.BatchNorm2d(32, eps = 0.001, momentum=0.99)
        self.batchnorm3=  nn.BatchNorm2d(16,eps = 0.001, momentum=0.99)        
    
    def forward(self,x):
        x = self.elu(self.batchnorm1(self.conv1(x)))
        x = self.elu(self.batchnorm2(self.conv2(x)))
        x = self.batchnorm3(self.conv3(x))

        return x

#starting residual block
class InitialResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=1, padding =1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,kernel_size=1, stride = 1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels,eps = 0.001, momentum=0.99)
        self.batch_norm2 = nn.BatchNorm2d(out_channels, eps = 0.001, momentum= 0.99)
        self.elu = ELU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.elu(self.batch_norm1(self.conv1(x)))
        x = self.elu(self.batch_norm2(self.conv2(x)))
        x+= identity

        return x

"""
Combine Both 1_3_1 and 1_5_1 classes
"""

#repititive residual block with (1x1) - (3x3) - (1x1) kernel
class BottleneckBlock1_3_1(nn.Module):
    def __init__(self, in_channels, out_channels, d_pad,expansion= 6):
        super(BottleneckBlock1_3_1,self).__init__()

        self.expansion = expansion
        self.d_pad = d_pad
        self.conv1 = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride =1, padding =0)
        self.conv2 = nn.Conv2d(out_channels*self.expansion, out_channels*self.expansion, kernel_size=3, stride =1, padding =self.d_pad)
        self.conv3 = nn.Conv2d(out_channels*self.expansion, out_channels, kernel_size=1, stride =1, padding =0)
        self.elu = ELU()
        self.batch_norm1 = nn.BatchNorm2d(out_channels*self.expansion, eps = 0.001, momentum = 0.99)
        self.batch_norm2 = nn.BatchNorm2d(out_channels*self.expansion, eps = 0.001, momentum = 0.99)
        self.batch_norm3 = nn.BatchNorm2d(out_channels, eps = 0.001, momentum = 0.99)

    def forward(self, x):

        identity = x.clone()
        x = self.elu(self.batch_norm1(self.conv1(x)))
        x = self.elu(self.batch_norm2(self.conv2(x)))
        x = self.elu(self.batch_norm3(self.conv3(x)))
        x+=identity
        return x

# reptitive block with (1x1) - (5x5) - (1x1) kernel
class BottleneckBlock1_5_1(nn.Module):
    def __init__(self,in_channels, out_channels,d_pad,expansion= 6):
        super(BottleneckBlock1_5_1, self).__init__()
        
        self.d_pad = d_pad 
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size=1, stride =1, padding =0)
        self.conv2 = nn.Conv2d(out_channels*self.expansion, out_channels*self.expansion, kernel_size=5, stride =1, padding =self.d_pad)
        self.conv3 = nn.Conv2d(out_channels*self.expansion, out_channels, kernel_size=1, stride =1, padding =0)
        self.elu = ELU()
        self.batch_norm1 = nn.BatchNorm2d(out_channels*self.expansion, eps = 0.001, momentum = 0.99)
        self.batch_norm2 = nn.BatchNorm2d(out_channels*self.expansion, eps = 0.001, momentum = 0.99)
        self.batch_norm3 = nn.BatchNorm2d(out_channels, eps = 0.001, momentum = 0.99)
    
    def forward(self,x):
        identity = x.clone()
        x = self.elu(self.batch_norm1(self.conv1(x)))
        x = self.elu(self.batch_norm2(self.conv2(x)))
        x = self.elu(self.batch_norm3(self.conv3(x)))
        x+=identity
        return x 
 

class AggregationBlock(nn.Module):
    def __init__(self,in_channels, out_channels, expansion, pad_stridepairs= [],kernel_3 = False):
        super(AggregationBlock, self).__init__()
        
        self.pad_stridepairs =pad_stridepairs  
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, in_channels*self.expansion, kernel_size=1, stride =self.pad_stridepairs[0][0], padding = self.pad_stridepairs[0][1])
        
        if kernel_3 == True:
            self.conv2 = nn.Conv2d(in_channels*self.expansion, in_channels*self.expansion, kernel_size=3, stride =self.pad_stridepairs[1][0], padding =self.pad_stridepairs[1][1])
        else:
            self.conv2 = nn.Conv2d(in_channels*self.expansion, in_channels*self.expansion, kernel_size=5, stride =self.pad_stridepairs[1][0], padding =self.pad_stridepairs[1][1])
        
        self.conv3 = nn.Conv2d(in_channels*self.expansion, out_channels, kernel_size=1, stride =self.pad_stridepairs[2][0], padding =self.pad_stridepairs[2][1])
        
        self.elu = ELU()
        self.batch_norm1 = nn.BatchNorm2d(in_channels*self.expansion, eps = 0.001, momentum=0.99)
        self.batch_norm2 = nn.BatchNorm2d(in_channels*self.expansion, eps = 0.001, momentum = 0.99)
        self.batch_norm3 = nn.BatchNorm2d(out_channels, eps = 0.001, momentum = 0.99)          

    def forward(self,x):
        x = self.elu(self.batch_norm1(self.conv1(x)))
        x = self.elu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv1(x))

        return x 


# Combined Resnet for conv features extraction

class ConvFeatureExtractor(nn.Module):
    def __init__(self, filter_list,expansion, num_in_channels =12):
        super(ConvFeatureExtractor,self).__init__()
        
        self.filter_list = filter_list
        self.expansion = expansion
        self.num_in_channels = num_in_channels 

        # last two conv. operations
        self.conv1 = nn.Conv2d(self.filter_list[-1],self.filter_list[-1],kernel_size=1, stride =1)
        self.conv2 = nn.Conv2d(self.filter_list[-1],32, kernel_size=1, stride=1)
        self.elu = ELU()
        self.batchnorm1 = nn.BatchNorm2d(self.filter_list[-1],eps=0.001, momentum=0.99)
        self.batchnorm2 = nn.BatchNorm2d(32,eps =0.001, momentum=0.99)
        
        self.layer1 = intialAggregationBlock(self.num_in_channels)    
        self.layer2 = InitialResBlock(self.filter_list[0], self.filter_list[0]) 
        self.layer3 = AggregationBlock(self.filter_list[0],self.filter_list[1],self.expansion,[(1,0),(2,1),(1,0)],True)
        self.layer4 = BottleneckBlock1_3_1(self.filter_list[1],self.filter_list[1],1)
        self.layer5 = BottleneckBlock1_3_1(self.filter_list[1],self.filter_list[1],1)
        self.layer6 = AggregationBlock(self.filter_list[1],self.filter_list[2],self.expansion,[(1,0),(2,2),(1,0)],False)
        self.layer7 = BottleneckBlock1_5_1(self.filter_list[2],self.filter_list[2],2)
        self.layer8 = BottleneckBlock1_5_1(self.filter_list[2],self.filter_list[2],2)
        self.layer9 = AggregationBlock(self.filter_list[2],self.filter_list[3],self.expansion,[(1,0),(2,1),(1,0)],True)
        self.layer10 = BottleneckBlock1_3_1(self.filter_list[3],self.filter_list[3],1)
        self.layer11 = BottleneckBlock1_3_1(self.filter_list[3],self.filter_list[3],1)
        self.layer12 = BottleneckBlock1_3_1(self.filter_list[3],self.filter_list[3],1)
        self.layer13 =AggregationBlock(self.filter_list[3],self.filter_list[4],self.expansion,[(1,0),(1,2),(1,0)],False)
        self.layer14 = BottleneckBlock1_5_1(self.filter_list[4],self.filter_list[4],2)
        self.layer15 = BottleneckBlock1_5_1(self.filter_list[4],self.filter_list[4],2)
        self.layer16 = BottleneckBlock1_5_1(self.filter_list[4],self.filter_list[4],2)
        self.layer17 = AggregationBlock(self.filter_list[4],self.filter_list[5],self.expansion,[(1,0),(2,2),(1,0)],False)
        self.layer18 = BottleneckBlock1_5_1(self.filter_list[5],self.filter_list[5],2)
        self.layer19 = BottleneckBlock1_5_1(self.filter_list[5],self.filter_list[5],2)
        self.layer20 = BottleneckBlock1_5_1(self.filter_list[5],self.filter_list[5],2)
        self.layer21 = BottleneckBlock1_5_1(self.filter_list[5],self.filter_list[5],2)
        self.layer22 = AggregationBlock(self.filter_list[5],self.filter_list[6],self.expansion,[(1,0),(1,1),(1,0)],True) 
        self.layer23 = BottleneckBlock1_3_1(self.filter_list[6],self.filter_list[6],1)

    def forward(self,x):
        """
        It can be refactored
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer20(x)
        x = self.layer21(x)
        x = self.layer22(x)
        x = self.layer23(x)
        x = self.elu(self.batchnorm1(self.conv1(x)))
        x = self.batchnorm2(self.conv2(x))

        return x 

"""
filters_list = [16,24,48,88,120,208,352]
model = AggregationBlock(filters_list[4],filters_list[5],6,[(1,0),(1,2),(1,0)])
this is an example how to make the aggregation layers
"""
# filters_list = [16,24,48,88,120,208,352]
# expansion =6
# model = ConvFeatureExtractor(filters_list,expansion)

# x = torch.randn(1,12,128,256)
# # x = x.permute(0,2,3,1)
# output = model(x)


"""
Fully connected layers are GEMM operators in ONNX computation graph.
"""
class preLSTMCell(nn.Module):
    def __init__(self,desire, conv_features,traffic_convention):
        super(preLSTMCell,self).__init__()

        self.desire =desire 
        self.conv_features = conv_features
        self.traffic_convention = traffic_convention

        assert self.desire.size() == (1,8), "desire tensor shape is wrong"
        assert self.conv_features.size() == (1,1024), "conv feature tensor shape is wrong"
        assert self.traffic_convention.size() == (1,2), "traffic convention tensor shape is wrong"
        
        self.gemmtolstm = nn.Linear(1034,1024)
        self.elu = ELU()
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.elu(torch.cat((self.conv_features,self.desire, self.traffic_convention),1))
        x = self.relu(self.gemmtolstm(x))
        return x

#LSTM Cell 
# class LSTMCell(nn.Module):
#     def __init__(self,in_prelstmcell, conv_ft_in, desire_in, traffic_conv):
#         super(LSTMCell,self).__init__()
#         self.conv_ft_in = conv_ft_in
#         self.desire_in = desire_in
#         self.traffic_conv = traffic_conv
#         self.in_prelstmcell = preLSTMCell(desire_in, conv_ft_in, traffic_conv)

        

#     def forward(self,x):

#         return x


#### all the dense output head for the network

class pathpredictionModel(nn.Module):
    def __init__(self):
        super(pathpredictionModel,self).__init__()

    def forward():
        pass 


class lanepredictionModel(nn.module):
    def __init__(self):
        super(lanepredictionModel,self).__init__()

    def forward(self,x):

        return x 


class lanePROB(nn.module):
    def __init__(self):
        super(lanePROB,self).__init__()

    def forward(self,x):

        return x 



class ROADpredictionModel(nn.module):
    def __init__(self):
        super(ROADpredictionModel,self).__init__()

    def forward(self,x):

        return x 


class LEADpredictionModel(nn.module):
    def __init__(self):
        super(LEADpredictionModel,self).__init__()

    def forward(self,x):

        return x 


class LEADPROB(nn.module):
    def __init__(self):
        super(LEADPROB,self).__init__()

    def forward(self,x):

        return x 

class desirepredictionModel(nn.module):
    def __init__(self):
        super(desirepredictionModel,self).__init__()

    def forward(self,x):

        return x 


class metapredictionModel(nn.module):
    def __init__(self):
        super(metapredictionModel,self).__init__()

    def forward(self,x):

        return x 

class posepredictionModel(nn.module):
    def __init__(self):
        super(posepredictionModel,self).__init__()

    def forward(self,x):

        return x 



### Combined model

class Combined_model(nn.Module):
    def __init__(self):
        super(Combined_model,self).__init__()



    def forward(self,x):

        return x



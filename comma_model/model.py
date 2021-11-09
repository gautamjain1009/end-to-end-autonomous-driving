import math
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

        x = x.view(-1,1024)

        return x 

"""
Fully connected layers are GEMM operators in ONNX computation graph.
"""
# GRU cell 
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 3 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size)
        self.reset_param()
        self.s = nn.Sigmoid()
        self.m = nn.Tanh()

    def reset_param(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, init_state):
        gate_x = self.x2h(x) 
        gate_h = self.h2h(init_state)
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = self.s(i_r + h_r)
        updategate = self.s(i_i + h_i)
        newgate = self.m(i_n + (resetgate * h_n))
        
        hidden_state = newgate + updategate * (init_state - newgate)
        return hidden_state

class GRUModel(nn.Module):
    def __init__(self):
        super(GRUCell,self).__init__()

        self.gemmtoGRU = nn.Linear(1034,1024)
        self.elu = ELU()
        self.relu = nn.ReLU()
        self.GRUlayer = GRUCell(1024,512)

    def forward(self,desire,conv_features,traffic_convention, init_state):
        
        assert desire.size() == (1,8), "desire tensor shape is wrong"
        assert conv_features.size() == (1,1024), "conv feature tensor shape is wrong"
        assert traffic_convention.size() == (1,2), "traffic convention tensor shape is wrong"

        x = self.elu(torch.cat((conv_features,desire,traffic_convention),1))
        in_GRU = self.relu(self.gemmtoGRU(x,init_state))

        out_GRU = self.GRUlayer(in_GRU,)

        return out_GRU

    def init_initial_tensors(self):
        
        if torch.cuda.is_available():
            self.initialize_initial_state = torch.zeros(1,512).cuda()
            self.initialize_desire = torch.zeros(1,8).cuda()
            self.initialize_traffic_convention = torch.zeros(1,2).cuda()    
        else: 
            self.initialize_initial_state = torch.zeros(1,512)
            self.initialize_desire = torch.zeros(1,8)
            self.initialize_traffic_convention = torch.zeros(1,2)

            return self.initialize_desire, self.initialize_traffic_convention, self.initialize_initial_state

#### all the dense output head for the network #####

class CommanBranchOuputModule(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(CommanBranchOuputModule,self).__init__()
        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.fc1 = nn.Linear(1024,self.input_dim)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim)
        self.fc3 = nn.Linear(self.input_dim,self.input_dim)
        self.fc4 = nn.Linear(self.input_dim,output_dim)
        self.relu = nn.ReLU()

    def forward(self,x):
        x =self.relu(self.fc1(x))
        identity = x.clone()
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x +=identity
        x = self.relu(x)
        x =self.fc4(x) 

        return x

class outputHeads(nn.Module):

    """
    all the ouput heads except meta and pose as they are using conv features only. 
    """
    def __init__(self,inputs_dim, outputs_dim):
        super(outputHeads,self).__init__()
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim 

        self.fc1 = nn.Linear(1536,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.relu = nn.ReLU()
        self.elu = ELU()
        
        self.path_layer= CommanBranchOuputModule(self.inputs_dim["path"], self.outputs_dim["path"])

        self.ll_pred_1_layer = CommanBranchOuputModule(self.inputs_dim["ll_pred"], self.outputs_dim["ll_pred"])
        self.ll_pred_2_layer = CommanBranchOuputModule(self.inputs_dim["ll_pred"], self.outputs_dim["ll_pred"])
        self.ll_pred_3_layer = CommanBranchOuputModule(self.inputs_dim["ll_pred"], self.outputs_dim["ll_pred"])
        self.ll_pred_4_layer = CommanBranchOuputModule(self.inputs_dim["ll_pred"], self.outputs_dim["ll_pred"])
    
        self.ll_prob_layer = CommanBranchOuputModule(self.inputs_dim["llprob"], self.outputs_dim["llprob"])

        self.road_edg_layer1 = CommanBranchOuputModule(self.inputs_dim["road_edges"],outputs_dim["road_edges"])
        self.road_edg_layer2 = CommanBranchOuputModule(self.inputs_dim["road_edges"],outputs_dim["road_edges"])

        self.lead_car_layer = CommanBranchOuputModule(self.inputs_dim["lead_car"], self.outputs_dim["lead_car"])
        self.lead_prob_layer = CommanBranchOuputModule(self.inputs_dim["leadprob"], self.outputs_dim["leadprob"])
        self.desire_layer = CommanBranchOuputModule(self.inputs_dim["desire_state"], self.outputs_dim["desire_state"])
        
        self.meta_layer1 = CommanBranchOuputModule(self.inputs_dim["meta"][0],self.outputs_dim["meta"][0])
        self.meta_layer2 = CommanBranchOuputModule(self.inputs_dim["meta"][1],self.outputs_dim["meta"][1])

        self.pose_layer = CommanBranchOuputModule(self.inputs_dim["pose"],self.outputs_dim["pose"]) 
    
    def forward(self,x,y):
        """
        change torch.reshape to view
        """
        x = self.relu(self.fc1(x))
        y = self.elu(y)
        y = self.relu(self.fc2(y))
        #paths 
        path_pred_out = self.path_layer(x)
        #lanelines
        test_tensor = self.ll_pred_1_layer(x) 
        ll1 = torch.reshape(self.ll_pred_1_layer(x),(1,2,66))
        ll2 = torch.reshape(self.ll_pred_2_layer(x),(1,2,66))
        ll3 = torch.reshape(self.ll_pred_3_layer(x),(1,2,66))
        ll4 = torch.reshape(self.ll_pred_4_layer(x),(1,2,66))
        ll_pred = torch.cat((ll1,ll2,ll3,ll4),2) # concatenated along axis =2
        ll_pred_f = ll_pred.view(-1,ll_pred.size()[0]*ll_pred.size()[1]*ll_pred.size()[2])
        #laneline prob 
        ll_prob = self.ll_prob_layer(x)
        #road Edges
        road_edg_pred1 = torch.reshape(self.road_edg_layer1(x),(1,2,66))
        road_edg_pred2 = torch.reshape(self.road_edg_layer2(x),(1,2,66))
        road_edg_pred = torch.cat((road_edg_pred1, road_edg_pred2),2)
        road_edg_pred_f = road_edg_pred.view(-1,road_edg_pred.size()[0]*road_edg_pred.size()[1]*road_edg_pred.size()[2]) 
        #lead car
        lead_car_pred = self.lead_car_layer(x)
        #lead prob
        lead_prob_pred = self.lead_prob_layer(x)
        #desire state
        desire_pred = self.desire_layer(x)
        #meta1 
        meta1_pred = self.meta_layer1(y)
        #meta2
        meta2_pred = self.meta_layer2(y)
        #pose 
        pose_pred = self.pose_layer(y)
        return path_pred_out, ll_pred_f, ll_prob, road_edg_pred_f, lead_car_pred, lead_prob_pred, desire_pred, meta1_pred,meta2_pred,pose_pred



# ### Combined model

# class Combined_model(nn.Module):
#     def __init__(self):
#         super(Combined_model,self).__init__()



#     def forward(self,x):

#         return x

#####  Random arguments to define the model ##### 

"""
filters_list = [16,24,48,88,120,208,352]
model = AggregationBlock(filters_list[4],filters_list[5],6,[(1,0),(1,2),(1,0)])
this is an example how to make the aggregation layers
"""
# filters_list = [16,24,48,88,120,208,352]
# expansion = 6
# model = ConvFeatureExtractor(filters_list,expansion)

# x = torch.randn(1,12,128,256)
# # x = x.permute(0,2,3,1)
# output = model(x)

# inputs_dim_outputheads= {"path":256, "ll_pred":32, "llprob":16,"road_edges":16 ,"lead_car":64 , "leadprob":16, "desire_state":32, "meta":[64,32], "pose":32}

# output_dim_outputheads = {"path":4955, "ll_pred":132, "llprob":8,"road_edges":132 ,"lead_car":102, "leadprob":3, "desire_state":8, "meta":[48,32], "pose":12}

# model = outputHeads(inputs_dim_outputheads, output_dim_outputheads)

# a = torch.rand(1,1536)
# b = torch.rand(1,1024)
# output = model(a,b)
# print(output.size())


model= GRUCell(1024,512)

a = torch.randn(1,512)
b = torch.randn(1,1024)

out1 = model(b,a)
print(out1.size())
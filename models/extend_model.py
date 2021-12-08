'''
This package provide combined model with trandition ML layer and Tensornetwork Layer.
    - The Tensornetwork Layer's output is supposed not every big.
      Because the output dimesntion `O` or `O= O_1 x O_2 ... O_n` just mean
      simply apply the same structure `O` times with different weight.
      So the cost would be `O` times large.
      In such case, the followed layers are supposed to be `Linear` Layer, or
      CNN layer with input smaller than (10x10)
    - The advanced layers could be any .
'''
import torch
import torch.nn as nn
import numpy as np
from .two_dim_model import *
class Patch2NetworkInput(nn.Module):
    def __init__(self,divide):
        super().__init__()
        self.divide  = divide
    def forward(self,x):
        B, C , dim0, dim1 = tuple(x.shape)
        divide=self.divide
        assert dim0%divide==0
        assert dim1%divide==0
        x     = x.reshape((B, C,dim0//divide,divide,dim1//divide,divide)).permute(0,2,4,3,5,1)
        x     = x.flatten(start_dim=-3,end_dim=-1)
        return 1-x
class LinearCombineModel1(nn.Module):
    def __init__(self,out_features=10,**kargs):
        super().__init__()
        in_features = 16
        self.data_align   = Patch2NetworkInput(4)
        self.feature_layer= PEPS_uniform_shape_symmetry_6x6(out_features=16,**kargs)
        self.classifier   = nn.Linear(in_features,out_features)
    def forward(self,x):
        x = self.data_align(x)
        x = self.feature_layer(x) #(B,16)
        x = self.classifier(x)  #(B,10)
        return x
class LinearCombineModel2(nn.Module):
    def __init__(self,out_features=10,divide=4,**kargs):
        super().__init__()
        inp_channel = 1
        mid_channel = 8
        out_channel = 16
        self.feature_layer= nn.Sequential(
              nn.Conv2d(inp_channel, mid_channel, kernel_size=3, bias=False),
              nn.BatchNorm2d(mid_channel),
              nn.ReLU(inplace=True),
              nn.Conv2d(mid_channel, out_channel, kernel_size=3, bias=False),
              nn.BatchNorm2d(out_channel),
              nn.ReLU(inplace=True))
        self.data_align   = Patch2NetworkInput(4)
        self.network_layer= PEPS_uniform_shape_symmetry_any(W=24//divide,H=24//divide,in_physics_bond=out_channel*divide*divide,out_features=16,**kargs)
        self.classifier   = nn.Linear(16,out_features)
    def forward(self,x):
        x = self.feature_layer(x)
        x = self.data_align(x)
        x = x/x.norm().sqrt()
        x = self.network_layer(x) #(B,16)
        x = self.classifier(x)  #(B,10)
        return x

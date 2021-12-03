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
class LinearCombineModel1(nn.Module):
    def __init__(self,out_features=10,**kargs):
        super().__init__()
        in_features = 16
        self.feature_layer= PEPS_uniform_shape_symmetry_6x6(out_features=16,**kargs)
        self.classifier   = nn.Linear(in_features,out_features)
    def forward(self,x):
        x = self.feature_layer(x) #(B,16)
        x = self.classifier(x)  #(B,10)
        return x
class LinearCombineModel2(nn.Module):
    def __init__(self,out_features=10,**kargs):
        super().__init__()
        in_features = 16
        self.feature_layer= PEPS_uniform_shape_symmetry_any(W=6,H=6,out_features=16,**kargs)
        self.classifier   = nn.Linear(in_features,out_features)
    def forward(self,x):
        x = self.feature_layer(x) #(B,16)
        x = self.classifier(x)  #(B,10)
        return x

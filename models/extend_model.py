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
from .two_dim_model_backup import *
from .convND import convNd
class Patch2NetworkInput(nn.Module):
    def __init__(self,divide,reverse=True):
        super().__init__()
        self.divide  = divide
        self.reverse = reverse

    def forward(self,x):
        B, C , dim0, dim1 = tuple(x.shape)
        divide=self.divide
        assert dim0%divide==0
        assert dim1%divide==0
        x     = x.reshape((B, C,dim0//divide,divide,dim1//divide,divide)).permute(0,2,4,3,5,1)
        x     = x.flatten(start_dim=-3,end_dim=-1)
        if self.reverse:x=1-x
        return x
class LinearCombineModel1(nn.Module):
    def __init__(self,out_features=10,**kargs):
        super().__init__()
        in_features = 16
        self.data_align   = Patch2NetworkInput(4)
        self.feature_layer= PEPS_uniform_shape_symmetry_6x6(out_features=16,in_physics_bond=16,**kargs)
        self.classifier   = nn.Linear(in_features,out_features)
    def forward(self,x):
        x = self.data_align(x)
        x = self.feature_layer(x) #(B,16)
        x = self.classifier(x)  #(B,10)
        return x
class CNNCombineModel(nn.Module):
    def __init__(self,out_features=10,divide=4,inp_channel=1,mid_channel=8,out_channel=16,version=1,**kargs):
        super().__init__()
        self.feature_layer= nn.Sequential(
              nn.Conv2d(inp_channel, mid_channel, kernel_size=3, bias=False),
              nn.BatchNorm2d(mid_channel),
              nn.ReLU(inplace=True),
              nn.Conv2d(mid_channel, out_channel, kernel_size=3, bias=False),
              nn.BatchNorm2d(out_channel),
              nn.ReLU(inplace=True))
        self.data_align   = Patch2NetworkInput(divide)
        if version == 1:
            self.network_layer= PEPS_uniform_shape_symmetry_any_old(W=24//divide,H=24//divide,in_physics_bond=out_channel*divide*divide,out_features=16,**kargs)
        elif version == 2:
            self.network_layer= PEPS_uniform_shape_symmetry_any(W=24//divide,H=24//divide,in_physics_bond=out_channel*divide*divide,out_features=16,**kargs)

        self.classifier   = nn.Linear(16,out_features)

    def load_from(self,path):
        checkpoint = torch.load(path)
        if ('state_dict' not in checkpoint):
            self.load_state_dict(checkpoint)
        else:
            self.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint and hasattr(self,'optimizer') and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'use_focal_loss' in checkpoint:self.focal_lossQ=checkpoint['use_focal_loss']

    def forward(self,x):
        x = self.feature_layer(x)
        x = self.data_align(x)
        x = x/x.norm().sqrt()
        x = self.network_layer(x) #(B,16)
        x = self.classifier(x)  #(B,10)
        return x
class LinearCombineModel2(CNNCombineModel):
    def __init__(self,**kargs):
        super().__init__(inp_channel=1,mid_channel=8,out_channel=16,version=1,**kargs)
class LinearCombineModel3(CNNCombineModel):
    def __init__(self,**kargs):
        super().__init__(inp_channel=1,mid_channel=32,out_channel=64,version=1,**kargs)

class TensorNetworkDeepModel1(nn.Module):
    def __init__(self,out_features=10,divide=4,inp_channel=1,mid_channel=8,out_channel=16,**kargs):
        super().__init__()
        self.data_align   = Patch2NetworkInput(divide,reverse=False)
        self.network_layer= PEPS_uniform_shape_symmetry_deep_model_old(W=24//divide,H=24//divide,
                            in_physics_bond=divide*divide,out_features=16,
                            normlized_layer_module=nn.InstanceNorm3d,
                            nonlinear_layer=nn.Tanh(),
                            **kargs)
        self.classifier   = nn.Linear(16,out_features)

    def forward(self,x):
        x = self.data_align(x)
        x = self.network_layer(x) #(B,16)
        x = self.classifier(x)  #(B,10)
        return x

class ArbitaryPartitionModel1(nn.Module):
    def __init__(self,out_features=10,divide=4,inp_channel=1,mid_channel=8,out_channel=16,**kargs):
        super().__init__()
        self.network_layer= PEPS_einsum_arbitrary_partition_optim(W=24//divide,H=24//divide,
                            in_physics_bond=16,out_features=16,
                            **kargs)

    def forward(self,x):
        x = self.network_layer(x) #(B,16)
        return x

class scale_sigmoid(nn.Module):
    def __init__(self,*args,**kargs):
        super().__init__()
        self.nonlinear=  nn.Tanh()
    def forward(self,x):
        x = self.nonlinear(x)
        #print(f"===>{torch.std_mean(x),x.shape}")
        mi= len(x.shape[1:])
        se= np.prod(x.shape[-1])
        coef = np.sqrt(np.sqrt(se/2))
        x = x/coef
        return x

class scaled_Tanh(nn.Module):
    def __init__(self,coef):
        super().__init__()
        self.nonlinear=  nn.Tanh()
        self.coef = coef
    def forward(self,x):
        x = self.nonlinear(x)
        x = x/self.coef
        return x

def get_ConND(in_channels,out_channels,num_dims,bias=True,**kargs):
    if num_dims == 1:
        cnn = torch.nn.Conv1d(in_channels,out_channels,bias=bias,**kargs)
    elif num_dims == 2:
        cnn = torch.nn.Conv2d(in_channels,out_channels,bias=bias,**kargs)
    elif num_dims == 3:
        cnn = torch.nn.Conv3d(in_channels,out_channels,bias=bias,**kargs)
    else:
        cnn = convNd(in_channels=in_channels,out_channels=out_channels,num_dims=num_dims,use_bias=bias,
                      **kargs)
    return cnn

def cal_scale(shape,alpha):
    factor = np.prod(shape)
    mi     = len(shape)
    se     = np.prod(shape)
    #coef = np.sqrt(np.sqrt(se/2))
    #coef   = np.sqrt(np.sqrt(factor))
    coef   = np.sqrt(np.power(alpha*factor,1/mi))
    return coef

class TensorNetConvND(nn.Module):

    def reset_parameters(self):
        def init_weights(m):
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                torch.nn.init.xavier_uniform_(m.weight)
                #torch.nn.init.kaiming_normal_(m.weight) for leack ReLU
                if m.bias is not None:torch.nn.init.zeros_(m.bias)
        self.apply(init_weights)

    def forward(self,x):
        squeeze = False
        if len(x.shape) ==self.num_dims + 1:
            x = x.unsqueeze(1)
            squeeze=True
        res = x
        out = self.engine(x)
        if self.rensetQ:out+= res
        x = self.resize_layer(out)
        #x = self.dropout(x) # dropout at last layer would heavily destroy learning processing.
        if squeeze:
            x = x.squeeze(1)
        return x

    def set_alpha(self,alpha):
        coef   = cal_scale(self.shape,alpha)
        self.resize_layer=scaled_Tanh(coef)

class TensorNetConvND_Single(TensorNetConvND):
    def __init__(self,shape,channels,alpha=4,rensetQ=True):
        super().__init__()
        num_dims      = len(shape)
        self.shape    = shape
        self.channels = channels
        self.num_dims = num_dims
        self.alpha    = alpha
        self.rensetQ  = rensetQ
        in_channels   = out_channels = channels
        kargs = {'kernel_size':3,
                 'stride':1,
                 'padding':1,
                }
        cnn1 = get_ConND(in_channels,out_channels,num_dims,bias=False,**kargs)
        bn1  = nn.LayerNorm(shape)
        self.engine = nn.Sequential(cnn1,bn1)

        coef   = cal_scale(shape,alpha)
        self.resize_layer=scaled_Tanh(coef)
        self.reset_parameters()
    def __repr__(self):
        return f"TensorNetConvND_Single(shape={self.shape},channels={self.channels},alpha={self.alpha})"
class TensorNetConvND_Block_a(TensorNetConvND):
    def __init__(self,shape,channels,alpha=4,rensetQ=True):
        super().__init__()
        num_dims      = len(shape)
        self.shape    = shape
        self.num_dims = num_dims
        self.channels = channels
        self.alpha    = alpha
        self.rensetQ  = rensetQ
        in_channels   = out_channels = channels
        kargs = {'kernel_size':3,
                 'stride':1,
                 'padding':1,
                }
        interchannels=32
        cnn1 = get_ConND(  in_channels,interchannels,num_dims,bias=False,**kargs)
        cnn2 = get_ConND(interchannels,out_channels ,num_dims,bias=False,**kargs)
        relu = nn.Tanh()
        bn1  = nn.LayerNorm(shape)
        bn2  = nn.LayerNorm(shape)
        self.engine = nn.Sequential(cnn1,bn1,relu,cnn2,bn2)
        coef   = cal_scale(shape,alpha)
        self.resize_layer=scaled_Tanh(coef)
        self.reset_parameters()

    def __repr__(self):
        return f"TensorNetConvND_Block_a(shape={self.shape},channels={self.channels},alpha={self.alpha})"

class TensorAttentionU2D(torch.nn.Module):
    def __init__(self,shape,**kargs):
        super().__init__()
        assert len(shape) == 2
        D1,D2 = shape
        assert D1==D2
        self.shape  = shape
        self.attn_1 = torch.nn.MultiheadAttention(D2,D1,1,batch_first =True)
        self.attn_2 = torch.nn.MultiheadAttention(D1,D2,1,batch_first =True)
    def forward(self,a):
        assert len(a.shape)==3
        assert a.shape[1:]==self.shape
        x = a.permute(0,1,2)
        y = a.permute(0,2,1)
        o1 =  self.attn_1(x,x,y)[0].reshape(-1,*self.shape).permute(0,1,2)
        o2 =  self.attn_2(y,y,x)[0].reshape(-1,*self.shape).permute(0,2,1)
        return (o1+o2)/2

class TensorAttentionU3D(torch.nn.Module):
    def __init__(self,shape,**kargs):
        super().__init__()
        assert len(shape) == 3
        D1,D2,D3 = shape
        assert D1==D2==D3
        self.shape  = shape
        self.attn_1 = torch.nn.MultiheadAttention(D2*D3,1,batch_first =True)
        self.attn_2 = torch.nn.MultiheadAttention(D1*D3,1,batch_first =True)
        self.attn_3 = torch.nn.MultiheadAttention(D1*D2,1,batch_first =True)
    def forward(self,a):
        assert len(a.shape)==4
        assert a.shape[1:]==self.shape
        x = a.permute(0,1,2,3).flatten(-2,-1)
        y = a.permute(0,2,3,1).flatten(-2,-1)
        z = a.permute(0,3,1,2).flatten(-2,-1)
        o1 =  self.attn_1(x,y,z)[0].reshape(-1,*self.shape).permute(0,1,2,3)
        o2 =  self.attn_2(y,z,x)[0].reshape(-1,*self.shape).permute(0,3,1,2)
        o3 =  self.attn_3(z,x,y)[0].reshape(-1,*self.shape).permute(0,2,3,1)
        return (o1+o2+o3)/3

class TensorAttentionU4D(torch.nn.Module):
    def __init__(self,shape,**kargs):
        super().__init__()
        assert len(shape) == 4
        D1,D2,D3,D4 = shape
        assert D1==D2==D3==D4
        self.shape  = shape
        self.attn_1 = torch.nn.MultiheadAttention(D3*D4,1,batch_first =True)
        self.attn_2 = torch.nn.MultiheadAttention(D2*D3,1,batch_first =True)
        self.attn_3 = torch.nn.MultiheadAttention(D1*D2,1,batch_first =True)
    def forward(self,a):
        assert len(a.shape)==5
        assert a.shape[1:]==self.shape
        x = a.permute(0,1,2,3,4).flatten(-4,-3).flatten(-2,-1)
        y = a.permute(0,4,1,2,3).flatten(-4,-3).flatten(-2,-1)
        z = a.permute(0,3,4,1,2).flatten(-4,-3).flatten(-2,-1)
        o1 =  self.attn_1(x,y,z)[0].reshape(*a.shape).permute(0,1,2,3,4)
        o2 =  self.attn_2(y,z,x)[0].reshape(*a.shape).permute(0,2,3,4,1)
        o3 =  self.attn_3(z,x,y)[0].reshape(*a.shape).permute(0,3,4,1,2)
        return (o1+o2+o3)/3

class TensorAttention(torch.nn.Module):
    def __init__(self,shape,channels,alpha=4,**kargs):
        super().__init__()
        self.shape    = shape
        self.channels = channels
        self.alpha    = alpha

        if len(shape) == 2:
            self.engine = TensorAttentionU2D(shape,**kargs)
        elif len(shape) == 3:
            self.engine = TensorAttentionU3D(shape,**kargs)
        elif len(shape) == 4:
            self.engine = TensorAttentionU4D(shape,**kargs)
        else:
            raise NotImplementedErrort
        coef   = cal_scale(shape,alpha)
        self.resize_layer=scaled_Tanh(coef)

    def __repr__(self):
        return f"TensorAttention(shape={self.shape},channels={self.channels},alpha={self.alpha})"

    def forward(self,x):
        reshape = False
        if len(x.shape) == len(self.shape) + 1:
            pass
        elif len(x.shape) == len(self.shape) + 2:
            B,C = x.shape[:2]
            assert C == self.channels
            x = x.reshape(B*C,*self.shape)
            reshape=True
        res = x
        out = self.engine(x)
        out+= res
        x = self.resize_layer(out)
        if reshape:
            x = x.reshape(B,C,*self.shape)
        return x

    def set_alpha(self,alpha):
        coef   = cal_scale(self.shape,alpha)
        self.resize_layer=scaled_Tanh(coef)

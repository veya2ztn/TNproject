import torch
import torch.nn as nn
import torch.nn.functional as F

class MPSLinear(nn.Module):
    '''
     MPSLinear(in_features: int, out_features: int,
                            in_physics_bond: int, out_physics_bond: int, virtual_bond_dim:int,
                            bias: bool = True, label_position: int or str):
        input  (B, in_features , in_physics_bond)
        output (B, out_features,out_physics_bond)
    '''

    def __init__(self, in_features,out_features,
                       in_physics_bond = 2, out_physics_bond=1, virtual_bond_dim=2,
                       bias=True,label_position='center',init_std=1e-10):
        super(MPSLinear, self).__init__()
        if label_position is 'center':label_position = in_features//2
        assert type(label_position) is int
        self.in_features   = in_features
        self.out_features  = out_features
        self.vbd           = virtual_bond_dim
        self.ipb           = in_physics_bond
        self.opb           = out_physics_bond
        self.hn            = label_position
        left_num           = self.hn
        right_num          = in_features - left_num

        bias_mat = torch.eye(self.vbd).unsqueeze(-1).repeat(1,1,self.ipb)
        self.left_tensors = nn.Parameter(init_std * torch.randn(left_num  ,self.vbd,self.vbd, self.ipb)+ bias_mat)
        self.rigt_tensors = nn.Parameter(init_std * torch.randn(right_num ,self.vbd,self.vbd, self.ipb)+ bias_mat)

        bias_mat = torch.eye(self.vbd).unsqueeze(-1).repeat(1,1,self.opb)
        self.cent_tensors = nn.Parameter(init_std * torch.randn(self.out_features,self.vbd,self.vbd, self.opb)+ bias_mat)

    @staticmethod
    def get_chain_contraction(tensor):
        size   = int(tensor.shape[0])
        while size > 1:
            half_size = size // 2
            nice_size = 2 * half_size
            leftover  = tensor[nice_size:]
            tensor    = torch.einsum("mbik,mbkj->mbij",tensor[0:nice_size:2], tensor[1:nice_size:2])
            #(k/2,NB,D,D),(k/2,NB,D,D) <-> (k/2,NB,D,D)
            tensor   = torch.cat([tensor, leftover], axis=0)
            size     = half_size + int(size % 2 == 1)
        return tensor.squeeze(0)


    def forward(self, input_data):
        # the input data shape is (B,L,pd)
        # expand to convolution patch
        embedded_data= input_data
        left_tensors = torch.einsum('wijp,nwp->wnij',self.left_tensors,embedded_data[:,:self.hn])#i.e. (K,NB,b,b)
        rigt_tensors = torch.einsum('wijp,nwp->wnij',self.rigt_tensors,embedded_data[:,-self.hn:])#i.e.(K,NB,b,b)

        left_tensors = self.get_chain_contraction(left_tensors) #i.e. (NB,b,b)
        rigt_tensors = self.get_chain_contraction(rigt_tensors) #i.e. (NB,b,b)

        tensor  = torch.einsum('bip,oplt,bli->bot',left_tensors,self.cent_tensors,rigt_tensors)
        # (NB,b,b) <-> (T,b,b,o) <-> (NB,b,b) ==> (NB,T,t)
        return tensor

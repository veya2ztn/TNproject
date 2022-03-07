import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from engine.torch_dense import approxmate_mps_line,left_canonicalize_MPS,truncated_SVD
from .tensornetwork_base import TN_Base
class MPSLinear_einsum_uniform_shape(TN_Base):
    '''
    For a naive Linear Layer(in_features,out_features,
                             in_physics_bond = 2, out_physics_bond=2, virtual_bond_dim=2,
                             bias=True,label_position='center',init_std=1e-10
                                       ):
        input  (B, in_features)
        output (B, out_features)
    For s simplest MPSLayer(in_features: int, out_features: int,
                            in_physics_bond: int, out_physics_bond: int, virtual_bond_dim:int,
                            bias: bool = True, label_position: int or str):
        input  (B, in_features , in_physics_bond)
        output (B, out_features,out_physics_bond)
    '''
    def __init__(self, in_features,out_features,
                                       in_physics_bond = 2, out_physics_bond=1, virtual_bond_dim=2,
                                       bias=True,label_position='center',init_std=1e-10,contraction_mode = 'recursion'):
        super().__init__()
        if label_position is 'center':
            label_position = in_features//2
        assert type(label_position) is int
        self.in_features   = in_features
        self.out_features  = out_features
        self.vbd           = virtual_bond_dim
        self.ipb           = in_physics_bond
        self.opb           = out_physics_bond
        self.hn            = label_position
        left_num           = self.hn
        right_num          = in_features - left_num

        self.left_tensors = nn.Parameter(self.rde1((left_num  ,self.vbd, self.ipb,self.vbd),init_std))
        self.rigt_tensors = nn.Parameter(self.rde1((right_num ,self.vbd, self.ipb,self.vbd),init_std))
        self.cent_tensors = nn.Parameter(self.rde1((   self.vbd,self.out_features,self.vbd),init_std))
        self.contraction_mode = contraction_mode

    def forward(self, input_data):
        # the input data shape is (B,L,pd)
        # expand to convolution patch
        if self.contraction_mode == 'loop':
            chain_contractor = self.get_batch_chain_contraction_loop
        elif self.contraction_mode == 'recursion':
            chain_contractor = self.get_batch_chain_contraction_fast
        else:raise NotImplementedError
        embedded_data= input_data
        left_tensors = torch.einsum('wipj,bwp->wbij',self.left_tensors,embedded_data[:,:self.hn])#i.e. (K,NB,b,b)
        rigt_tensors = torch.einsum('wipj,bwp->wbij',self.rigt_tensors,embedded_data[:,-self.hn:])#i.e.(K,NB,b,b)

        left_tensors = chain_contractor(left_tensors) #i.e. (NB,b,b)
        rigt_tensors = chain_contractor(rigt_tensors) #i.e. (NB,b,b)

        tensor  = torch.einsum('bip,pol,bli->bo',left_tensors,self.cent_tensors,rigt_tensors)
        # (NB,b,b) <-> (T,b,b,o) <-> (NB,b,b) ==> (NB,T,t)
        return tensor




class MPSLinear_einsum_arbitary_shape(TN_Base):
    '''
    in_features: int,
    out_features: int,
    in_physics_bond: int,
    out_physics_bond:int,
    virtual_bond_dim:list or int,
    bias: bool = True,
    label_position: int or str):
    input  (B, in_features , in_physics_bond)
    output (B, out_features,out_physics_bond)
    '''
    def __init__(self, in_features,out_features,
                                       in_physics_bond = 2, out_physics_bond=1, virtual_bond_dim=2,
                                       bias=True,label_position='center',init_std=1e-10,
                                       contraction_mode = 'loop',
                                       batch_method     = 'loop',
                                       batch_svd_karg   = None):
        super().__init__()
        if label_position is 'center':
            label_position = in_features//2
        assert type(label_position) is int
        if isinstance(virtual_bond_dim,int):
            virtual_bond_dim = [virtual_bond_dim]*(in_features)
        assert isinstance(virtual_bond_dim,list)
        assert len(virtual_bond_dim)==in_features
        self.in_features   = in_features
        self.out_features  = out_features
        self.ipb           = in_physics_bond
        self.opb           = out_physics_bond
        self.hn            = label_position
        left_num           = self.hn
        right_num          = in_features - left_num

        mps_var    =([self.rde1((self.ipb,virtual_bond_dim[0]),init_std).unsqueeze(0)] +
                     [self.rde1((virtual_bond_dim[i-1],self.ipb,virtual_bond_dim[i]),init_std) for i in range(1,self.hn)] +
                     [self.rde1((virtual_bond_dim[self.hn-1],self.out_features,virtual_bond_dim[self.hn]),init_std)]+
                     [self.rde1((virtual_bond_dim[i-1],self.ipb,virtual_bond_dim[i]),init_std) for i in range(self.hn+1,int(in_features))]+
                     [self.rde1((virtual_bond_dim[-1],self.ipb),init_std).unsqueeze(-1)]
                     )
        assert len(mps_var)-1 == int(in_features)
        self.mps_var = [nn.Parameter(v) for v in mps_var]
        self.center  = left_num
        for i, v in enumerate(self.mps_var):
            self.register_parameter(f'mps{i}', param=v)

        self.contraction_mode = contraction_mode
        self.batch_method     = batch_method
        self.batch_svd_karg   = batch_svd_karg
        # print(len(self.mps_var))
        # print(self.hn)
        # print(self.mps_var[0].shape)
        # print(self.mps_var[self.hn-1].shape)
        # print(self.mps_var[self.hn].shape)
        # print(self.mps_var[self.hn+1].shape)
        # print(self.mps_var[-1].shape)

    def forward(self, input_data,offline_data=False,batch_method = "loop"):
        # the input data shape is (B,L,pd)
        # expand to convolution patch
        embedded_data= input_data
        if batch_method == 'loop' and (not offline_data):
            chain_contractor = self.get_batch_chain_contraction_loop
            out_list = []
            L_B_D_shape_input       = input_data.permute(1,0,2)
            batch_contracted_tensor = []
            for i in range(len(L_B_D_shape_input)):
                j = i if i < self.center else i+1
                batch_contracted_tensor.append(
                    torch.einsum('kp,apb->kab', L_B_D_shape_input[i]  ,self.mps_var[j])
                )
            left_tensors = chain_contractor(batch_contracted_tensor[:self.center]) #i.e. (NB,b,b)
            rigt_tensors = chain_contractor(batch_contracted_tensor[self.center:]) #i.e. (NB,b,b)

            out          = torch.einsum('bip,pol,bli->bo',left_tensors,self.mps_var[self.hn],rigt_tensors)
            return out
            # for batch_input_unit in :
            #     left_tensors = [torch.einsum('p,apb->ab', now_batch[i]  ,self.mps_var[i]) for i in range(self.hn)]
            #     rigt_tensors = [torch.einsum('p,apb->ab', now_batch[i-1],self.mps_var[i]) for i in range(self.hn+1,len(self.mps_var))]
            #     left_tensors = chain_contractor(left_tensors) #i.e. (NB,b,b)
            #     rigt_tensors = chain_contractor(rigt_tensors) #i.e. (NB,b,b)
            #     out          = torch.einsum('ip,pol,li->o',left_tensors,self.mps_var[self.hn],rigt_tensors)
            #     out_list.append(out)
            # return torch.stack(out_list)
        elif  offline_data or (batch_method == "batch"):
            chain_contractor = self.get_chain_contraction_loop
            if not offline_data:
                input_data    = self.batch_together(input_data,self.batch_svd_karg)
            input_data[0] = input_data[0].unsqueeze(0)
            # (1,P,B) <-> (B,P,B) <-> (B,P,B) <-> (B,P,B) <-> (B,P,B)
            left_tensors = [torch.einsum('cpd,apb->acbd', input_data[i]  ,self.mps_var[i]).flatten(0,1).flatten(-2,-1) for i in range(self.hn)]
            # (1,BD) <-> (BD,BD) <-> (BD,BD) <-> (BD,BD) <-> (BD,BD)
            rigt_tensors = [torch.einsum('cpd,apb->acbd', input_data[i-1]  ,self.mps_var[i]).flatten(0,1).flatten(-2,-1) for i in range(self.hn+1,len(self.mps_var))]
            ## (BD,BD) <-> (BD,BD) <-> (BD,BD) <-> (BD,BD) <-> (BD,B)
            D_L,_,D_R = self.mps_var[self.hn].shape
            Batch     = rigt_tensors[-1].shape[-1]
            left_tensors = chain_contractor(left_tensors).reshape(D_L,-1) #i.e. (1,BP) ==> (B,P)
            rigt_tensors = chain_contractor(rigt_tensors).reshape(D_R,-1,Batch) #i.e. (B,B)
            out          = torch.einsum('ia,ioj,jab->bo',left_tensors,self.mps_var[self.hn],rigt_tensors)
            # (BP) <-> (B,P,B) <-> (B,B) ===> (1,P,B) ===> (B,P)
            return out
        else:
            raise NotImplementedError

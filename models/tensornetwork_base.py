import torch
import torch.nn as nn
import torch.nn.functional as F
import opt_einsum as oe
from inspect import signature
import numpy as np
import os
import json
from .model_utils import full_size_array_string
path_recorder  = "./arbitrary_shape_path_recorder.json"
class TN_Base(nn.Module):
    def __init__(self,einsum_engine=torch.einsum,path_record=path_recorder):
        super().__init__()
        if einsum_engine == torch.einsum and 'optimize' in signature(torch.einsum).parameters:
            self.einsum = torch.einsum
        else:
            self.einsum = oe.contract
        if path_record is not None:
            if isinstance(path_record,str):
                if os.path.exists(path_record):
                    with open(path_record,'r') as f:
                        self.path_record = json.load(f)
                else:
                    self.path_record  = {}
                self.path_record_file = path_record
            else:
                self.path_record = path_record
    @staticmethod
    def rde1(shape,init_std):
        if len(shape) == 2:
            a,b = shape
            bias_mat     = torch.eye(a, b)
            tensor       = init_std * torch.randn(a, b)+ bias_mat
        elif len(shape) >= 3:
            a,b,c        = shape[-3:]
            bias_mat     = torch.eye(a, c).unsqueeze(1).repeat(1,b,1)
            tensor       = init_std * torch.randn(*shape)+ bias_mat
        #tensor/=tensor.norm()
        return tensor

    @staticmethod
    def rde2D3(shape,init_std,offset=2):
        size_shape = shape[:offset]
        bias_shape = shape[offset:]
        if len(bias_shape) ==2 :
            bias_mat   = torch.eye(*bias_shape)
        elif len(bias_shape) == 3:
            a,b,c   = bias_shape
            bias_mat   = torch.kron(torch.ones(a),torch.eye(b,c)).reshape(a,b,c)
        elif len(bias_shape) == 4:
            a,b,c,d   = bias_shape
            bias_mat   = torch.kron(torch.ones(a,b),torch.eye(c,d)).reshape(a,b,c,d)
        bias_mat  /= bias_mat.norm()
        bias_mat   = bias_mat.repeat(*size_shape,*([1]*len(bias_shape)))
        tensor     = init_std * torch.randn(*shape)+ bias_mat
        #tensor/=tensor.norm()
        return tensor

    @staticmethod
    def rde2D(shape,init_std,offset=2,physics_index=None):
        size_shape = shape[:offset]
        bias_shape = shape[offset:]
        max_dim    = max(bias_shape)
        full_rank  = len(bias_shape)
        half_rank  = full_rank//2
        rest_rank  = full_rank-half_rank
        bias_mat   = torch.eye(max_dim**half_rank,max_dim**rest_rank)
        bias_mat   = bias_mat.reshape(*([max_dim]*full_rank))
        for i,d in enumerate(bias_shape):
            bias_mat   = torch.index_select(bias_mat, i,torch.arange(d))
        norm       = np.sqrt(np.prod(bias_shape))
        if physics_index is not None:
            norm  *= np.sqrt(size_shape[physics_index])
        #print(norm)
        bias_mat  /= norm
        bias_mat   = bias_mat.repeat(*size_shape,*([1]*len(bias_shape)))
        tensor     = init_std * torch.randn(*shape)+ bias_mat
        #tensor    /= tensor.norm()
        return tensor

    @staticmethod
    def batch_together(inputs,svd_kargs=None):
        inputs= inputs.permute(1,2,0)#(B,num,k)->(num,k,B)
        inputs= torch.diag_embed(inputs)#(num,k,B)->(num,k,B,B)
        inputs= inputs.permute(0,2,1,3)#(num,k,B,B)->(num,B,k,B)
        inputs= [v for v in inputs]
        inputs[0]= torch.diagonal(inputs[0], dim1=0, dim2=-1)#(B,k,B) -> #(k,B)
        # now is
        # (P,B) <-> (B,P,B) <-> (B,P,B) <-> (B,P,B) <-> (B,P,B)
        if svd_kargs is not None:
            DCEngine = lambda x:truncated_SVD(x,output='QR',**svd_kargs)
            inputs,Z_list = left_canonicalize_MPS(inputs,Decomposition_Engine=DCEngine,normlization=False)
        return inputs

    @staticmethod
    def get_batch_chain_contraction_fast(tensor):
        size   = len(tensor)
        while size > 1:
            half_size = size // 2
            nice_size = 2 * half_size
            leftover  = tensor[nice_size:]
            tensor    = torch.einsum("mbik,mbkj->mbij",tensor[0:nice_size:2], tensor[1:nice_size:2])
            #(k/2,NB,D,D),(k/2,NB,D,D) <-> (k/2,NB,D,D)
            tensor   = torch.cat([tensor, leftover], axis=0)
            size     = half_size + int(size % 2 == 1)
        return tensor.squeeze(0)

    @staticmethod
    def get_batch_chain_contraction_loop(tensor):
        now_tensor= tensor[0]
        for next_tensor in tensor[1:]:
            now_tensor = torch.einsum("bik,bkj->bij",now_tensor, next_tensor)
        return now_tensor


    @staticmethod
    def get_chain_contraction_memrory_save(tensor):
        size   = len(tensor)
        while size > 1:
            half_size = size // 2
            nice_size = 2 * half_size
            leftover  = tensor[nice_size:]
            left_idx  = list(range(0,nice_size,2))
            rigt_idx  = list(range(1,nice_size,2))
            tensor    = [torch.einsum("ik,kj->ij",tensor[l_idx],tensor[r_idx]) for l_idx,r_idx in zip(left_idx,rigt_idx)]
            tensor   += leftover
            size      = len(tensor)
        return tensor[0]

    @staticmethod
    def get_chain_contraction_loop(tensor):
        size   = len(tensor)
        now_tensor= tensor[0]
        for next_tensor in tensor[1:]:
            now_tensor = torch.einsum("ik,kj->ij",now_tensor, next_tensor)
        return now_tensor


    @staticmethod
    def flatten_image_input(batch_image_input):
        if len(batch_image_input.shape)==5:batch_image_input=batch_image_input.squeeze(1)
        bulk_input = batch_image_input[...,1:-1,1:-1,:].flatten(-3,-2)
        edge_input = torch.cat([batch_image_input[...,0,1:-1,:],
                                batch_image_input[...,1:-1,[0,-1],:].flatten(-3,-2),
                                batch_image_input[...,-1,1:-1,:]
                               ],-2)
        corn_input = batch_image_input[...,[0,0,-1,-1],[0,-1,0,-1],:]
        return bulk_input,edge_input,corn_input

    def get_best_contracting_path(self,*operands):
        array_string = full_size_array_string(*operands)
        if array_string not in self.path_record:
            self.path_record[array_string] = oe.contract_path(*operands,optimize='random-greedy')[0]
            if hasattr(self,'path_record_file'):
                with open(self.path_record_file,'w') as f:json.dump(self.path_record,f)
        if 'path' in self.path_record[array_string]:
            return self.path_record[array_string]['path']
        else:
            return self.path_record[array_string]
    def einsum_engine(self,*operands,optimize=None):
        #path_id,equation,*batch_input):
        if optimize is not None:
            # for now, we will compute path at outside since we need build the contraction map
            # in __init__ phase. So build map at that time make sense
            return self.einsum(*operands,optimize=optimize)
        if not hasattr(self,'path_record'):self.path_record={}
        path = self.get_best_contracting_path(*operands)
        return self.einsum(*operands, optimize=path)

    def load_from(self,path):
        checkpoint = torch.load(path)
        if ('state_dict' not in checkpoint):
            self.load_state_dict(checkpoint)
        else:
            self.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint and hasattr(self,'optimizer') and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'use_focal_loss' in checkpoint:self.focal_lossQ=checkpoint['use_focal_loss']

    def weight_init(self):
        raise NotImplementedError

import torch
import torch.nn as nn
import torch.nn.functional as F
class TN_Base(nn.Module):
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
        return tensor

    @staticmethod
    def rde2D(shape,init_std,offset=2):
        size_shape = shape[:offset]
        bias_shape = shape[offset:]
        bias_mat   = torch.zeros(bias_shape)
        diag_idx   = [list(range(min(bias_shape)))]*len(bias_shape)
        bias_mat[diag_idx] = 1
        for i in range(offset):bias_mat=bias_mat.unsqueeze(0)
        bias_mat   = bias_mat.repeat(*size_shape,*([1]*len(bias_shape)))
        tensor     = init_std * torch.randn(*shape)+ bias_mat
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

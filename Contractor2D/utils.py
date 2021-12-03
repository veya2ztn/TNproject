import torch

def apply_SVD(tensor,left_bond=[0,1],right_bond=[2,3],truncate=None):
    shape     = tensor.shape
    size_s    = list(shape[:-4])
    offset    = len(tensor.shape)-4
    left_bond = [i+offset for i in left_bond]
    right_bond= [i+offset for i in right_bond]
    out_left_shape = [shape[i] for i in left_bond]
    out_right_shape= [shape[i] for i in right_bond]
    index_ord = list(range(offset))+left_bond+right_bond
    tensor    = tensor.permute(*index_ord).flatten(-4,-3).flatten(-2,-1)
    U,S,V     = torch.svd(tensor)
    V = V.transpose(-2,-1)
    S = S.sqrt()
    dim = S.shape[-1]
    U = U*S.unsqueeze(-2)
    V = V*S.unsqueeze(-1)
    U = U.reshape(*size_s,*out_left_shape,dim)
    V = V.reshape(*size_s,dim,*out_right_shape)
    return U,V

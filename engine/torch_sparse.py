import torch
import torch_sparse
from tqdm.notebook import tqdm
def transpose_sparse_tensor(coalesce_sparse_tensor):
    return coalesce_sparse_tensor.transpose(1,0).coalesce()

def matmul_sparse_tensor(coalesce_sparse_tensor_1,coalesce_sparse_tensor_2):
    m,k1   = coalesce_sparse_tensor_1.size()
    k2,n   = coalesce_sparse_tensor_2.size()
    assert k1==k2
    indexA = coalesce_sparse_tensor_1.indices()
    valueA = coalesce_sparse_tensor_1.values()

    indexB = coalesce_sparse_tensor_2.indices()
    valueB = coalesce_sparse_tensor_2.values()

    indexC, valueC = torch_sparse.spspmm(indexA, valueA, indexB, valueB, m, k1, n)
    return torch.sparse_coo_tensor(indexC, valueC, (m,n)).coalesce()
def reshape_sparse_tensor(coalesce_sparse_tensor,target_shape):
    indices = coalesce_sparse_tensor.indices().tolist()
    size    = coalesce_sparse_tensor.size()
    assert np.prod(size)==np.prod(target_shape)
    target_indices = np.stack(np.unravel_index(np.ravel_multi_index(indices,size),target_shape))
    target_indices = torch.Tensor(target_indices)
    target_indices = target_indices.to(coalesce_sparse_tensor.device)
    tensor = torch.sparse_coo_tensor(target_indices, coalesce_sparse_tensor.values(), target_shape).coalesce()
    #tensor.to(coalesce_sparse_tensor.device)
    return tensor

def sparse_diagonal(dense_matrix):
    A = len(dense_matrix)
    idx = torch.stack([torch.arange(A),torch.arange(A)]).to(dense_matrix.device)
    return torch.sparse_coo_tensor(idx,dense_matrix, (A,A)).coalesce()

def sparse_take_first(coalesce_sparse_tensor,row_num,axis=0):
    size    = list(coalesce_sparse_tensor.shape)
    if row_num > size[axis]:return coalesce_sparse_tensor
    indexes = coalesce_sparse_tensor.indices()
    value   = coalesce_sparse_tensor.values()
    good_i=indexes[axis]<row_num
    indexes = torch.stack([t[good_i] for t in indexes])
    value   = value[good_i]
    size[axis] = row_num
    return torch.sparse_coo_tensor(indexes,value, size).coalesce().to(coalesce_sparse_tensor.device)

def diagonal_tensor_svd_torch_sparse_2D(tensor):
    W,H   = tensor.size()
    if W>=H:
        MstarM     = matmul_sparse_tensor(transpose_sparse_tensor(tensor),tensor)
    else:
        MstarM     = matmul_sparse_tensor(tensor,transpose_sparse_tensor(tensor))
    diagonal_svd_flag = True
    a,b = MstarM.indices()
    if (a!=b).any():
        return None
    L,L = MstarM.shape
    batch_diag  = torch.Tensor([MstarM[i,i] for i in range(L)]).to(tensor.device)
    #batch_diag = MstarM.values()
    # in sparse representation, only value > 0 takes.
    batch_diag,batch_order = batch_diag.sort(-1,descending=True)
    s          = batch_diag.sqrt()
    nonzero_num= torch.sum(s>0)
    s          = s[:nonzero_num]
    A = len(batch_order)
    index = torch.stack([torch.arange(A).to(tensor.device),batch_order])
    value = torch.ones(A).to(tensor.device)
    if W>=H:
        v = torch.sparse_coo_tensor(index,value,(A,A)).coalesce()
        if nonzero_num < A :v = sparse_take_first(v,nonzero_num,axis=1)
        u = matmul_sparse_tensor(tensor,matmul_sparse_tensor(transpose_sparse_tensor(v),sparse_diagonal(1/s)))
    else:
        u = torch.sparse_coo_tensor(index,value,(A,A)).coalesce()
        if nonzero_num < A :u = sparse_take_first(u,nonzero_num,axis=0)
        v = matmul_sparse_tensor(matmul_sparse_tensor(sparse_diagonal(1/s),u),tensor)
        u = transpose_sparse_tensor(u)
    return u,s,v

def reciprocal_sparse_tensor(coalesce_sparse_tensor):
    size   = coalesce_sparse_tensor.size()
    indexA = coalesce_sparse_tensor.indices()
    valueA = coalesce_sparse_tensor.values()
    return torch.sparse_coo_tensor(indexA, 1/valueA, size).coalesce()

def diagonal_tensor_RQ_torch_sparse(tensor):
    W,H   = tensor.shape
    assert W<=H
    MstarM     = matmul_sparse_tensor(tensor,transpose_sparse_tensor(tensor))
    a,b = MstarM.indices()
    assert (a==b).any()
    #batch_diag = MstarM.values()
    #s = batch_diag.sqrt()
    R = MstarM.sqrt()
    Q = matmul_sparse_tensor(reciprocal_sparse_tensor(R),tensor)
    return R,Q

def right_canonicalize_MPS_torch_sparse(mps_line,Decomposition_Engine=diagonal_tensor_RQ_torch_sparse,
                          final_normlization =True,all_renormlization=False
                          ):
    new_chain = []
    R         = None
    Z_list    = []
    # assume every mps_unit is store (kD,D)
    for i,tensor in enumerate(mps_line[::-1]):
        if R is not None:
            new_tensor = matmul_sparse_tensor(tensor,R)
        else:
            new_tensor = tensor
        kD,D  = new_tensor.shape
        if kD>D:
            new_tensor = reshape_sparse_tensor(new_tensor,(D,kD))

        if i == len(mps_line) - 1:
            if final_normlization:
                Z = (new_tensor**2).values().sum().sqrt()
                new_tensor /= Z
                Z_list.append(Z)
            new_chain.append(new_tensor)
        else:
            if all_renormlization:
                Z = (new_tensor**2).values().sum().sqrt()
                new_tensor /= Z
                Z_list.append(Z)
            R,Q = Decomposition_Engine(new_tensor)[:2]
            new_chain.append(Q)
    new_chain=new_chain[::-1]
    return new_chain,Z_list

def left_canonicalize_MPS_torch_sparse(mps_line,Decomposition_Engine=None,
                          final_normlization =True,all_renormlization=False):
    # for any not canonical mps line
    # the chain size (D,P,D)
    new_chain = []
    R         = None
    Z_list    = []# record the scale information for each unit.
    # assume every mps_unit is store (D,kD)
    for i,tensor in enumerate(tqdm(mps_line)):
        if R is not None:
            D,kD       = tensor.shape
            B,D        = R.shape
            new_tensor = matmul_sparse_tensor(R,tensor)
            new_shape  = new_tensor.shape
            new_tensor = reshape_sparse_tensor(new_tensor,(B*kD//D,D))
        else:
            new_tensor = tensor
        if i == len(mps_line) - 1:
            if final_normlization:
                Z = (new_tensor**2).values().sum().sqrt()
                new_tensor /= Z
                Z_list.append(Z)
            new_chain.append(new_tensor)
        else:
            if all_renormlization:
                Z = (new_tensor**2).values().sum().sqrt()
                new_tensor /= Z
                Z_list.append(Z)
            Q,R,_,diagonal_svd_flag = Decomposition_Engine(new_tensor)
            new_chain.append(Q)
           # print(Q.shape)
#         if not diagonal_svd_flag:
#             print(f"full matrix SVD at unit {i}")

    return new_chain,Z_list

def truncated_SVD_torch_sparse(tensor,output='RQ',max_singular_values= None,
                          max_truncation_error= None,
                          relative = True,
                          normlized= True,
                          verbose  = False):
    # the canonocal
    # tensor is batched
    assert len(tensor.shape)  == 2
    diagonal_svd_flag = True
    out = diagonal_tensor_svd_torch_sparse_2D(tensor)
    if out is None:
        diagonal_svd_flag=False
        q = min(tensor.size())
        q = min(q,max_singular_values)
        u, s, v = torch.svd_lowrank(tensor,q=q)
        v       = v.T
        #print(q,s.shape)
    else:
        u, s, v = out
        max_singular_values = s.shape[-1] if max_singular_values is None else max_singular_values
        if max_truncation_error is not None and len(s)>max_singular_values:
            # Cumulative norms of singular values in ascending order
            s_normlized  = s**2
            s_normlized /= torch.sum(s_normlized)
            s_sorted,_   = torch.sort(s_normlized)# 0.1,0.2,...,0.4
            trunc_errs   = torch.sqrt(torch.cumsum(s_sorted,0))# 0.1,0.3,....1
            # If relative is true, rescale max_truncation error with the largest
            # singular value to yield the absolute maximal truncation error.
            num_sing_vals_err = torch.sum(trunc_errs > max_truncation_error)
            if max_singular_values>num_sing_vals_err and verbose:
                print(f"use {num_sing_vals_err}/{max_singular_values} sing vals")
        else:
            num_sing_vals_err  = max_singular_values

        nk = min(max_singular_values, num_sing_vals_err)


        #s_rest = s[...,num_sing_vals_keep:]

        u  = sparse_take_first(u,nk,axis=1) if u.is_sparse else u[...,:nk]
        s  = s[...,:nk]
        v  = sparse_take_first(v,nk,axis=0) if v.is_sparse else v[:nk,:]
    Z  = None #maybe add in the furture
    if output == 'RQ':
        R = u*s[None] if not u.is_sparse else matmul_sparse_tensor(u,sparse_diagonal(s))
        Q = v
        if not R.is_sparse:R = R.to_sparse()
        if not Q.is_sparse:Q = Q.to_sparse()
        output = [R,Q,Z,diagonal_svd_flag]
    elif output == 'QR':
        Q = u
        R = s[:,None]*v if not  v.is_sparse else matmul_sparse_tensor(sparse_diagonal(s),v)
        if not R.is_sparse:R = R.to_sparse()
        if not Q.is_sparse:Q = Q.to_sparse()
        output = [Q,R,Z,diagonal_svd_flag]
    else:
        if not u.is_sparse:u = u.to_sparse()
        if not v.is_sparse:v = v.to_sparse()
        output = [u,s,v,Z,diagonal_svd_flag]
    return output

import torch
import numpy as np
import sparse
import scipy
def diagonal_tensor_svd(tensor):
    assert len(tensor.shape) in [2,3]
    if len(tensor.shape)==2:
        return diagonal_tensor_svd_2D(tensor)
    else:
        return diagonal_tensor_svd_3D(tensor)
def diagonal_tensor_svd_2D(tensor):
    assert len(tensor.shape) ==2
    W,H   = tensor.shape
    if W>=H:
        MstarM     = tensor.T@tensor
    else:
        MstarM     = tensor@tensor.T
    # checkdiagonal(MstarM)
    #print(np.count_nonzero(MstarM.todense() - np.diag(np.diagonal(MstarM.todense()))))
    #print(MstarM.todense())
    idx,idy = MstarM.coords
    # check is diagonal
    if (idx!=idy).any():return None
    batch_diag = sparse.diagonal(MstarM)
    #if (batch_diag**2).sum()!=(MstarM**2).sum():return None
    batch_diag = batch_diag.todense()
    batch_order= batch_diag.argsort()[::-1]
    batch_diag =batch_diag[batch_order]
    s          = np.sqrt(batch_diag)
    nonzero_idx= s>0
    s          = s[nonzero_idx]
    if W>=H:
        A = len(batch_order)
        v = sparse.COO([list(range(A)),batch_order], [1.0]*A,(A,A))
        v = v[nonzero_idx]
        u = tensor@((v.T)/s[None])
    else:
        A = len(batch_order)
        u = sparse.COO([list(range(A)),batch_order], [1.0]*A,(A,A)).T
        u = u[...,nonzero_idx]
        v = ((u.T)/s[:,None])@(tensor)

    return u,s,v,diagonal_svd_flag
def diagonal_tensor_svd_3D(tensor):
    assert len(tensor.shape) ==3
    W,H   = tensor.shape[-2:]
    if W>=H:
        batch_diag = np.stack([sparse.diagonal(t.T@t).todense() for t in tensor])
    else:
        batch_diag = np.stack([sparse.diagonal(t@t.T).todense() for t in tensor])
    batch_order= batch_diag.argsort(axis=-1)[...,::-1]
    batch_diag = np.take_along_axis(batch_diag,batch_order,axis=-1)
    s  = np.sqrt(batch_diag)
    K,A= batch_order.shape
    if W>=H:
        v  = sparse.COO([list(range(K*A)),batch_order.flatten().tolist()], [1.0]*K*A,(K*A,A))
        v  = v.reshape((K,A,A))
        u  = sparse.stack([t@((vv.T)/ss[None]) for t,vv,ss in zip(tensor,v,s)])
    else:
        u  = sparse.COO([list(range(K*A)),batch_order.flatten().tolist()], [1.0]*K*A,(K*A,A)).transpose((-1,-2))
        u  = v.reshape((K,A,A))
        v  = sparse.stack([((vv.T)/ss[:,None])@t for t,vv,ss in zip(tensor,u,s)])
    return u,s,v
def diagonal_tensor_RQ(tensor):
    W,H   = tensor.shape
    assert W<=H
    MstarM     = tensor@tensor.T
    batch_diag = sparse.diagonal(MstarM)
    #assert (batch_diag**2).sum()==(MstarM**2).sum()
    s = np.sqrt(batch_diag.todense())
    R = np.sqrt(MstarM)
    Q = tensor/s[:,None]
    return R,Q
def truncated_SVD(tensor,output='RQ',max_singular_values= None,
                          max_truncation_error= None,
                          relative = True,
                          normlized= True,
                          verbose  = False):
    # the canonocal
    # tensor is batched
    assert len(tensor.shape)  == 2
    out  = diagonal_tensor_svd_2D(tensor)
    if out is None:
        u,s,v = scipy.linalg.svd(tensor.todense(),full_matrices=False)
        # TODO: may use randomlized SVD for low rank matrix
        diagonal_svd_flag=False
        u = sparse.as_coo(u)
        v = sparse.as_coo(v)
    else:
        u,s,v = out
        diagonal_svd_flag=True
    max_singular_values = s.shape[-1] if max_singular_values is None else max_singular_values

    if max_truncation_error is not None:
        # Cumulative norms of singular values in ascending order
        s_normlized  = s**2
        s_normlized /= np.sum(s_normlized)
        s_sorted= np.sort(s_normlized)# 0.1,0.2,...,0.4
        trunc_errs  = np.sqrt(np.cumsum(s_sorted))# 0.1,0.3,....1
        # If relative is true, rescale max_truncation error with the largest
        # singular value to yield the absolute maximal truncation error.
        num_sing_vals_err = np.sum(trunc_errs > max_truncation_error)
        if max_singular_values>num_sing_vals_err and verbose:
            print(f"use {num_sing_vals_err}/{max_singular_values} sing vals")
    else:
        num_sing_vals_err  = max_singular_values

    num_sing_vals_keep = min(max_singular_values, num_sing_vals_err)


    #s_rest = s[...,num_sing_vals_keep:]
    u      = u[...,:num_sing_vals_keep]
    s      = s[...,:num_sing_vals_keep]
    v      = v[...,:num_sing_vals_keep,:]

    if num_sing_vals_keep == s.shape[-1] and normlized:
        Z = 1.0
    else:
        Z = np.sum(s**2).sqrt()

    if output == 'RQ':
        R = u*s[None]
        Q = v
        output = [R,Q,Z,diagonal_svd_flag]
    elif output == 'QR':
        Q = u
        R = s[:,None]*v
        output = [Q,R,Z,diagonal_svd_flag]
    else:
        output = [u,s,v,Z,diagonal_svd_flag]
    return output
def left_canonicalize_MPS(mps_line,Decomposition_Engine=None,
                          final_normlization =True,all_renormlization=False):
    # for any not canonical mps line
    # the chain size (D,P,D)
    if Decomposition_Engine is None:
        Decomposition_Engine=lambda x:truncated_SVD(x,output='QR',
                                                    max_truncation_error=0.00,
                                                    #max_singular_values=500,
                                                   )
    new_chain = []
    R         = None
    Z_list    = []# record the scale information for each unit.
    # for a perfect MPS state, we expect the norm for each tensor is 1.
    for i,tensor in enumerate(tqdm(mps_line)):
        if len(tensor.shape)==2:
            new_tensor = sparse.tensordot(R,tensor,[[1],[0]]) if R is not None else tensor
            #new_tensor = torch.einsum('ab,bd->ad',R,tensor) if R is not None else tensor
            shape      = new_tensor.shape
        elif len(tensor.shape)==3:
            new_tensor = sparse.tensordot(R,tensor,[[1],[0]]) if R is not None else tensor
            #new_tensor = torch.einsum('ab,bcd->acd',R,tensor) if R is not None else tensor
            shape      = new_tensor.shape
            a,b,c = shape
            new_tensor = new_tensor.reshape((a*b,c))
        else:
            raise NotImplementedError

        if i == len(mps_line) - 1:
            if final_normlization:
                Z = np.sqrt((new_tensor**2).sum())
                new_tensor /= (Z)
                Z_list.append(Z)
            new_chain.append(new_tensor.reshape((*shape[:-1],-1)))
        else:
            if all_renormlization:
                Z = np.sqrt((new_tensor**2).sum())
                new_tensor /= Z
                Z_list.append(Z)
            Q,R,_,diagonal_svd_flag = Decomposition_Engine(new_tensor)
            #print(Q.shape)
            Q   = Q.reshape((*shape[:-1],-1))
            new_chain.append(Q)
        if not diagonal_svd_flag:
            print(f"full matrix SVD at unit {i}")

    return new_chain,Z_list
def right_canonicalize_MPS(mps_line,Decomposition_Engine=diagonal_tensor_RQ,
                          final_normlization =True,all_renormlization=False
                          ):
    if Decomposition_Engine is None:
        Decomposition_Engine=lambda x:truncated_SVD(x,output='RQ',
                                                    #max_truncation_error=0.00,
                                                    #max_singular_values=500,
                                                   )
    new_chain = []
    R         = None
    Z_list    = []
    #svd_Z         = torch.Tensor([1.0])#input has already been normalized
    for i,tensor in enumerate(mps_line[::-1]):

        if len(tensor.shape)==2:
            new_tensor = sparse.tensordot(tensor,R,[[1],[0]]) if R is not None else tensor
            #new_tensor = torch.einsum('ab,bc->ac',tensor, R) if R is not None else tensor
            shape      = new_tensor.shape
        elif len(tensor.shape)==3:
            new_tensor = sparse.tensordot(tensor,R,[[2],[0]]) if R is not None else tensor
            #new_tensor = torch.einsum('alb,bc->alc',tensor, R) if R is not None else tensor
            shape      = new_tensor.shape
            a,b,c      = shape
            new_tensor = new_tensor.reshape((a,b*c))
        else:
            raise NotImplementedError

        # normlization is necessary; directly use the SVD_Z may cause problem due to precision
        #print(f"svd_Z{svd_Z.item()}<->all_Z {Z.item()} <-> after_Z{torch.norm(new_tensor).item()}")
        #if normlization:new_tensor /= Z
        if i == len(mps_line) - 1:
            if final_normlization:
                Z = np.sqrt((new_tensor**2).sum())
                new_tensor /= (Z)
                Z_list.append(Z)
            new_chain.append(new_tensor.reshape((-1,*shape[1:])))
        else:
            if all_renormlization:
                Z = np.sqrt((new_tensor**2).sum())
                new_tensor /= Z
                Z_list.append(Z)
            R,Q = Decomposition_Engine(new_tensor)[:2]
            Q   = Q.reshape((-1,*shape[1:]))
            new_chain.append(Q)
    new_chain=new_chain[::-1]
    return new_chain,Z_list

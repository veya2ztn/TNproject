import torch
import numpy as np
def truncated_SVD(tensor,output='RQ',max_singular_values= None,
                  max_truncation_error= None,
                  relative = True,
                  normlized= True,
                  verbose  = False,auto_check_diagonal=False):
    # the canonocal
    # tensor is batched
    reduce = False
    u=s=v = None
    if len(tensor.shape)==2:
        tensor = tensor.unsqueeze(0)
        reduce = True
    if auto_check_diagonal:
        out = diagonal_tensor_svd_torch_dense(tensor)
        if out is not None:u, s, v = out
    if u is None: u, s, v = torch.svd(tensor)


    max_singular_values = s.shape[-1] if max_singular_values is None else max_singular_values

    if max_truncation_error is not None:
        # Cumulative norms of singular values in ascending order
        s_sorted, _ = torch.sort(s**2,-1)
        trunc_errs  = torch.sqrt(torch.cumsum(s_sorted, -1))
        # If relative is true, rescale max_truncation error with the largest
        # singular value to yield the absolute maximal truncation error.
        abs_max_truncation_error = max_truncation_error * s[:,0:1] if relative else max_truncation_error
        # We must keep at least this many singular values to ensure the
        # truncation error is <= abs_max_truncation_error.
        num_sing_vals_err = torch.sum(trunc_errs > abs_max_truncation_error,-1).max()
        if max_singular_values>num_sing_vals_err and verbose:
            print(f"use {num_sing_vals_err}/{max_singular_values} sing vals")
    else:
        num_sing_vals_err  = max_singular_values

    num_sing_vals_keep = min(max_singular_values, num_sing_vals_err)


    #s_rest = s[...,num_sing_vals_keep:]
    u      = u[...,:num_sing_vals_keep]
    s      = s[...,:num_sing_vals_keep]
    v      = v[...,:num_sing_vals_keep]
    v      = torch.transpose(v, -1, -2)#vh

    if num_sing_vals_keep == s.shape[-1] and normlized:
        Z = 1.0*torch.ones(s.shape[0])
    else:
        Z = torch.sum(s**2,-1).sqrt()

    if output == 'RQ':
        R = torch.einsum('iab,ibc->iac',u ,torch.diag_embed(s))
        Q = v
        output = [R,Q,Z]
    elif output == 'QR':
        Q = u
        R = torch.einsum('iab,ibc->iac',torch.diag_embed(s),v)
        output = [Q,R,Z]
    else:
        output = [u,s,v,Z]
    if reduce:output = [t[0] for t in output]
    return output
def left_canonicalize_MPS(mps_line,Decomposition_Engine=torch.linalg.qr,
                          normlization =True):
    # for any not canonical mps line
    # the chain size (D,P,D)
    new_chain = []
    R         = None
    Z_list    = []# record the scale information for each unit.
    # for a perfect MPS state, we expect the norm for each tensor is 1.
    for i,tensor in enumerate(mps_line):
        if len(tensor.shape)==2:
            new_tensor = torch.einsum('ab,bd->ad',R,tensor) if R is not None else tensor
            shape      = new_tensor.shape
        elif len(tensor.shape)==3:
            new_tensor = torch.einsum('ab,bcd->acd',R,tensor) if R is not None else tensor
            shape      = new_tensor.shape
            a,b,c = shape
            new_tensor = new_tensor.reshape(a*b,c)
        else:
            raise NotImplementedError
        Z = torch.norm(new_tensor)
        Z_list.append(Z)
        new_tensor /= Z
        if i == len(mps_line) - 1:
            new_chain.append(new_tensor.reshape(*shape[:-1],-1))
        else:
            Q,R = Decomposition_Engine(new_tensor)[:2]
            Q   = Q.reshape(*shape[:-1],-1)
            new_chain.append(Q)

    return new_chain,Z_list
def right_canonicalize_MPS(mps_line,Decomposition_Engine=truncated_SVD,
                          #normlization =True
                          ):
    new_chain = []
    R         = None
    Z_list    = []
    svd_Z         = torch.Tensor([1.0])#input has already been normalized
    for i,tensor in enumerate(mps_line[::-1]):

        if len(tensor.shape)==2:
            new_tensor = torch.einsum('ab,bc->ac',tensor, R) if R is not None else tensor
            shape      = new_tensor.shape
        elif len(tensor.shape)==3:
            new_tensor = torch.einsum('alb,bc->alc',tensor, R) if R is not None else tensor
            shape      = new_tensor.shape
            a,b,c      = shape
            new_tensor = new_tensor.reshape(a,b*c)
        else:
            raise NotImplementedError
        Z = torch.norm(new_tensor)
        Z_list.append(Z)
        new_tensor /= Z
        # normlization is necessary; directly use the SVD_Z may cause problem due to precision
        #print(f"svd_Z{svd_Z.item()}<->all_Z {Z.item()} <-> after_Z{torch.norm(new_tensor).item()}")
        #if normlization:new_tensor /= Z
        if i == len(mps_line) - 1:
            new_chain.append(new_tensor.reshape(-1,*shape[1:]))
        else:
            R,Q = Decomposition_Engine(new_tensor)[:2]
            Q   = Q.reshape(-1,*shape[1:])
            new_chain.append(Q)
    new_chain=new_chain[::-1]
    return new_chain,Z_list
def torchrq(tensor):
    q, r = torch.qr(torch.transpose(tensor, -2, -1))
    r, q = torch.transpose(r, -2, -1), torch.transpose(q, -2, -1)  #M=r*q at this point
    return r,q
def approxmate_mps_line(mps_line,
                        max_singular_values= None,max_truncation_error= None,relative = True,
                        mode='full',left_method='qr'
                       ):


    scalar = 1
    if mode != 'right':
        if left_method == 'qr':
            DCEngine = torch.linalg.qr if float(torch.__version__[:4])>1.07 else torch.qr
        elif left_method == 'svd':
            DCEngine = lambda x:truncated_SVD(x,output='QR')
        else:
            raise NotImplementedError
        mps_line,Z_list = left_canonicalize_MPS(mps_line,Decomposition_Engine=DCEngine)
        #print(get_mps_size_list(mps_line))
        print(f"   left canonical scalar:{np.prod(Z_list)}")
        scalar *= np.prod(Z_list)
        #print(f"now tensor norm: {torch.norm(mps_line[-1])}")
    SVD_Engine = lambda x:truncated_SVD(x,max_singular_values = max_singular_values,
                                          max_truncation_error= max_truncation_error,
                                          relative = relative)
    mps_line,Z_list = right_canonicalize_MPS(mps_line,Decomposition_Engine=SVD_Engine)
    #print(get_mps_size_list(mps_line))
    #print(f"   right canonical Z:{[np.round(t.item(),3) for t in Z_list]}")
    print(f"   right canonical scalar:{np.prod(Z_list)}")
    scalar *= np.prod(Z_list)
    return mps_line,scalar
def diagonal_tensor_svd_torch_dense(tensor):
    # support batch tensor
    reduce = False
    if len(tensor.shape)==2:
        tensor = tensor.unsqueeze(0)
        reduce = True
    W,H   = tensor.shape[-2:]
    batch_shape= tensor.shape[:-2]
    u = s = v = None
    if W>=H:
        batch_diag = torch.matmul(tensor.transpose(-1,-2),tensor)#auto broadcast, or can use bmm
    else:
        batch_diag = torch.matmul(tensor,tensor.transpose(-1,-2))#auto broadcast, or can use bmm
    diagnol_num = torch.diagonal(a,dim1=-2,dim2=-1).nelement()
    tensor_num  = tensor.nelement()
    if diagnol_num != tensor_num:return None
    A,A        = batch_diag.shape[-2:]
    batch_diag = batch_diag[...,range(A),range(A)]
    batch_diag,batch_order= batch_diag.sort(-1,descending=True)
    #fast_V      = [get_sort_matrix(order).to_dense() for order in batch_order]
    batch_order = batch_order.flatten(start_dim=0,end_dim=-2)
    K,A         = batch_order.shape
    s           = batch_diag.sqrt()
    s           = s.reshape(*batch_shape,A)
    if W>=H:
        v = torch.sparse_coo_tensor([list(range(K*A)),batch_order.flatten().tolist()], [1.0]*K*A,(K*A,A))
        v = v.to_dense().reshape(-1,A,A)
        u = torch.bmm(tensor,v.transpose(-1,-2)/s.unsqueeze(-2))
        v = v.reshape(*batch_shape,A,A)
        u = u.reshape(*batch_shape,W,A)
    else:
        u = torch.sparse_coo_tensor([list(range(K*A)),batch_order.flatten().tolist()], [1.0]*K*A,(K*A,A))
        u = u.to_dense().reshape(-1,A,A).transpose(-1,-2)
        v = torch.bmm(u.transpose(-1,-2)/s.unsqueeze(-1),tensor)#TODO: case when s==0
        u = u.reshape(*batch_shape,A,A)
        v = v.reshape(*batch_shape,A,H)
    output = [u,s,v.transpose(-1,-2)]
    if reduce:
        output = [t[0] for t in output]
    return output

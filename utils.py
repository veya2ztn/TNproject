import torch
import numpy as np
def preprocess_sincos(x):
    n_data, C , dim0, dim1 = tuple(x.shape)
    n_sites = C * dim0 * dim1
    x = x.reshape((n_data, n_sites))
    return torch.stack([torch.cos(x*np.pi),torch.sin(x*np.pi)],-1)
def preprocess_binary(x):
    n_data, C , dim0, dim1 = tuple(x.shape)
    n_sites = C * dim0 * dim1
    x = x.reshape((n_data, n_sites))
    x = x.round()
    return torch.stack([1-x,x],-1)
def preprocess_sum_one(x):
    n_data, C , dim0, dim1 = tuple(x.shape)
    n_sites = C * dim0 * dim1
    x = x.reshape((n_data, n_sites))
    return torch.stack([1-x,x],-1)

def right_mps_form(mps_list):
    if (len(mps_list)==3 and len(mps_list[0].shape)+ 2 == len(mps_list[1].shape)):
        return [mps_list[0]]+list(*mps_list[1:-1])+[mps_list[-1]]
    else:
        return mps_list

def contract_two_mps(mps_list_1,mps_list_2):
    mps_list_1   = right_mps_form(mps_list_1)
    mps_list_2   = right_mps_form(mps_list_2)
    mps_nodes_1  = [tn.Node(v, name=f"t{i}") for i,v in enumerate(mps_list_1)]
    mps_edges_1  = [mps_nodes_1[i][-1]^mps_nodes_1[i+1][0] for i in range(len(mps_nodes_1)-1)]
    mps_nodes_2  = [tn.Node(v, name=f"t{i}") for i,v in enumerate(mps_list_2)]
    mps_edges_2  = [mps_nodes_2[i][-1]^mps_nodes_2[i+1][0] for i in range(len(mps_nodes_2)-1)]
    mps_edge     =([mps_nodes_1[0][0]^mps_nodes_2[0][0]]
                  +[mps_nodes_1[i][1]^mps_nodes_2[i][1] for i in range(1,len(mps_nodes_2))])
    return contractors.auto(mps_nodes_1+mps_nodes_2).tensor

def contract_mps_mpo(mps_list,mpo_list):
    assert len(mps_list) > 2
    assert len(mpo_list) > 2
    stack_unit_mps = (len(mps_list)==3 and len(mps_list[0].shape)+ 2 == len(mps_list[1].shape))
    stack_unit_mpo = (len(mpo_list)==3 and len(mpo_list[0].shape)+ 2 == len(mpo_list[1].shape))
    if stack_unit_mps and stack_unit_mpo:
        return contract_mps_mpo_uniform_mode(mps_list,mpo_list)
    else:
        if stack_unit_mps:mps_list = [mps_list[0]]+list(*mps_list[1:-1])+[mps_list[-1]]
        if stack_unit_mpo:mpo_list = [mpo_list[0]]+list(*mpo_list[1:-1])+[mpo_list[-1]]
        return contract_mps_mpo_ordinary_mode(mps_list,mpo_list)

def contract_mps_mpo_ordinary_mode(mps_list,mpo_list):
    # mps_list                                      P
    # (P,a)-(a,P,b)-(b,P,c)-...-(f,P,e)-(e,P)  --a--|--b--
    # mpo_list                                                P
    # (P,a,P)-(a,P,b,P)-(b,P,c,P)-...-(f,P,e,P)-(e,P,P) --a--|--b--
    #                                                       P
    assert len(mps_list) == len(mpo_list)
    new_mps_list= []
    tensor_left = torch.einsum("ab,cda->cbd",mps_list[ 0],mpo_list[ 0]).flatten(-2,-1)
    new_mps_list.append(tensor_left)
    for i in range(1,len(mps_list)-1):
        tensor_inne =torch.einsum("abc,defb->adefc",mps_list[i],mpo_list[i]).flatten(-5,-4).flatten(-2,-1)
        new_mps_list.append(tensor_inne)
    tensor_rigt = torch.einsum("ab,cdb->acd",mps_list[-1],mpo_list[-1]).flatten(-3,-2)
    new_mps_list.append(tensor_rigt)
    return new_mps_list

def contract_mps_mpo_uniform_mode(mps_list,mpo_list):
    # mps_list                                      P
    # (P,D)-(D,P,D)-(D,P,D)-...-(D,P,D)-(D,P)  --D--|--D--
    # mpo_list                                             P
    # (P,D,P)-(D,P,D,P)-(D,P,D,P)-...-(D,P,D,P)-(D,P,P) --D--|--D--
    #                                                     P
    assert len(mps_list) == len(mpo_list)
    mps_left,mps_inne,mps_rigt = mps_list
    mpo_left,mpo_inne,mpo_rigt = mpo_list
    tensor_left = torch.einsum("ab,cda->cbd",mps_left,mpo_left).flatten(-2,-1)
    tensor_rigt = torch.einsum("ab,cdb->acd",mps_rigt,mpo_rigt).flatten(-3,-2)
    tensor_inne = torch.einsum("kabc,kdefb->kadefc",mps_inne,mpo_inne).flatten(-5,-4).flatten(-2,-1)
    return tensor_left,tensor_inne,tensor_rigt

def get_mps_size_list(mps):
    if len(mps)==3 and len(mps[1].shape)==4:
        return f"{tuple(mps[0].shape)} - {mps[1].shape[0]}x{tuple(mps[1][0].shape)} - {tuple(mps[-1].shape)}"
    last_shape = mps[0].shape
    count      = 1
    out_str    = ""
    for t in mps[1:]:
        now_shape = t.shape
        if last_shape == now_shape:
            count+=1
        else:
            if count >1 :
                out_str += f" {count}x{tuple(last_shape)} "+'-'
            else:
                out_str += f"{tuple(last_shape)}"+'-'
            count =1
        last_shape = now_shape
    if count >1 :
        out_str += f" {count}x{tuple(last_shape)} "+'-'
    else:
        out_str += f"{tuple(last_shape)}"+'-'
    return out_str.strip("-")

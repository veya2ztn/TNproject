import torch
import tensornetwork as tn
import numpy as np
from tensornetwork import contractors
tn.set_default_backend("pytorch")

def rde(*x,**kargs):
    x =  torch.randn(*x,device='cuda',**kargs)
    x/=  torch.norm(x).sqrt()
    return x

def rde_g(*x,**kargs):
    x =  torch.randn(*x,device='cuda',**kargs)
    x/=  torch.norm(x).sqrt()
    x = torch.autograd.Variable(x,requires_grad=True)
    return x

def create_one_eps(pd = 3,vd = 2,L  = 3,rde=rde):

    tensor = ([[rde(pd,vd,vd)   ]+[rde(pd,vd,vd,vd)    for _ in range(L-2)]+[rde(pd,vd,vd)  ]]+
              [[rde(pd,vd,vd,vd)]+[rde(pd,vd,vd,vd,vd) for _ in range(L-2)]+[rde(pd,vd,vd,vd)] for _ in range(L-2)]+
              [[rde(pd,vd,vd)   ] +[rde(pd,vd,vd,vd)     for _ in range(L-2)]+[rde(pd,vd,vd) ]])
    return tensor

def finite_tn_peps(tensor,**kargs):
    node_array = []
    W = len(tensor)
    H = len(tensor[0])
    for i in range(W):
        node_line = []
        for j in range(H):
            node = tn.Node(tensor[i][j],name=f"{i}-{j}")
            node_line.append(node)
        node_array.append(node_line)

    for row in range(W):
        for col in range(H-1):
            tn.connect(node_array[row][col][2],node_array[row][col+1][-1],f"{row},{col}<->{row},{col+1}")

    col = 0
    for row in range(W-2):
        tn.connect(node_array[row][col][1],node_array[row+1][col][3],f"{row},{col}<->{row+1},{col}")
    for col in range(1,H):
        for row in range(W-2):
            tn.connect(node_array[row][col][1],node_array[row+1][col][-2],f"{row},{col}<->{row+1},{col}")

    row = -1
    for col in range(H):
        tn.connect(node_array[row][col][1],node_array[row-1][col][1],f"{row},{col}<->{row-1},{col}")

    node_list = [item for sublist in node_array for item in sublist]
    return node_list

def contract_two_peps(tensor_1,tensor_2,**kargs):
    fpeps_1 = finite_tn_peps(tensor_1)
    fpeps_2 = finite_tn_peps(tensor_2)
    for n1,n2 in zip(fpeps_1,fpeps_2):n1[0]^n2[0]
    return contractors.auto(fpeps_1+fpeps_2,**kargs).tensor


from torch.utils.checkpoint import checkpoint
tensor_core  = create_one_eps(pd = 3,vd = 4,L  = 9,rde=rde_g)
tensor_next  = create_one_eps(pd = 3,vd = 6,L  = 9,rde=rde)
def contracte(*tensor_core):
    ans  = contract_two_peps(tensor_core,tensor_next)
    return ans
ans  = checkpoint(contracte,*tensor_core)
ans.backward()
print(tensor_core[0][0].grad)

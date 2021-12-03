# the input tensor network should be store in (W,H,a,a,a,a)
# for example, (8,8,2,2,2,2)
# or in center (W-2,H-2,a,a,a,a)
#       corner (4,a,a,a,a)
#       edge   (W,2,a,a,a,a)
#       edge   (2,H,a,a,a,a)

import torch
from Contractor2D.utils import apply_SVD

import tensornetwork as tn
from tensornetwork import contractors
tn.set_default_backend("pytorch")
def contractor_engine(node_list,**kargs):
    return contractors.auto(node_list,**kargs)[0].tensor
def uniform_shape_tensor_contractor_tn(tensor,**kargs):
    node_array = []
    W,H = tensor.shape[:2]
    for i in range(W):
        node_line = []
        for j in range(H):
            node = tn.Node(tensor[i][j],name=f"{i}-{j}")
            node_line.append(node)
        node_array.append(node_line)

    for i in range(W):
        for j in range(H):
            if j==H-1:node_array[i][j][2]^node_array[i  ][0  ][0]
            else:     node_array[i][j][2]^node_array[i  ][j+1][0]
            if i==W-1:node_array[i][j][3]^node_array[0  ][j  ][1]
            else:     node_array[i][j][3]^node_array[i+1][j  ][1]

    node_list = [item for sublist in node_array for item in sublist]
    return contractor_engine(node_list,**kargs)
def bulk_right_left_corner_contractor_tn(bulk_tensor,right_edge,down_edge,corn_tensor):
    node_array = []
    if len(bulk_tensor)==0:
        W=H=0
    else:
        W,H = bulk_tensor.shape[:2]
    W  += 1
    H  += 1
    for i in range(W):
        node_line = []
        for j in range(H):
            if (i==W-1) and (j==H-1):
                t = corn_tensor
            elif i==W-1:
                t = right_edge[j]
            elif j==H-1:
                t = down_edge[i]
            else:
                t = bulk_tensor[i][j]
            node = tn.Node(t,name=f"{i}-{j}")
            node_line.append(node)
        node_array.append(node_line)

    for i in range(W):
        for j in range(H):
            if j==H-1:node_array[i][j][2]^node_array[i  ][0  ][0]
            else:     node_array[i][j][2]^node_array[i  ][j+1][0]
            if i==W-1:node_array[i][j][3]^node_array[0  ][j  ][1]
            else:     node_array[i][j][3]^node_array[i+1][j  ][1]

    node_list = [item for sublist in node_array for item in sublist]
    return contractor_engine(node_list)


def naive_contractor(tensor):
    # only work for (W,H,2^n,2^n,2^n,2^n) tensor
    assert tensor.shape[-1] in [2,4,8,16,32]
    while tensor.shape[0]>2:
        w,h = tensor.shape[:2]
        half_w = w // 2
        half_h = h // 2
        end_w  = 2 * half_w
        end_h  = 2 * half_h
        lu_tensor    = tensor[0:end_w:2,0:end_h:2]
        ld_tensor    = tensor[0:end_w:2,1:end_h:2]
        ru_tensor    = tensor[1:end_w:2,0:end_h:2]
        rd_tensor    = tensor[1:end_w:2,1:end_h:2]
        tensor     = torch.einsum("xyabcd,xyhdij,xycefg,xyigkl->xyahbefkjl",
                                lu_tensor,
                                ld_tensor,
                                ru_tensor,
                                rd_tensor).flatten(4,5).flatten(-4,-3).flatten(2,3).flatten(-2,-1)
        # remain_w_up  = tensor[end_w:,0::2]
        # remain_w_dw  = tensor[end_w:,1::2]
        # tensor     = torch.einsum("xyabcd,xyhdij,xycefg,xyigkl->xyahbefkjl",
        #                         lu_tensor,
        #                         ld_tensor,
        #                         ru_tensor,
        #                         rd_tensor).flatten(4,5).flatten(-4,-3).flatten(2,3).flatten(-2,-1)
        #
    return tensor

def TRG_step(tensor,mode='auto'):
    if isinstance(tensor,list):
        bulk_tensor,right_edge,down_edge,corn_tensor = tensor
    else:
        W,H = tensor.shape[:2]
        if W%2==0 and H%2==2 and mode=='auto':
            return TRG_step_even_number(tensor)
        else:
            bulk_tensor = tensor
            if W%2==1:
                bulk_tensor= bulk_tensor[:-2]
                right_edge1,right_edge2 = bulk_tensor[-2:]
                right_edge = torch.einsum("wabcd,wedfg->waebcfg",right_edge1,right_edge2).flatten(1,2).flatten(-3,-2)
            else:
                bulk_tensor= bulk_tensor[:-1]
                right_edge = bulk_tensor[-1:]

            if H%2==1:
                bulk_tensor= bulk_tensor[:,:-2]
                down_edge1, down_edge2 = bulk_tensor[:,-2:].transpose(1,0)
                down_edge  = torch.einsum("wabcd,wcefg->wabefdg",down_edge1,down_edge2).flatten(2,3).flatten(-2,-1)
                corn_tensor1,corn_tensor2= right_edge[-2:]
                corn_tensor= torch.einsum("abcd,cefg->abefdg",corn_tensor1,corn_tensor2).flatten(1,2).flatten(-2,-1)
                right_edge = right_edge[:-2]
            else:
                bulk_tensor= bulk_tensor[:,:-1]
                down_edge  = bulk_tensor[:,-1:].transpose(1,0)
                right_edge = right_edge[:-1]
                corn_tensor= right_edge[-1]    
    return TRG_step_brdc(bulk_tensor,right_edge,down_edge,corn_tensor)
def TRG_step_even_number(tensor):
    lu_tensor = tensor[0::2,0::2]
    ld_tensor = tensor[0::2,1::2]
    ru_tensor = tensor[1::2,0::2]
    rd_tensor = tensor[1::2,1::2]
    lu_lu,lu_rd = apply_SVD(lu_tensor,left_bond=[0,1],right_bond=[2,3],truncate=truncate)# abk ,kcd
    rd_lu,rd_rd = apply_SVD(rd_tensor,left_bond=[0,1],right_bond=[2,3],truncate=truncate)# abk, kcd
    ld_ld,ld_ru = apply_SVD(ld_tensor,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# bck, kda
    ru_ld,ru_ru = apply_SVD(ru_tensor,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# bck, kda
    tensor1     = torch.einsum("whicd,whjac,whbak,whdbl->whijkl",
                                lu_rd,
                                ld_ru,
                                rd_lu,
                                ru_ld)
    tensor2     = torch.einsum("whicd,whjac,whbak,whdbl->whijkl",
                               rd_rd,
                               ru_ru.roll(-1,1),
                               lu_lu.roll(shifts=(-1, -1), dims=(0, 1)),
                               ld_ld.roll(-1,0))
    left,right  = apply_SVD(tensor1,left_bond=[0,1],right_bond=[2,3],truncate=truncate)# ABK,KCD
    lower,uppe  = apply_SVD(tensor2,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# BCK,KDA
    new_tensor  = torch.einsum("whadi,whjba,whkcb,whdcl->whijkl",
                               lower,
                               right.roll(-1,1),
                               uppe.roll(-1,1),
                               left.roll(shifts=(-1, -1), dims=(0, 1))
                               )
    return new_tensor

def TRG_step_brdc(bulk_tensor,right_edge,down_edge,corn_tensor):
    truncate = None
    lu_lu,lu_rd     = apply_SVD(bulk_tensor[0::2,0::2],left_bond=[0,1],right_bond=[2,3],truncate=None)# abk ,kcd
    rd_lu,rd_rd     = apply_SVD(bulk_tensor[1::2,1::2],left_bond=[0,1],right_bond=[2,3],truncate=None)# abk, kcd
    ld_ld,ld_ru     = apply_SVD(bulk_tensor[0::2,1::2],left_bond=[1,2],right_bond=[3,0],truncate=None)# bck, kda
    ru_ld,ru_ru     = apply_SVD(bulk_tensor[1::2,0::2],left_bond=[1,2],right_bond=[3,0],truncate=None)# bck, kda
    rd_lu_r,rd_rd_r = apply_SVD(right_edge[1::2],left_bond=[0,1],right_bond=[2,3],truncate=None)# abk, kcd
    ru_ld_r,ru_ru_r = apply_SVD(right_edge[0::2],left_bond=[1,2],right_bond=[3,0],truncate=None)# bck, kda
    rd_lu_d,rd_rd_d = apply_SVD( down_edge[1::2],left_bond=[0,1],right_bond=[2,3],truncate=None)# abk, kcd
    ld_ld_d,ld_ru_d = apply_SVD( down_edge[0::2],left_bond=[1,2],right_bond=[3,0],truncate=None)# bck, kda
    rd_lu_c,rd_rd_c = apply_SVD(corn_tensor     ,left_bond=[0,1],right_bond=[2,3],truncate=None)# bck, kda
    bulk_tensor1    = torch.einsum("whicd,whjac,whbak,whdbl->whijkl",lu_rd[:-1,:-1],ld_ru[:-1],rd_lu,ru_ld[:,:-1])
    rigt_tensor1    = torch.einsum("wicd,wjac,wbak,wdbl->wijkl",lu_rd[-1,:-1],ld_ru[-1]   ,rd_lu_r, ru_ld_r[:-1])
    down_tensor1    = torch.einsum("wicd,wjac,wbak,wdbl->wijkl",lu_rd[:-1,-1],ld_ru_d[:-1],rd_lu_d, ru_ld[:,-1])
    corn_tensor1    = torch.einsum(" icd, jac, bak, dbl-> ijkl",lu_rd[-1,-1] ,ld_ru_d[-1] ,rd_lu_c, ru_ld_r[-1])
    bulk_tensor2    = torch.einsum("whicd,whjac,whbak,whdbl->whijkl",rd_rd,ru_ru[:,1:],lu_lu[1:,1:],ld_ld[1:])
    rigt_tensor2    = torch.einsum("wicd,wjac,wbak,wdbl->wijkl",rd_rd_r, ru_ru_r[1:],lu_lu[0,1:], ld_ld[0])
    down_tensor2    = torch.einsum("wicd,wjac,wbak,wdbl->wijkl",rd_rd_d, ru_ru[:,0],lu_lu[1:,0], ld_ld_d[1:])
    corn_tensor2    = torch.einsum(" icd, jac, bak, dbl-> ijkl",rd_rd_c, ru_ru_r[0],lu_lu[0,0] , ld_ld_d[0])
    bulk_left ,bulk_right = apply_SVD(bulk_tensor1,left_bond=[0,1],right_bond=[2,3],truncate=truncate)
    rigt_left ,rigt_right = apply_SVD(rigt_tensor1,left_bond=[0,1],right_bond=[2,3],truncate=truncate)
    down_left ,down_right = apply_SVD(down_tensor1,left_bond=[0,1],right_bond=[2,3],truncate=truncate)
    corn_left ,corn_right = apply_SVD(corn_tensor1,left_bond=[0,1],right_bond=[2,3],truncate=truncate)
    bulk_lower,bulk_uppe  = apply_SVD(bulk_tensor2,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# BCK,KDA
    rigt_lower,rigt_uppe  = apply_SVD(rigt_tensor2,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# BCK,KDA
    down_lower,down_uppe  = apply_SVD(down_tensor2,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# BCK,KDA
    corn_lower,corn_uppe  = apply_SVD(corn_tensor2,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# BCK,KDA

    bulk_tensor           = torch.einsum("whadi,whjba,whkcb,whdcl->whijkl", bulk_lower[:-1,:-1],bulk_right[:-1,1:],bulk_uppe[:-1,1:],bulk_left[1:,1:])
    rigt_bulk_tensor      = torch.einsum("hadi,hjba,hkcb,hdcl->hijkl"     , rigt_lower[:-1]    ,rigt_right[1:]    ,rigt_uppe[1:]    ,bulk_left[0,1:])
    down_bulk_tensor      = torch.einsum("hadi,hjba,hkcb,hdcl->hijkl"     , down_lower[:-1]    ,bulk_right[:-1,0] ,bulk_uppe[:-1,0] ,bulk_left[1:,0])
    bulk_rigt_tensor      = torch.einsum("hadi,hjba,hkcb,hdcl->hijkl"     , bulk_lower[-1,:-1] ,bulk_right[-1,1:] ,bulk_uppe[-1,1:] ,rigt_left[1:])
    bulk_down_tensor      = torch.einsum("hadi,hjba,hkcb,hdcl->hijkl"     , bulk_lower[:-1,-1] ,down_right[:-1]   ,down_uppe[:-1]   ,down_left[1:])
    bulk_down_corn_tensor = torch.einsum("adi,jba,kcb,dcl->ijkl"          , bulk_lower[-1,-1]  ,down_right[-1]    ,down_uppe[-1]    ,corn_left)
    corn_right_bulk_tensor= torch.einsum("adi,jba,kcb,dcl->ijkl"          , corn_lower         ,rigt_right[0]     ,rigt_uppe[0]     ,bulk_left[0,0])
    right_corn_down_tensor= torch.einsum("adi,jba,kcb,dcl->ijkl"          , rigt_lower[-1]     ,corn_right        ,corn_uppe        ,down_left[0])
    down_bulk_right_tensor= torch.einsum("adi,jba,kcb,dcl->ijkl"          , down_lower[-1]     ,bulk_right[-1,0]  ,bulk_uppe[-1,0]  ,rigt_left[0])

    if len(bulk_tensor_cent)==0:
        bulk_tensor = down_bulk_right_tensor[None][None]
    else:
        bulk_tensor = torch.cat([
            torch.cat([bulk_tensor_cent,down_bulk_tensor[None]],1),
            torch.cat([bulk_rigt_tensor,down_bulk_right_tensor[None]])[None]
        ])


    right_edge = torch.cat([rigt_bulk_tensor,corn_right_bulk_tensor[None]])
    down_edge  = torch.cat([bulk_down_tensor,bulk_down_corn_tensor[None]])
    corn_tensor =right_corn_down_tensor
    return bulk_tensor,right_edge,down_edge,corn_tensor

import torch

from Contractor2D.utils import apply_SVD

tensor = torch.randn(5,5,2,2,2,2)
print(f"the input tensor network store in {tensor.shape}")
print(f"the ultimate result should be {uniform_shape_tensor_contractor_tn(tensor)}")


right_edge1,right_edge2 = tensor[-2:]
right_edge = torch.einsum("wabcd,wedfg->waebcfg",right_edge1,right_edge2).flatten(1,2).flatten(-3,-2)
down_edge1, down_edge2 = tensor[:-2,-2:].transpose(1,0)
corn_tensor1,corn_tensor2= right_edge[-2:]
bulk_tensor= tensor[:-2,:-2]
right_edge = right_edge[:-2]
down_edge  = torch.einsum("wabcd,wcefg->wabefdg",down_edge1,down_edge2).flatten(2,3).flatten(-2,-1)
corn_tensor= torch.einsum("abcd,cefg->abefdg",corn_tensor1,corn_tensor2).flatten(1,2).flatten(-2,-1)

print(f"the input bulk_right_left_corner tensor :")
print(f"bulk_tensor.shape = {bulk_tensor.shape}")
print(f"right_edge.shape  = {right_edge.shape }")
print(f"down_edge.shape   = {down_edge.shape  }")
print(f"corn_tensor.shape = {corn_tensor.shape}")

print(f"the bulk_right_left_corner_contractor result {bulk_right_left_corner_contractor_tn(bulk_tensor,right_edge,down_edge,corn_tensor)}")

lu_lu,lu_rd = apply_SVD(bulk_tensor[0::2,0::2],left_bond=[0,1],right_bond=[2,3],truncate=None)# abk ,kcd
rd_lu,rd_rd = apply_SVD(bulk_tensor[1::2,1::2],left_bond=[0,1],right_bond=[2,3],truncate=None)# abk, kcd
ld_ld,ld_ru = apply_SVD(bulk_tensor[0::2,1::2],left_bond=[1,2],right_bond=[3,0],truncate=None)# bck, kda
ru_ld,ru_ru = apply_SVD(bulk_tensor[1::2,0::2],left_bond=[1,2],right_bond=[3,0],truncate=None)# bck, kda

rd_lu_r,rd_rd_r = apply_SVD(right_edge[1::2],left_bond=[0,1],right_bond=[2,3],truncate=None)# abk, kcd
ru_ld_r,ru_ru_r = apply_SVD(right_edge[0::2],left_bond=[1,2],right_bond=[3,0],truncate=None)# bck, kda
rd_lu_d,rd_rd_d = apply_SVD( down_edge[1::2],left_bond=[0,1],right_bond=[2,3],truncate=None)# abk, kcd
ld_ld_d,ld_ru_d = apply_SVD( down_edge[0::2],left_bond=[1,2],right_bond=[3,0],truncate=None)# bck, kda
rd_lu_c,rd_rd_c = apply_SVD(corn_tensor     ,left_bond=[0,1],right_bond=[2,3],truncate=None)# bck, kda

bulk_tensor1 = torch.einsum("whicd,whjac,whbak,whdbl->whijkl",lu_rd[:-1,:-1],ld_ru[:-1],rd_lu,ru_ld[:,:-1])
rigt_tensor1 = torch.einsum("wicd,wjac,wbak,wdbl->wijkl",lu_rd[-1,:-1],ld_ru[-1]   ,rd_lu_r, ru_ld_r[:-1])
down_tensor1 = torch.einsum("wicd,wjac,wbak,wdbl->wijkl",lu_rd[:-1,-1],ld_ru_d[:-1],rd_lu_d, ru_ld[:,-1])
corn_tensor1 = torch.einsum(" icd, jac, bak, dbl-> ijkl",lu_rd[-1,-1] ,ld_ru_d[-1] ,rd_lu_c, ru_ld_r[-1])
bulk_tensor2 = torch.einsum("whicd,whjac,whbak,whdbl->whijkl",rd_rd,ru_ru[:,1:],lu_lu[1:,1:],ld_ld[1:])
rigt_tensor2 = torch.einsum("wicd,wjac,wbak,wdbl->wijkl",rd_rd_r, ru_ru_r[1:],lu_lu[0,1:], ld_ld[0])
down_tensor2 = torch.einsum("wicd,wjac,wbak,wdbl->wijkl",rd_rd_d, ru_ru[:,0],lu_lu[1:,0], ld_ld_d[1:])
corn_tensor2 = torch.einsum(" icd, jac, bak, dbl-> ijkl",rd_rd_c, ru_ru_r[0],lu_lu[0,0] , ld_ld_d[0])

print("phase1:result")
print(f"bulk_tensor1.shape={bulk_tensor1.shape}")
print(f"rigt_tensor1.shape={rigt_tensor1.shape}")
print(f"down_tensor1.shape={down_tensor1.shape}")
print(f"corn_tensor1.shape={corn_tensor1.shape}")
print()
print(f"bulk_tensor2.shape={bulk_tensor2.shape}")
print(f"rigt_tensor2.shape={rigt_tensor2.shape}")
print(f"down_tensor2.shape={down_tensor2.shape}")
print(f"corn_tensor2.shape={corn_tensor2.shape}")

# if we don't truncate at this step, the map will become too complex and can not code as unique way,
# so in such case, we will require truncate 16 in here and the recommend input not big than 32.
truncate = None
bulk_left ,bulk_right= apply_SVD(bulk_tensor1,left_bond=[0,1],right_bond=[2,3],truncate=truncate)
rigt_left ,rigt_right= apply_SVD(rigt_tensor1,left_bond=[0,1],right_bond=[2,3],truncate=truncate)
down_left ,down_right= apply_SVD(down_tensor1,left_bond=[0,1],right_bond=[2,3],truncate=truncate)
corn_left ,corn_right= apply_SVD(corn_tensor1,left_bond=[0,1],right_bond=[2,3],truncate=truncate)
bulk_lower,bulk_uppe = apply_SVD(bulk_tensor2,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# BCK,KDA
rigt_lower,rigt_uppe = apply_SVD(rigt_tensor2,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# BCK,KDA
down_lower,down_uppe = apply_SVD(down_tensor2,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# BCK,KDA
corn_lower,corn_uppe = apply_SVD(corn_tensor2,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# BCK,KDA

bulk_tensor_cent      = torch.einsum("whadi,whjba,whkcb,whdcl->whijkl", bulk_lower[:-1,:-1],bulk_right[:-1,1:],bulk_uppe[:-1,1:],bulk_left[1:,1:])
rigt_bulk_tensor      = torch.einsum("hadi,hjba,hkcb,hdcl->hijkl"     , rigt_lower[:-1]    ,rigt_right[1:]    ,rigt_uppe[1:]    ,bulk_left[0,1:])
down_bulk_tensor      = torch.einsum("hadi,hjba,hkcb,hdcl->hijkl"     , down_lower[:-1]    ,bulk_right[:-1,0] ,bulk_uppe[:-1,0] ,bulk_left[1:,0])
bulk_rigt_tensor      = torch.einsum("hadi,hjba,hkcb,hdcl->hijkl"     , bulk_lower[-1,:-1] ,bulk_right[-1,1:] ,bulk_uppe[-1,1:] ,rigt_left[1:])
bulk_down_tensor      = torch.einsum("hadi,hjba,hkcb,hdcl->hijkl"     , bulk_lower[:-1,-1] ,down_right[:-1]   ,down_uppe[:-1]   ,down_left[1:])
bulk_down_corn_tensor = torch.einsum("adi,jba,kcb,dcl->ijkl"          , bulk_lower[-1,-1]  ,down_right[-1]    ,down_uppe[-1]    ,corn_left)
corn_right_bulk_tensor= torch.einsum("adi,jba,kcb,dcl->ijkl"          , corn_lower         ,rigt_right[0]     ,rigt_uppe[0]     ,bulk_left[0,0])
right_corn_down_tensor= torch.einsum("adi,jba,kcb,dcl->ijkl"          , rigt_lower[-1]     ,corn_right        ,corn_uppe        ,down_left[0])
down_bulk_right_tensor= torch.einsum("adi,jba,kcb,dcl->ijkl"          , down_lower[-1]     ,bulk_right[-1,0]  ,bulk_uppe[-1,0]  ,rigt_left[0])
# down_bulk_tensor       = down_bulk_tensor[None]
# down_bulk_right_tensor = down_bulk_right_tensor[None]
# corn_right_bulk_tensor = corn_right_bulk_tensor[None]
# bulk_down_corn_tensor  = bulk_down_corn_tensor[None]

print("phase2 result:")
print(f"bulk_tensor_cent.shape      ={bulk_tensor_cent.shape      }")
print(f"down_bulk_tensor.shape      ={down_bulk_tensor.shape      }")      
print(f"bulk_rigt_tensor.shape      ={bulk_rigt_tensor.shape      }")   
print(f"down_bulk_right_tensor.shape={down_bulk_right_tensor.shape}")
print(f"rigt_bulk_tensor.shape      ={rigt_bulk_tensor.shape      }")      
print(f"corn_right_bulk_tensor.shape={corn_right_bulk_tensor.shape}")
print(f"bulk_down_tensor.shape      ={bulk_down_tensor.shape      }") 
print(f"bulk_down_corn_tensor.shape ={bulk_down_corn_tensor.shape }") 
print(f"right_corn_down_tensor.shape={right_corn_down_tensor.shape}")

bulk_tensor = torch.cat([
    torch.cat([bulk_tensor_cent,down_bulk_tensor[None]],1),
    torch.cat([bulk_rigt_tensor,down_bulk_right_tensor[None]])[None]
])

bulk_tensor.shape

bulk_tensor = down_bulk_right_tensor[None][None]

right_edge = torch.cat([rigt_bulk_tensor,corn_right_bulk_tensor[None]])
down_edge  = torch.cat([bulk_down_tensor,bulk_down_corn_tensor[None]])
corn_tensor =right_corn_down_tensor

print(f"next tensor shape:")
print(f"bulk_tensor.shape = {bulk_tensor.shape}")
print(f"right_edge.shape  = {right_edge.shape }")
print(f"down_edge.shape   = {down_edge.shape  }")
print(f"corn_tensor.shape = {corn_tensor.shape}")

print(f"next tensor contraction result:")
print(bulk_right_left_corner_contractor_tn(bulk_tensor,right_edge,down_edge,corn_tensor))

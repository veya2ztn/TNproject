import torch
from Contractor2D.utils import apply_SVD

tensor     = torch.randn(8,8,2,2,2,2)/2
uniform_shape_tensor_contractor_tn(tensor)

lu_lu_origin,lu_rd_origin = apply_SVD(tensor[0::2,0::2],left_bond=[0,1],right_bond=[2,3],truncate=None)# abk ,kcd
rd_lu_origin,rd_rd_origin = apply_SVD(tensor[1::2,1::2],left_bond=[0,1],right_bond=[2,3],truncate=None)# abk, kcd
ld_ld_origin,ld_ru_origin = apply_SVD(tensor[0::2,1::2],left_bond=[1,2],right_bond=[3,0],truncate=None)# bck, kda
ru_ld_origin,ru_ru_origin = apply_SVD(tensor[1::2,0::2],left_bond=[1,2],right_bond=[3,0],truncate=None)# bck, kda

lu_lu,lu_rd = apply_SVD(bulk_tensor[0::2,0::2],left_bond=[0,1],right_bond=[2,3],truncate=None)# abk ,kcd
rd_lu,rd_rd = apply_SVD(bulk_tensor[1::2,1::2],left_bond=[0,1],right_bond=[2,3],truncate=None)# abk, kcd
ld_ld,ld_ru = apply_SVD(bulk_tensor[0::2,1::2],left_bond=[1,2],right_bond=[3,0],truncate=None)# bck, kda
ru_ld,ru_ru = apply_SVD(bulk_tensor[1::2,0::2],left_bond=[1,2],right_bond=[3,0],truncate=None)# bck, kda

print(lu_lu.shape)
print(rd_lu.shape)
print(ld_ld.shape)
print(ru_ld.shape)

lu_lu_should = lu_lu_origin[:,:]    ;lu_rd_should =lu_rd_origin[:,:]
rd_lu_should = rd_lu_origin[:-1,:-1];rd_rd_should =rd_rd_origin[:-1,:-1]
ld_ld_should = ld_ld_origin[:,:-1]  ;ld_ru_should =ld_ru_origin[:,:-1]
ru_ld_should = ru_ld_origin[:-1,:]  ;ru_ru_should =ru_ru_origin[:-1,:]

print(lu_lu_should.shape)
print(rd_lu_should.shape)
print(ld_ld_should.shape)
print(ru_ld_should.shape)

print(torch.dist(lu_lu_should,lu_lu),torch.dist(lu_rd_should,lu_rd))
print(torch.dist(rd_lu_should,rd_lu),torch.dist(rd_rd_should,rd_rd))
print(torch.dist(ld_ld_should,ld_ld),torch.dist(ld_ru_should,ld_ru))
print(torch.dist(ru_ld_should,ru_ld),torch.dist(ru_ru_should,ru_ru))

# right_edge == bulk_tensor[-1]
# down_edge  == bulk_tensor[:,-1]

rd_lu_r,rd_rd_r = apply_SVD(right_edge[1::2],left_bond=[0,1],right_bond=[2,3],truncate=None)# abk, kcd
ru_ld_r,ru_ru_r = apply_SVD(right_edge[0::2],left_bond=[1,2],right_bond=[3,0],truncate=None)# bck, kda
rd_lu_d,rd_rd_d = apply_SVD( down_edge[1::2],left_bond=[0,1],right_bond=[2,3],truncate=None)# abk, kcd
ld_ld_d,ld_ru_d = apply_SVD( down_edge[0::2],left_bond=[1,2],right_bond=[3,0],truncate=None)# bck, kda
rd_lu_c,rd_rd_c = apply_SVD(corn_tensor     ,left_bond=[0,1],right_bond=[2,3],truncate=None)# bck, kda

rd_lu_r_should=rd_lu_origin[-1,:-1];rd_rd_r_should=rd_rd_origin[-1,:-1]
ru_ld_r_should=ru_ld_origin[-1,:]  ;ru_ru_r_should=ru_ru_origin[-1,:]
rd_lu_d_should=rd_lu_origin[:-1,-1];rd_rd_d_should=rd_rd_origin[:-1,-1]
ld_ld_d_should=ld_ld_origin[:,-1]  ;ld_ru_d_should=ld_ru_origin[:,-1]
rd_lu_c_should=rd_lu_origin[ -1,-1];rd_rd_c_should=rd_rd_origin[ -1,-1]

print(rd_lu_r_should.shape)
print(ru_ld_r_should.shape)
print(rd_lu_d_should.shape)
print(ld_ld_d_should.shape)
print(rd_lu_c_should.shape)

print(torch.dist(rd_lu_r_should,rd_lu_r),torch.dist(rd_rd_r_should,rd_rd_r))
print(torch.dist(ru_ld_r_should,ru_ld_r),torch.dist(ru_ru_r_should,ru_ru_r))
print(torch.dist(rd_lu_d_should,rd_lu_d),torch.dist(rd_rd_d_should,rd_rd_d))
print(torch.dist(ld_ld_d_should,ld_ld_d),torch.dist(ld_ru_d_should,ld_ru_d))
print(torch.dist(rd_lu_c_should,rd_lu_c),torch.dist(rd_rd_c_should,rd_rd_c))

tensor1     = torch.einsum("whicd,whjac,whbak,whdbl->whijkl",
                                    lu_rd_origin,
                                    ld_ru_origin,
                                    rd_lu_origin,
                                    ru_ld_origin)
tensor2     = torch.einsum("whicd,whjac,whbak,whdbl->whijkl",
                           rd_rd_origin,
                           ru_ru_origin.roll(shifts=-1,dims=1),
                           lu_lu_origin.roll(shifts=(-1, -1), dims=(0, 1)),
                           ld_ld_origin.roll(shifts=-1,dims=0))

bulk_tensor1_should= tensor1[:-1,:-1]
rigt_tensor1_should= tensor1[-1,:-1]
down_tensor1_should= tensor1[:-1,-1]
corn_tensor1_should= tensor1[-1,-1]
bulk_tensor2_should= tensor2[:-1,:-1]
rigt_tensor2_should= tensor2[-1,:-1]
down_tensor2_should= tensor2[:-1,-1]
corn_tensor2_should= tensor2[-1,-1]

print(bulk_tensor1_should.shape)
print(rigt_tensor1_should.shape)  
print(down_tensor1_should.shape)  
print(corn_tensor1_should.shape)  
print(bulk_tensor2_should.shape)  
print(rigt_tensor2_should.shape)  
print(down_tensor2_should.shape)  
print(corn_tensor2_should.shape)  

bulk_tensor1 = torch.einsum("whicd,whjac,whbak,whdbl->whijkl",lu_rd[:-1,:-1],ld_ru[:-1],rd_lu,ru_ld[:,:-1])
rigt_tensor1 = torch.einsum("wicd,wjac,wbak,wdbl->wijkl",lu_rd[-1,:-1],ld_ru[-1]   ,rd_lu_r, ru_ld_r[:-1])
down_tensor1 = torch.einsum("wicd,wjac,wbak,wdbl->wijkl",lu_rd[:-1,-1],ld_ru_d[:-1],rd_lu_d, ru_ld[:,-1])
corn_tensor1 = torch.einsum(" icd, jac, bak, dbl-> ijkl",lu_rd[-1,-1] ,ld_ru_d[-1] ,rd_lu_c, ru_ld_r[-1])
bulk_tensor2 = torch.einsum("whicd,whjac,whbak,whdbl->whijkl",rd_rd,ru_ru[:,1:],lu_lu[1:,1:],ld_ld[1:])
rigt_tensor2 = torch.einsum("wicd,wjac,wbak,wdbl->wijkl",rd_rd_r, ru_ru_r[1:],lu_lu[0,1:], ld_ld[0])
down_tensor2 = torch.einsum("wicd,wjac,wbak,wdbl->wijkl",rd_rd_d, ru_ru[:,0],lu_lu[1:,0], ld_ld_d[1:])
corn_tensor2 = torch.einsum(" icd, jac, bak, dbl-> ijkl",rd_rd_c, ru_ru_r[0],lu_lu[0,0] , ld_ld_d[0])

print(bulk_tensor1.shape)
print(rigt_tensor1.shape)  
print(down_tensor1.shape)  
print(corn_tensor1.shape)  
print(bulk_tensor2.shape)  
print(rigt_tensor2.shape)  
print(down_tensor2.shape)  
print(corn_tensor2.shape)  

print(torch.dist(bulk_tensor1,bulk_tensor1_should))
print(torch.dist(rigt_tensor1,rigt_tensor1_should))  
print(torch.dist(down_tensor1,down_tensor1_should))  
print(torch.dist(corn_tensor1,corn_tensor1_should))  
print(torch.dist(bulk_tensor2,bulk_tensor2_should))  
print(torch.dist(rigt_tensor2,rigt_tensor2_should))  
print(torch.dist(down_tensor2,down_tensor2_should))  
print(torch.dist(corn_tensor2,corn_tensor2_should)) 

computer_vie_tn(tensor)

truncate = None
bulk_left ,bulk_right= apply_SVD(bulk_tensor1,left_bond=[0,1],right_bond=[2,3],truncate=truncate)
rigt_left ,rigt_right= apply_SVD(rigt_tensor1,left_bond=[0,1],right_bond=[2,3],truncate=truncate)
down_left ,down_right= apply_SVD(down_tensor1,left_bond=[0,1],right_bond=[2,3],truncate=truncate)
corn_left ,corn_right= apply_SVD(corn_tensor1,left_bond=[0,1],right_bond=[2,3],truncate=truncate)
bulk_lower,bulk_uppe = apply_SVD(bulk_tensor2,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# BCK,KDA
rigt_lower,rigt_uppe = apply_SVD(rigt_tensor2,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# BCK,KDA
down_lower,down_uppe = apply_SVD(down_tensor2,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# BCK,KDA
corn_lower,corn_uppe = apply_SVD(corn_tensor2,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# BCK,KDA

bulk_tensor           = torch.einsum("whadi,whjba,whkcb,whdcl->whijkl", bulk_lower[:-1,:-1],bulk_right[:-1,1:],bulk_uppe[:-1,1:],bulk_left[1:,1:])
rigt_bulk_tensor      = torch.einsum("hadi,hjba,hkcb,hdcl->hijkl"     , rigt_lower[:-1]    ,rigt_right[1:]    ,rigt_uppe[1:]    ,bulk_left[0,1:])
down_bulk_tensor      = torch.einsum("hadi,hjba,hkcb,hdcl->hijkl"     , down_lower[:-1]    ,bulk_right[:-1,0] ,bulk_uppe[:-1,0] ,bulk_left[1:,0])
bulk_rigt_tensor      = torch.einsum("hadi,hjba,hkcb,hdcl->hijkl"     , bulk_lower[-1,:-1] ,bulk_right[-1,1:] ,bulk_uppe[-1,1:] ,rigt_left[1:])
bulk_down_tensor      = torch.einsum("hadi,hjba,hkcb,hdcl->hijkl"     , bulk_lower[:-1,-1] ,down_right[:-1]   ,down_uppe[:-1]   ,down_left[1:])
bulk_down_corn_tensor = torch.einsum("adi,jba,kcb,dcl->ijkl"          , bulk_lower[-1,-1]  ,down_right[-1]    ,down_uppe[-1]    ,corn_left)
corn_right_bulk_tensor= torch.einsum("adi,jba,kcb,dcl->ijkl"          , corn_lower         ,rigt_right[0]     ,rigt_uppe[0]     ,bulk_left[0,0])
right_corn_down_tensor= torch.einsum("adi,jba,kcb,dcl->ijkl"          , rigt_lower[-1]     ,corn_right        ,corn_uppe        ,down_left[0])
down_bulk_right_tensor= torch.einsum("adi,jba,kcb,dcl->ijkl"          , down_lower[-1]     ,bulk_right[-1,0]  ,bulk_uppe[-1,0]  ,rigt_left[0])

left,right  = apply_SVD(tensor1,left_bond=[0,1],right_bond=[2,3],truncate=truncate)# ABK,KCD
lower,uppe  = apply_SVD(tensor2,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# BCK,KDA
new_tensor_should  = torch.einsum("whadi,whjba,whkcb,whdcl->whijkl",
                   lower,
                   right.roll(-1,1),
                   uppe.roll(-1,1),
                   left.roll(shifts=(-1, -1), dims=(0, 1))
                   )

print(bulk_tensor.shape)           
print(rigt_bulk_tensor.shape)      
print(down_bulk_tensor.shape)      
print(bulk_rigt_tensor.shape)      
print(bulk_down_tensor.shape)      
print(bulk_down_corn_tensor.shape) 
print(corn_right_bulk_tensor.shape)
print(right_corn_down_tensor.shape)
print(down_bulk_right_tensor.shape)

bulk_tensor_should            = new_tensor_should[:-2, :-2]
rigt_bulk_tensor_should       = new_tensor_should[-1,:-2]
down_bulk_tensor_should       = new_tensor_should[:-2,-1]
bulk_rigt_tensor_should       = new_tensor_should[-2,:-2]
bulk_down_tensor_should       = new_tensor_should[:-2,-2]
bulk_down_corn_tensor_should  = new_tensor_should[-2,-2]
corn_right_bulk_tensor_should = new_tensor_should[-1,-1]
right_corn_down_tensor_should = new_tensor_should[-1,-2]
down_bulk_right_tensor_should = new_tensor_should[-2,-1]

print(bulk_tensor_should.shape           )
print(rigt_bulk_tensor_should.shape      )
print(down_bulk_tensor_should.shape      )
print(bulk_rigt_tensor_should.shape      )
print(bulk_down_tensor_should.shape      )
print(bulk_down_corn_tensor_should.shape )
print(corn_right_bulk_tensor_should.shape)
print(right_corn_down_tensor_should.shape)
print(down_bulk_right_tensor_should.shape)

print(torch.dist(bulk_tensor_should            ,bulk_tensor           ))
print(torch.dist(rigt_bulk_tensor_should       ,rigt_bulk_tensor      ))
print(torch.dist(down_bulk_tensor_should       ,down_bulk_tensor      ))
print(torch.dist(bulk_rigt_tensor_should       ,bulk_rigt_tensor      ))
print(torch.dist(bulk_down_tensor_should       ,bulk_down_tensor      ))
print(torch.dist(bulk_down_corn_tensor_should  ,bulk_down_corn_tensor ))
print(torch.dist(corn_right_bulk_tensor_should ,corn_right_bulk_tensor))
print(torch.dist(right_corn_down_tensor_should ,right_corn_down_tensor))
print(torch.dist(down_bulk_right_tensor_should ,down_bulk_right_tensor))

new_tensor = torch.cat(
    [torch.cat([bulk_tensor,bulk_rigt_tensor.unsqueeze(0),rigt_bulk_tensor.unsqueeze(0)]),
     torch.cat([bulk_down_tensor,bulk_down_corn_tensor.unsqueeze(0),right_corn_down_tensor.unsqueeze(0)]).unsqueeze(1),
     torch.cat([down_bulk_tensor,down_bulk_right_tensor.unsqueeze(0),corn_right_bulk_tensor.unsqueeze(0)]).unsqueeze(1),
    ],dim=1)

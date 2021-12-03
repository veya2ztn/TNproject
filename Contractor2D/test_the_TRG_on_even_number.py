import torch
from Contractor2D.utils import apply_SVD

tensor = torch.randn(4,4,2,2,2,2)

print(uniform_shape_tensor_contractor_tn(tensor))

truncate= None
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

print(tensor1_and_tensor2_contractor_tn(tensor1,tensor2))

#print(torch.einsum("abcd,cdab->",tensor1[0,0],tensor2[0,0]))
left,right  = apply_SVD(tensor1,left_bond=[0,1],right_bond=[2,3],truncate=truncate)# ABK,KCD
lower,uppe  = apply_SVD(tensor2,left_bond=[1,2],right_bond=[3,0],truncate=truncate)# BCK,KDA
# new_tensor  = torch.einsum("whicd,whjac,whbak,whdbl->whijkl",
#                            right,
#                            uppe,
#                            left.roll(shifts=-1,dims=0),
#                            lower.roll(shifts=1,dims=1))
new_tensor  = torch.einsum("whadi,whjba,whkcb,whdcl->whijkl",
                           lower,
                           right.roll(-1,1),
                           uppe.roll(-1,1),
                           left.roll(shifts=(-1, -1), dims=(0, 1))
                           )
print(uniform_shape_tensor_contractor_tn(new_tensor))

torch.einsum("abn,nji,lkh,hdc,caw,wkj,ilm,mbd->",
             lu_lu[0,0],lu_rd[0,0],
             rd_lu[0,0],rd_rd[0,0],
             ld_ld[0,0],ld_ru[0,0],
             ru_ld[0,0],ru_ru[0,0])

torch.einsum("abji,lkdc,jcak,dilb->",
             torch.einsum("abn,nji->abji",lu_lu[0,0],lu_rd[0,0]),
             torch.einsum("lkh,hdc->lkdc",rd_lu[0,0],rd_rd[0,0]),
             torch.einsum("caw,wkj->jcak",ld_ld[0,0],ld_ru[0,0]),
             torch.einsum("ilm,mbd->dilb",ru_ld[0,0],ru_ru[0,0]))

torch.dist(torch.einsum("abn,nji->abji",lu_lu[0,0],lu_rd[0,0]),lu_tensor)

torch.dist(torch.einsum("lkh,hdc->lkdc",rd_lu[0,0],rd_rd[0,0]),rd_tensor)

torch.dist(torch.einsum("caw,wkj->jcak",ld_ld[0,0],ld_ru[0,0]),ld_tensor)

torch.dist(torch.einsum("ilm,mbd->dilb",ru_ld[0,0],ru_ru[0,0]),ru_tensor)

torch.dist(lu_tensor[0,0],tensor[0,0])

torch.dist(rd_tensor[0,0],tensor[1,1])

torch.dist(ld_tensor[0,0],tensor[0,1])

torch.dist(ru_tensor[0,0],tensor[1,0])

torch.einsum("abji,lkdc,jcak,dilb->",
             torch.einsum("abn,nji->abji",lu_lu[0,0],lu_rd[0,0]),
             torch.einsum("lkh,hdc->lkdc",rd_lu[0,0],rd_rd[0,0]),
             torch.einsum("caw,wkj->jcak",ld_ld[0,0],ld_ru[0,0]),
             torch.einsum("ilm,mbd->dilb",ru_ld[0,0],ru_ru[0,0]))

torch.einsum("abcd,idjb,ckal,jlik->",lu_tensor[0,0],ld_tensor[0,0],ru_tensor[0,0],rd_tensor[0,0])

torch.einsum("abcd,ciaj,ldkb,kjli->",lu_tensor[0,0],ld_tensor[0,0],ru_tensor[0,0],rd_tensor[0,0])

tensor1 = tensor[0,0]
tensor2 = tensor[0,1]
tensor3 = tensor[1,0]
tensor4 = tensor[1,1]
torch.einsum("abcd,idjb,ckal,jlik->",tensor1,tensor2,tensor3,tensor4)

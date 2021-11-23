import numpy as np
import torch
import time
from torchvision import datasets, transforms
from utils import *
import json,os
import sparse
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     #transforms.Normalize(mean=(0.0,), std=(1.0,))
# ])
# DATAPATH    = '/data/DATA/'
# mnist_train = datasets.MNIST(DATAPATH, train=True, download=True, transform=transform)
# mnist_test  = datasets.MNIST(DATAPATH, train=False,download=True, transform=transform)
# train_loader= torch.utils.data.DataLoader(dataset=mnist_train, batch_size=1000, shuffle=False)
# test_loader = torch.utils.data.DataLoader(dataset=mnist_test , batch_size=1000, shuffle=False)
#
# from tqdm import  tqdm
# device = 'cpu'

# for cut in [50,100,300]:
#     i=0
#     for images,labels in tqdm(train_loader):
#         #images,labels = iter(train_loader).next()
#         inputs = preprocess_sum_one(images).to(device)
#         #inputs= rd_engine(10,50,2)
#         # inputs= inputs.permute(1,2,0)#(B,num,k)->(num,k,B)
#         # inputs= torch.diag_embed(inputs)#(num,k,B)->(num,k,B,B)
#         # inputs= inputs.permute(0,2,1,3)#(num,k,B,B)->(num,B,k,B)
#         # inputs= [v for v in inputs]
#         # inputs[0]= torch.diagonal(inputs[0], dim1=0, dim2=-1)#(B,k,B) -> #(k,B)
#         L=inputs.shape[0]
#         idx1     = list(range(2*L))
#         idx2     = [i for i in range(L) for j in range(2)]
#         mps_line=[sparse.as_coo(inputs[:,0,:].transpose(1,0).numpy())]
#         for tensor in inputs.permute(1,0,2)[1:]:
#             #print(tensor.flatten().numpy().shape)
#             mps_line.append(sparse.COO([idx1,idx2],tensor.flatten().numpy(),(2*L,L)).reshape((L,2,L)))
#         #DCEngine = lambda x:truncated_SVD(x,output='QR',max_truncation_error=0.00,max_singular_values=500)
#         #mps_line,Z_list = left_canonicalize_MPS(inputs,Decomposition_Engine=DCEngine,normlization=False)
#         mps_line,z = right_canonicalize_MPS_sparse(mps_line,final_normlization =True,all_renormlization=False)
#         Decomposition_Engine=lambda x:truncated_SVD_sparse(x,output='QR',
#                                                         max_truncation_error=0.00,
#                                                         max_singular_values=cut,
#                                                        )
#         mps_line,z_2 = left_canonicalize_MPS_sparse(mps_line,final_normlization =True,
#                                                        all_renormlization=True,
#                                                        Decomposition_Engine=Decomposition_Engine)
#         mps_line[-1]*=np.prod(z)*np.prod(z_2)
#         path= f"/data/DATA/offline_SVD_data/preprocess_sum_one.cut{cut}.sparse/mps_{i}"
#         if not os.path.exists(path):os.makedirs(path)
#         for j,COO_tensor in enumerate(mps_line):
#             filename = os.path.join(path,f"unit_{j}.npz")
#             sparse.save_npz(filename,COO_tensor)
#
#         normlized_coef ={"z":z,"z_2":z_2}
#         with open(os.path.join(path,f"normlized_coef.json"),'w') as f:
#             json.dump(normlized_coef,f)
#         # state_dict = {}
#         # state_dict['xdata']=dict([[i,t.cpu()] for i,t in enumerate(mps_line)])
#         # state_dict['ydata']=labels
#         # torch.save(state_dict,f'cd /preprocess_sum_one.cut100/mps_line_{i}.pt')
#         i+=1


root_path= "/data/DATA/offline_SVD_data/preprocess_sum_one.cut50.sparse"
sc = Efficient_Sparse_Matrix_List_Saver()
mps_all = []
for i in tqdm(range(60)):
    mps_path = os.path.join(root_path,f"mps_{i}")
    mps_now  = []
    for j in tqdm(range(28*28)):
        unit_path = os.path.join(mps_path,f'unit_{j}.npz')
        unit = sparse.load_npz(unit_path)
        mps_now.append(unit)
    fast_save_path = os.path.join(mps_path,f"fast_saver")
    if not os.path.exists(fast_save_path):os.makedirs(fast_save_path)
    sc.save_sparse_data(mps_now,fast_save_path)
    mps_all.append(mps_now)
fast_save_path = os.path.join(root_path,f"fast_saver")
if not os.path.exists(fast_save_path):os.makedirs(fast_save_path)
sc.save_batch_sparse_data(mps_all,fast_save_path)

# batch_indexs  =[]
# batch_idx_size=[]
# batch_values  =[]
# batch_shape   =[]
# batch_nnz_list=[]
# for i in tqdm(range(60)):
#     mps_path = os.path.join(root_path,f"mps_{i}")
#     fast_save_path = os.path.join(mps_path,f"fast_saver")
#     all_indexs,all_idx_size,all_values,all_shape,nnz_list = sc.load_sparse_data(fast_save_path)
#     all_indexs = all_indexs.astype('uint16')
#     all_shape  = all_shape.astype('uint16')
#     batch_indexs.append(all_indexs  )
#     batch_idx_size.append(all_idx_size)
#     batch_values.append(all_values  )
#     batch_shape.append(all_shape   )
#     batch_nnz_list.append(nnz_list)
#
# fast_save_path = os.path.join(root_path,f"fast_saver")
# if not os.path.exists(fast_save_path):os.makedirs(fast_save_path)
# sc.save_batch_sparse_data_from_list([batch_indexs,batch_idx_size,batch_values,batch_shape,batch_nnz_list],
# fast_save_path)

#sc.save_sparse_data(mps_all,fast_save_path)

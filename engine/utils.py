import sparse
import numpy as np
import os
class Efficient_Sparse_Matrix_List_Saver:
    def __init__(self,dtype = 'sparse'):
        self.dtype = dtype
    def save(self,sparse_matrix_list,save_dir):
        if self.dtype == 'sparse':
            self.save_sparse_data(sparse_matrix_list,save_dir)
        else:
            raise NotImplementedError
    def load(self,save_dir):
        if self.dtype == 'sparse':
            return self.load_sparse_data(save_dir)
        else:
            raise NotImplementedError
    @staticmethod
    def get_sparse_save_data(sparse_matrix_list):
        max_shape_len = max([len(t.shape) for t in sparse_matrix_list])
        save_indexes  = []
        save_shapes   = []
        for i in range(len(sparse_matrix_list)):
            save_index = sparse_matrix_list[i].coords.transpose()
            save_shape = list(sparse_matrix_list[i].shape)
            if len(save_shape)< max_shape_len:
                padding     = max_shape_len-len(save_shape)
                save_index  = np.pad(save_index,[[0,0],[0,padding]])
                save_shape  = save_shape+[1]*padding
            save_indexes.append(save_index)
            save_shapes.append(save_shape)
        all_indexs  = np.concatenate(save_indexes)
        all_values  = np.concatenate([t.data for t in sparse_matrix_list])
        all_shape   = np.stack(save_shapes)
        all_idx_size= np.array([len(t.shape) for t in sparse_matrix_list])
        nnz_list    = np.array([t.nnz for t in sparse_matrix_list])
        assert sum(nnz_list) == len(all_indexs) == len(all_values)
        return all_indexs,all_idx_size,all_values,all_shape,nnz_list

    def save_sparse_data(self,sparse_matrix_list,save_dir):
        all_indexs,all_idx_size,all_values,all_shape,nnz_list = self.get_sparse_save_data(sparse_matrix_list)
        np.save(os.path.join(save_dir,"all_indexs"),all_indexs)
        np.save(os.path.join(save_dir,"all_idx_size"),all_idx_size)
        np.save(os.path.join(save_dir,"all_values"),all_values)
        np.save(os.path.join(save_dir,"all_shape"),all_shape)
        np.save(os.path.join(save_dir,"nnz_list"),nnz_list)

    @staticmethod
    def save_batch_sparse_data_from_list(batch_data_array,save_dir):
        batch_indexs,batch_idx_size,batch_values,batch_shape,batch_nnz_list = batch_data_array
        batch_split   =np.array([len(t) for t in batch_indexs])
        batch_indexs  =np.concatenate(batch_indexs  )
        batch_idx_size=np.concatenate(batch_idx_size)
        batch_values  =np.concatenate(batch_values  )
        batch_shape   =np.concatenate(batch_shape   )
        batch_nnz_list=np.concatenate(batch_nnz_list)

        np.save(os.path.join(save_dir,"batch_indexs"  ),batch_indexs  )
        np.save(os.path.join(save_dir,"batch_idx_size"),batch_idx_size)
        np.save(os.path.join(save_dir,"batch_values"  ),batch_values  )
        np.save(os.path.join(save_dir,"batch_shape"   ),batch_shape   )
        np.save(os.path.join(save_dir,"batch_nnz_list"),batch_nnz_list)
        np.save(os.path.join(save_dir,"batch_split")   ,batch_split)

    def save_batch_sparse_data(self,batch_sparse_matrix_list,save_dir):
        batch_indexs  =[]
        batch_idx_size=[]
        batch_values  =[]
        batch_shape   =[]
        batch_nnz_list=[]
        for sparse_matrix_list in batch_sparse_matrix_list:
            all_indexs,all_idx_size,all_values,all_shape,nnz_list = self.get_sparse_save_data(sparse_matrix_list)
            batch_indexs.append(all_indexs  )
            batch_idx_size.append(all_idx_size)
            batch_values.append(all_values  )
            batch_shape.append(all_shape   )
            batch_nnz_list.append(nnz_list)
        self.save_batch_sparse_data_from_list([batch_indexs,batch_idx_size,batch_values,batch_shape,batch_nnz_list],save_dir)
    @staticmethod
    def structure_sparse_matrix(data):
        all_indexs,all_idx_size,all_values,all_shape,nnz_list  = data
        sparse_matrix_list =[]
        start = 0
        for nnz,sz,shape in zip(nnz_list,all_idx_size,all_shape):
            indexs = all_indexs[start:start+nnz][...,:sz].transpose()
            values = all_values[start:start+nnz]
            shape  = shape[:sz]
            tensor = sparse.COO(indexs,values,shape.tolist())
            start  = start+nnz
            sparse_matrix_list.append(tensor)
        return sparse_matrix_list

    @staticmethod
    def load_sparse_data(save_dir):
        all_indexs   = np.load(os.path.join(save_dir,"all_indexs.npy"))
        all_idx_size = np.load(os.path.join(save_dir,"all_idx_size.npy"))
        all_values   = np.load(os.path.join(save_dir,"all_values.npy"))
        all_shape    = np.load(os.path.join(save_dir,"all_shape.npy"))
        nnz_list     = np.load(os.path.join(save_dir,"nnz_list.npy"))
        return all_indexs,all_idx_size,all_values,all_shape,nnz_list

    @staticmethod
    def load_batch_sparse_data(save_dir):
        batch_indexs  =np.load(os.path.join(save_dir,"batch_indexs.npy"  ))
        batch_idx_size=np.load(os.path.join(save_dir,"batch_idx_size.npy"))
        batch_values  =np.load(os.path.join(save_dir,"batch_values.npy"  ))
        batch_shape   =np.load(os.path.join(save_dir,"batch_shape.npy"   ))
        batch_nnz_list=np.load(os.path.join(save_dir,"batch_nnz_list.npy"))
        batch_split   =np.load(os.path.join(save_dir,"batch_split.npy")   )
        batch_idx_size=np.split(batch_idx_size,len(batch_split))
        batch_shape   =np.split(batch_shape   ,len(batch_split))
        batch_nnz_list=np.split(batch_nnz_list,len(batch_split))

        batch_split   =np.cumsum(batch_split)[:-1]
        batch_indexs  =np.split(batch_indexs  ,batch_split)
        batch_values  =np.split(batch_values  ,batch_split)
        


        return batch_indexs,batch_idx_size,batch_values,batch_shape,batch_nnz_list

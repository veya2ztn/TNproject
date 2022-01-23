import tensornetwork as tn
import numpy as np
from tensornetwork import contractors
import numpy as np
import tensorflow as tf
import torch

def get_BatchedTensorNetwork(inputs,backend):
    # only for input tensor network virtual bond is 1 (so it is regarded as no connection)
    # if the virtual bond big than 1, then call `slow_block_diag`
    if backend in ['torch','torchgpu','pytorch']:
        inputs= inputs.permute(1,2,0)#(B,num,k)->(num,k,B)
        out   = torch.diag_embed(inputs)#(num,k,B)->(num,k,B,B)
        out   = out.permute(0,2,1,3)#(num,k,B,B)->(num,B,k,B)
        out   = [v for v in out]
        out[0]= torch.diagonal(out[0], dim1=0, dim2=-1).transpose(1,0)#(B,k,B) -> #(B,k)
    elif backend == 'numpy':
        out=[]
        for i,tenser_list in enumerate(inputs.transpose(1,2,0)):#(B,num,k)->(num,k,B)
            if i == 0:
                out.append(tenser_list.transpose(1,0))    #(B,k)
            else:
                out.append(np.stack([np.diag(a) for a in tenser_list],1))#(B,k,B)
    elif backend == 'tensorflow':
        inputs = tf.transpose(inputs,(1,2,0))#(B,num,k)->(num,k,B)
        out    = tf.linalg.diag(inputs)#(num,k,B)->(num,k,B,B)
        out    = tf.transpose(out,(0,2,1,3))#(num,k,B,B)->(num,B,k,B)
        out    = tf.split(out, num_or_size_splits=out.shape[0], axis=0)
        out    = [tf.squeeze(v,0) for v in out]
        out[0] = tf.transpose(out[0],(1,0,2))#(B,k,B) -> (k,B,B)
        out[0] = tf.linalg.diag_part(out[0])#(k,B,B) -> (k,B)
        out[0] = tf.transpose(out[0],(1,0))#(k,B)-> (B,k)
    else:
        print(backend)
        raise NotImplementedError
    return out
def contract_mps_one_sample(mps_var,input_vec,hidden_node_idx):
    num        = len(mps_var)
    mps_nodes  = [tn.Node(v, name=f"t{i}") for i,v in enumerate(mps_var)]
    mps_edges  = [mps_nodes[0][0]^mps_nodes[1][0]]+[mps_nodes[i][-1]^mps_nodes[i+1][0] for i in range(1,len(mps_nodes)-1)]
    if mps_nodes[0].backend.name == 'tensorflow':
        input_vec= tf.split(input_vec, num_or_size_splits=input_vec.shape[0], axis=0)
        inp_nodes= [tn.Node(tf.reshape(v,(v.shape[1],)), name=f"i{i}") for i,v in enumerate(input_vec)]
    else:
        inp_nodes = [tn.Node(v, name=f"i{i}") for i,v in enumerate(input_vec)]
    assert len(hidden_node_idx) == len(inp_nodes)
    for node_idx, input_node in zip(hidden_node_idx,inp_nodes):
        mps_nodes[node_idx][1]^input_node[0]
    ans = contractors.auto(mps_nodes+inp_nodes).tensor
    return ans

def way_vectorized_map(mps_var,batch,hidden_node_idx):
    # tensorflow mode. and it equals to parallal cpu mode
    return tf.vectorized_map(lambda vec: contract_mps_one_sample(mps_var,vec, hidden_node_idx), batch)

def way_for_loop(mps_var,batch,hidden_node_idx):
    out= [contract_mps_one_sample(mps_var,vec,hidden_node_idx) for vec in batch]
    if isinstance(mps_var[0],torch.Tensor):
        return torch.stack(out)
    elif isinstance(mps_var[0],np.ndarray):
        return np.stack(out )
    else:
        return out

def way_dense_kron_sum(mps_var,input_data,hidden_node_idx,givenBTN=False):

    num        = len(mps_var)
    mps_nodes  = [tn.Node(v, name=f"t{i}") for i,v in enumerate(mps_var)]
    mps_edges  = [mps_nodes[0][0]^mps_nodes[1][0]]+[mps_nodes[i][-1]^mps_nodes[i+1][0] for i in range(1,len(mps_nodes)-1)]

    batch      = get_BatchedTensorNetwork(input_data,mps_nodes[0].backend.name) if not givenBTN else input_data
    inp_nodes  = [tn.Node(v, name=f"i{i}") for i,v in enumerate(batch)]
    inp_edges  = [inp_nodes[0][0]^inp_nodes[1][0]]+ [inp_nodes[i][-1]^inp_nodes[i+1][0] for i in range(1,len(inp_nodes)-1)]

    for node_idx, input_node in zip(hidden_node_idx,inp_nodes):
        mps_nodes[node_idx][1]^input_node[1]
        #mps_nodes[node_idx][1]^inp_nodes[node_idx][1]
    label_node_idx = [i for i in range(len(mps_var)) if i not in hidden_node_idx][0]
    ans = contractors.auto(mps_nodes+inp_nodes,output_edge_order=[inp_nodes[-1][2],mps_nodes[label_node_idx][1]]).tensor
    return ans

def way_dense_kron_sum_pure(mps_var,input_data,hidden_node_idx):
    return way_dense_kron_sum(mps_var,input_data,hidden_node_idx,givenBTN=True)



def way_for_efficient_coding(mps_var,input_data,hidden_node_idx):
    batch_input = []
    num = 0
    for i in range(len(mps_var)):
        now_unit = mps_var[i]
        if i not in hidden_node_idx:
            batch_unit = now_unit
        else:
            now_inpt   = input_data[:,num]
            if isinstance(mps_var[0],torch.Tensor):
                batch_unit = torch.tensordot(now_inpt,now_unit,dims=([-1], [1]))
            elif isinstance(mps_var[0],np.ndarray):
                batch_unit = np.tensordot(now_inpt,now_unit,axes=([-1], [1]))
            else:
                batch_unit = tf.tensordot(now_inpt,now_unit,axes=([-1], [1]))

            num+=1
        batch_input.append(batch_unit)

    # (B,bd)<->(bd,cd,bd)<->(B,bd,bd)<->(B,bd,bd)<->(B,bd,bd).....<->(B,bd)
    if isinstance(mps_var[0],torch.Tensor):
        einsum = torch.einsum
    elif isinstance(mps_var[0],np.ndarray):
        einsum = np.einsum
    else:
        einsum = tf.einsum
    last = batch_input[-1]
    for i in range(len(batch_input)-2,0,-1):
        if i in hidden_node_idx:
            if len(last.shape)==2:
                last = einsum("ba,bca->bc",last,batch_input[i])
            elif len(last.shape)==3:
                last = einsum("boa,bca->boc",last,batch_input[i])
        else:
            last = einsum("ba,coa->boc",last,batch_input[i])
    last = einsum("boa,ba->bo",last,batch_input[0])
    return last



def slow_block_diag(m,version=1):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    block_diag(torch.ones(4,3,2,1))
    should give a 12 x 8 x 4 matrix with blocks of 3 x 2 x 1 ones.

    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:m = torch.stack(m, 0)
    d    = m.dim()
    n    = m.shape[0]
    block_size = m.shape[1:]
    final_shape = [n*s for s in block_size]
    if version ==1:
        insect_size= [1]*(d-1)
        tensor_shape = [rv for r in zip(insect_size,block_size) for rv in r]
        tensor_shape[0]=n
        m2   = m.reshape(tuple(tensor_shape))#(B,n1,n2,n3..,)->(B,n1,1,n2,1,n3,..)

        eye  = torch.zeros([n]*(d-1))
        for idx in range(n):eye[[torch.arange(n)]*(d-1)]=1
        eye_shape = [rv for r in zip([n]*(d-1),[1]*(d-1)) for rv in r]
        eye = eye.reshape(tuple(eye_shape)).to(m.device)#(B,1,B,1,B,1..,)
        return (m2*eye).reshape(tuple(final_shape))
    elif version ==2:
        temp   = torch.zeros(final_shape).to(m.device)
        if d == 2:
            s = block_size[0]
            for idx,block in enumerate(m):temp[idx*s:(idx+1)*s]=block
        elif d == 3:
            s1,s2 = block_size
            for idx,block in enumerate(m):temp[idx*s1:(idx+1)*s1,idx*s2:(idx+1)*s2]=block
        elif d == 3:
            s1,s2 = block_size
            for idx,block in enumerate(m):temp[idx*s1:(idx+1)*s1,idx*s2:(idx+1)*s2]=block
        elif d == 4:
            s1,s2,s3 = block_size
            for idx,block in enumerate(m):temp[idx*s1:(idx+1)*s1,
                                               idx*s2:(idx+1)*s2,
                                               idx*s3:(idx+1)*s3]=block
        elif d == 5:
            s1,s2,s3,s4 = block_size
            for idx,block in enumerate(m):temp[idx*s1:(idx+1)*s1,
                                               idx*s2:(idx+1)*s2,
                                               idx*s3:(idx+1)*s3,
                                               idx*s4:(idx+1)*s4]=block
        else:
            raise NotImplementedError
        return temp

def batch_stack(tensor,**kargs):
    if len(tensor[0].shape)==1:#(B,X)
        return torch.flatten(tensor)#(BX,)
    elif len(tensor[0].shape)==2:#(B,X,X)
        return torch.block_diag(*tensor)#(BX,BX)
    else:
        return slow_block_diag(tensor,**kargs)#(BX,BX)

import os
def parse(line,qargs):
    '''
    https://zhuanlan.zhihu.com/p/28690706
    line:
        a line of text
    qargs:
        query arguments
    return:
        a dict of gpu infos
    Pasing a line of csv format text returned by nvidia-smi
    解析一行nvidia-smi返回的csv格式文本
    '''
    #numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']#可计数的参数
    numberic_args = ['memory.used','memory.free','memory.total']
    power_manage_enable=lambda v:(not 'Not Support' in v)#lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
    to_numberic=lambda v:float(v.upper().strip().replace('MIB','').replace('W',''))#带单位字符串去掉单位
    process = lambda k,v:((int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
    return {k:process(k,v) for k,v in zip(qargs,line.strip().split(','))}

def query_gpu(qargs=['memory.used','memory.free','memory.total']):
    '''
    qargs:
        query arguments
    return:
        a list of dict
    Querying GPUs infos
    查询GPU信息
    '''
    #qargs =['index','gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']+ qargs
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()
    return [parse(line,qargs) for line in results]



def get_test_mps(backend='numpy',bd = 3,pd = 2,cd = 10,num=21):
    if backend == 'numpy':
        tn.set_default_backend("numpy")
        generator = np.random.rand
    elif backend == 'tensorflow':
        tn.set_default_backend("tensorflow")
        generator = lambda *a:tf.random.normal(a)
    elif backend == 'torch':
        tn.set_default_backend("pytorch")
        generator = torch.randn
    elif backend == 'torchgpu':
        tn.set_default_backend("pytorch")
        generator = lambda *a:torch.randn(a).to('cuda:0')
    else: NotImplementedError
    mps_var = ([generator(bd,pd)]+
               [generator(bd,pd,bd)]*(num//2-1)+
               [generator(bd,cd,bd)]+
               [generator(bd,pd,bd)]*(num//2-1)+
               [generator(bd,pd)])
    hidden_node_idx = [i for i,v in enumerate(mps_var) if v.shape!=(bd,cd,bd)]
    return mps_var,hidden_node_idx,generator
def get_test_mps_ring(backend='numpy',bd = 3,pd = 2,cd = 10,num=21):
    if backend == 'numpy':
        tn.set_default_backend("numpy")
        generator = np.random.rand
    elif backend == 'tensorflow':
        tn.set_default_backend("tensorflow")
        generator = lambda *a:tf.random.normal(a)
    elif backend == 'torch':
        tn.set_default_backend("pytorch")
        generator = torch.randn
    elif backend == 'torchgpu':
        tn.set_default_backend("pytorch")
        generator = lambda *a:torch.randn(a).to('cuda:0')
    else: NotImplementedError
    mps_var = ([generator(bd,pd,bd)]+
               [generator(bd,pd,bd)]*(num//2-1)+
               [generator(bd,cd,bd)]+
               [generator(bd,pd,bd)]*(num//2-1)+
               [generator(bd,pd,bd)])
    hidden_node_idx = [i for i,v in enumerate(mps_var) if v.shape!=(bd,cd,bd)]
    return mps_var,hidden_node_idx,generator

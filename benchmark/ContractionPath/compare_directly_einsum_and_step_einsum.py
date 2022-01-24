from utils import *
from tensornetwork import contractors

#set_rdn_seed(200)

def rd_engine(*x,**kargs):
    x =  torch.randn(*x,device='cuda',**kargs)
    x/=  torch.norm(x).sqrt()
    x =  torch.autograd.Variable(x,requires_grad=True)
    return x
def cuda_reset():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    #print(f"empty:{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"),

def submatrix_test_full_contraction_einsum(D=5,L=5,W=None,H=None,B=1,backward=True):
    W = L if W is None else W
    H = L if H is None else H
    tn2D_shape_list = [[(D,D)]+[(D,D,D)]*(H-1)]+[[(D,D,D)]+[(D,D,D,D)]*(H-1)]*(W-1)
    # for t in tn2D_shape_list:
    #     for s in t:print(s,end=' ')
    #     print("========")
    node_list,sublist_list,outlist = sub_network_tn(tn2D_shape_list)
    # re-align to get proper index order
    last_idx = outlist.pop()
    outlist.insert(W-1,last_idx)
    path,info                      = get_optim_path_by_oe_from_tn(node_list)
    tensor_list     = [rd_engine(B,*l) for t in tn2D_shape_list for l in t]
    assert len(tensor_list)==len(sublist_list)
    operands=[]
    for tensor,sublist in zip(tensor_list,sublist_list):
        operand = [tensor,[...,*sublist]]
        operands+=operand
    operands+= [[...,*outlist]]
    out = torch.einsum(*operands,optimize=path).flatten(-H-H,-H-1).flatten(-H,-1)
    if backward:
        loss= out.norm()
        loss.backward()
    return out,f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"
def submatrix_test_sequence_contraction_einsum(D=5,L=5,B=1,backward=True):
    L=L-1
    tn2D_shape_list = [[(D,D)]+[(D,D,D)]*(L-1)]+[[(D,D,D)]+[(D,D,D,D)]*(L-1)]*(L-1)
    node_list,sublist_list,outlist = sub_network_tn(tn2D_shape_list)
    path,info                      = get_optim_path_by_oe_from_tn(node_list)
    tensor_list   = [rd_engine(B,*l) for t in tn2D_shape_list for l in t]
    operands=[]
    for tensor,sublist in zip(tensor_list,sublist_list):
        operand = [tensor,[...,*sublist]]
        operands+=operand
    #operands+= [[...,4,6,3,7]]
    operands+= [[...,*outlist]]
    corn = torch.einsum(*operands,optimize=path).flatten(-2*L,-L-1).flatten(-L,-1)

    tn2D_shape_list                = [[(D,D,D)] + [(D,D,D,D) for i in range(L-1)]]
    node_list,sublist_list,outlist = sub_network_tn(tn2D_shape_list)
    path,info                      = get_optim_path_by_oe_from_tn(node_list)
    tensor_list   = [rd_engine(2,B,*l) for t in tn2D_shape_list for l in t]
    assert len(tensor_list)==len(sublist_list)
    operands=[]
    for tensor,sublist in zip(tensor_list,sublist_list):
        operand = [tensor,[...,*sublist]]
        operands+=operand
    operands+= [[...,*outlist]]
    edge1,edge2 = torch.einsum(*operands,optimize=path).flatten(-2*L,-L-1).flatten(-L,-1)
    # tensor_l   = edge_tensors
    # path_final = oe.contract_path(equation, *tensor_l)[0]
    # edge1,edge2= torch.einsum(equation,*tensor_l, optimize=path_final).flatten(-2*L,-L-1).flatten(-L,-1)

    cent_tensor   = rd_engine(B,D,D,D,D)
    equation   = "kab,kcdb,kefcg,kgha->khedf"
    tensor_l   = [corn ,edge1,cent_tensor,edge2]
    path_final = oe.contract_path(equation, *tensor_l)[0]
    out        = torch.einsum(equation,*tensor_l, optimize=path_final).flatten(-4,-3).flatten(-2,-1)
    if backward:
        loss= out.norm()
        loss.backward()
    return out,f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"
def submatrix_test_recurrence_contraction_einsum(D=5,L=5,B=1,backward=True):
    core_shape = [[(D,D)]+[(D,D,D)]*(L-2)]+[[(D,D,D)]+[(D,D,D,D)]*(L-2)]*(L-2)
    edge_shape = [[(D,D,D)]+[(D,D,D,D)]*(L-2)]
    cent_shape = (D,D,D,D)
    L = L-1
    if L==2:
        node_list,sublist_list,outlist = sub_network_tn(core_shape)
        path,info                      = get_optim_path_by_oe_from_tn(node_list)
        tensor_list   = [rd_engine(B,*l) for t in core_shape for l in t]
        operands=[]
        for tensor,sublist in zip(tensor_list,sublist_list):
            operand = [tensor,[...,*sublist]]
            operands+=operand
        operands+= [[...,*outlist]]
        corn = torch.einsum(*operands,optimize=path).flatten(-2*L,-L-1).flatten(-L,-1)
    else:
        corn,_= submatrix_test_recurrence_contraction_einsum(D=D,L=L,B=B,backward=False)

    node_list,sublist_list,outlist = sub_network_tn(edge_shape)
    path,info                      = get_optim_path_by_oe_from_tn(node_list)
    tensor_list   = [rd_engine(2,B,*l) for t in edge_shape for l in t]
    assert len(tensor_list)==len(sublist_list)
    operands=[]
    for tensor,sublist in zip(tensor_list,sublist_list):
        operand = [tensor,[...,*sublist]]
        operands+=operand
    operands+= [[...,*outlist]]
    edge1,edge2   = torch.einsum(*operands,optimize=path).flatten(-2*L,-L-1).flatten(-L,-1)
    cent_tensor   = rd_engine(B,*cent_shape)
    equation      = "kab,kcdb,kefcg,kgha->khedf"
    #print(corn.shape,edge1.shape,cent_tensor.shape,edge2.shape)
    tensor_l      = [corn ,edge1,cent_tensor,edge2]
    path_final    = oe.contract_path(equation, *tensor_l)[0]
    out           = torch.einsum(equation,*tensor_l, optimize=path_final).flatten(-4,-3).flatten(-2,-1)
    #print(f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M",end="->")
    if backward:
        loss= out.norm()
        loss.backward()
    return out,f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"

#cuda_reset();print(submatrix_test_full_contraction_einsum(D=5,W=3,H=2)[1])
#cuda_reset();print(  submatrix_test_sequence_contraction_einsum(D=5,L=5)[1])
#cuda_reset();print(submatrix_test_recurrence_contraction_einsum(D=5,L=5)[1])

def fullmatrix_test_with_tn(D=5,L=5):
    # tensornetwork pack has memory leak problem.
    # the contraction performance should same as using oe.contract directly.
    top_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [  (D,D)]
    mid_shape_list  =[  [(D,D,D)]+ [(D,D,D,D) for i in range(L-2)] + [(D,D,D)] for _ in range(L-2)]
    bot_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [(D,D)]
    tn2D_shape_list = [top_shape_list]+mid_shape_list+[bot_shape_list]
    node_list,sublist_list,outlist =  create_templete_2DTN_tn(tn2D_shape_list,engine=rd_engine)
    ans  = contractors.auto(node_list,output_edge_order=get_all_dangling(node_list))
    out  = ans.tensor
    loss = out.norm()
    loss.backward()
    del node_list
    return out,f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"
def fullmatrix_test_directly(D=5,L=5):
    top_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [  (D,D)]
    mid_shape_list  =[  [(D,D,D)]+ [(D,D,D,D) for i in range(L-2)] + [(D,D,D)] for _ in range(L-2)]
    bot_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [(D,D)]
    tn2D_shape_list = [top_shape_list]+mid_shape_list+[bot_shape_list]
    node_list,sublist_list,outlist =  create_templete_2DTN_tn(tn2D_shape_list)
    path,info     = get_optim_path_by_oe_from_tn(node_list)
    operands=[]
    #tensor_list = [t.tensor for t in node_list]
    tensor_list   = [rd_engine(*l) for t in tn2D_shape_list for l in t]
    for tensor,sublist in zip(tensor_list,sublist_list):
        operand = [tensor,[...,*sublist]]
        operands+=operand
    operands+= [[...,*outlist]]
    out = torch.einsum(*operands,optimize=path)
    #out = oe.contract(*operands,optimize='random-greedy')# then should not use B
    loss = out.norm()
    loss.backward()
    return out,f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"
def fullmatrix_test_corn_4(D=5,L=6,B=1):
    corn,_ = submatrix_test_full_contraction_einsum(D=D,W=L//2+L%2,H=L//2,B=4,backward=False)
    if L%2 == 0:
        corn   = torch.einsum("kab,kbc->kac",corn[[0,2]],corn[[1,3]])
        out    = torch.einsum("ab,ba->",corn[0],corn[1])
    else:
        cent = rd_engine(D,D,D,D)
        out  = oe.contract("iab,jbc,kcd,lda,ijkl->",corn[0],corn[1],corn[2],corn[3],cent)
    loss   = out.norm()
    loss.backward()
    return corn,f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"
def fullmatrix_RG(D=3,L=7):
    def get_index(list1,list2):
        x,y=np.meshgrid(list1,list2)
        return y.flatten(),x.flatten()
    #assert L in [2,4,8,16,32,64]
    top_shape_list  = [(D,D,1,1)]  + [(D,D,1,D) for i in range(L-2)] + [(D,1,1,D)]
    mid_shape_list  =[[(D,D,D,1)]+   [(D,D,D,D) for i in range(L-2)] + [(D,1,D,D)] for _ in range(L-2)]
    bot_shape_list  = [(1,D,D,1)]  + [(1,D,D,D) for i in range(L-2)] + [(1,1,D,D)]
    tn2D_shape_list = [top_shape_list]+mid_shape_list+[bot_shape_list]
    tensor_list     = [[rd_engine(*l) for l in t] for t in tn2D_shape_list ]

    while True:
        W = len(tensor_list)
        H = len(tensor_list)
        if W<=1:break
        if W%2 ==1:
            last_line_dw = tensor_list.pop()
            last_line_up = tensor_list.pop()
            tensor_list.append([oe.contract("abcd,kjal->kbjcdl",up,dw).flatten(-2,-1).flatten(1,2)
                    for up,dw in zip(last_line_up,last_line_dw)])
        if H%2 ==1:
            tensor_list  = list(map(list,zip(*tensor_list)))
            last_line_dw = tensor_list.pop()
            last_line_up = tensor_list.pop()
            tensor_list.append([oe.contract("abcd,efgb->aefcgd",up,dw).flatten(-3,-2).flatten(0,1)
                    for up,dw in zip(last_line_up,last_line_dw)])
            tensor_list  = list(map(list,zip(*tensor_list)))
        W = len(tensor_list)
        H = len(tensor_list)
        tensor_list = [l for t in tensor_list for l in t]
        #print(f"tensor number {len(tensor_list)}= {W}x{H}")
        assert (W%2 == 0) and (H%2 == 0 )

        lu_tensor_idx = np.ravel_multi_index(get_index(np.arange(0,W,2),np.arange(0,H,2)), (W,H))
        ld_tensor_idx = np.ravel_multi_index(get_index(np.arange(0,W,2),np.arange(1,H,2)), (W,H))
        ru_tensor_idx = np.ravel_multi_index(get_index(np.arange(1,W,2),np.arange(0,H,2)), (W,H))
        rd_tensor_idx = np.ravel_multi_index(get_index(np.arange(1,W,2),np.arange(1,H,2)), (W,H))

        lu_tensor = [tensor_list[i] for i in lu_tensor_idx]
        ld_tensor = [tensor_list[i] for i in ld_tensor_idx]
        ru_tensor = [tensor_list[i] for i in ru_tensor_idx]
        rd_tensor = [tensor_list[i] for i in rd_tensor_idx]

        tensor_list = []
        for lu,ld,ru,rd in zip(lu_tensor,ld_tensor,ru_tensor,rd_tensor):
            out       = oe.contract("abcd,kjal,efgb,hiej->khficgdl",lu,ld,ru,rd).flatten(2,3).flatten(-4,-3).flatten(0,1).flatten(-2,-1)
            tensor_list.append(out)
        #print([t.shape for t in tensor_list])
        n = W//2
        tensor_list = [tensor_list[i:i + n] for i in range(0, len(tensor_list), n)]
    out = tensor_list[0][0].squeeze()
    loss = out.norm()
    loss.backward()
    return out,f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"
def fullmatrix_boundary(D=3,L=7,W=None,H=None):
    W = L if W is None else W
    H = L if H is None else H
    def get_index(list1,list2):
        x,y=np.meshgrid(list1,list2)
        return y.flatten(),x.flatten()
    #assert L in [2,4,8,16,32,64]
    top_shape_list  = [(2,D,D,1,1)]  + [(2,D,D,1,D) for i in range(H-2)] + [(2,D,1,1,D)]
    bkl_shape_list  =[[(2,D,D,D,1)]+   [(2,D,D,D,D) for i in range(H-2)] + [(2,D,1,D,D)] for _ in range((W-2)//2)]
    mid_shape_list  = [  (D,D,D,1)]  + [  (D,D,D,D) for i in range(H-2)] + [(D,1,D,D)]
    tn2D_shape_list = [top_shape_list]+mid_shape_list
    
    tensor_list     = [[rd_engine(*l) for l in t] for t in tn2D_shape_list ]
    while True:
        W = len(tensor_list)
        H = len(tensor_list)
        if W<=1:break
        if W%2 ==1:
            last_line_dw = tensor_list.pop()
            last_line_up = tensor_list.pop()
            tensor_list.append([oe.contract("abcd,kjal->kbjcdl",up,dw).flatten(-2,-1).flatten(1,2)
                    for up,dw in zip(last_line_up,last_line_dw)])
        if H%2 ==1:
            tensor_list  = list(map(list,zip(*tensor_list)))
            last_line_dw = tensor_list.pop()
            last_line_up = tensor_list.pop()
            tensor_list.append([oe.contract("abcd,efgb->aefcgd",up,dw).flatten(-3,-2).flatten(0,1)
                    for up,dw in zip(last_line_up,last_line_dw)])
            tensor_list  = list(map(list,zip(*tensor_list)))
        W = len(tensor_list)
        H = len(tensor_list)
        tensor_list = [l for t in tensor_list for l in t]
        #print(f"tensor number {len(tensor_list)}= {W}x{H}")
        assert (W%2 == 0) and (H%2 == 0 )

        lu_tensor_idx = np.ravel_multi_index(get_index(np.arange(0,W,2),np.arange(0,H,2)), (W,H))
        ld_tensor_idx = np.ravel_multi_index(get_index(np.arange(0,W,2),np.arange(1,H,2)), (W,H))
        ru_tensor_idx = np.ravel_multi_index(get_index(np.arange(1,W,2),np.arange(0,H,2)), (W,H))
        rd_tensor_idx = np.ravel_multi_index(get_index(np.arange(1,W,2),np.arange(1,H,2)), (W,H))

        lu_tensor = [tensor_list[i] for i in lu_tensor_idx]
        ld_tensor = [tensor_list[i] for i in ld_tensor_idx]
        ru_tensor = [tensor_list[i] for i in ru_tensor_idx]
        rd_tensor = [tensor_list[i] for i in rd_tensor_idx]

        tensor_list = []
        for lu,ld,ru,rd in zip(lu_tensor,ld_tensor,ru_tensor,rd_tensor):
            out       = oe.contract("abcd,kjal,efgb,hiej->khficgdl",lu,ld,ru,rd).flatten(2,3).flatten(-4,-3).flatten(0,1).flatten(-2,-1)
            tensor_list.append(out)
        #print([t.shape for t in tensor_list])
        n = W//2
        tensor_list = [tensor_list[i:i + n] for i in range(0, len(tensor_list), n)]
    out = tensor_list[0][0].squeeze()
    loss = out.norm()
    loss.backward()
    return out,f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"

# import torch.utils.benchmark as benchmark
# L=5;D=20
# results = []
# for L in [3,4,5,6,7,8]:
#     for D in [2,4,6,8,10,12]:
#         results.append(benchmark.Timer(
#             stmt='cuda_reset();cost=function(D=D,L=L)[1]',
#             globals={'D': D, 'L': L,'cuda_reset':cuda_reset,'function':fullmatrix_test_directly},
#             description='fullmatrix_test_directly',
#         ).blocked_autorange())
#         results.append(benchmark.Timer(
#             stmt='cuda_reset();cost=function(D=D,L=L)[1]',
#             globals={'D': D, 'L': L,'cuda_reset':cuda_reset,'function':fullmatrix_RG},
#             description='fullmatrix_RG',
#         ).blocked_autorange())
# compare = benchmark.Compare(results)
# compare.trim_significant_figures()
# compare.colorize(rowwise=True)
# compare.print()
D=10;L=5
cuda_reset();print(fullmatrix_test_directly(D=D,L=L)[1]);
cuda_reset();print(fullmatrix_test_corn_4(D=D,L=L)[1]);
cuda_reset();print(fullmatrix_RG(D=D,L=L)[1]);

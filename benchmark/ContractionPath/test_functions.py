import sys,os
import numpy as np
SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models.model_utils import *
import random
import torch.backends.cudnn as cudnn
def set_rdn_seed(seed):
    #seed = 200
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    torch.manual_seed(seed)
    cudnn.enabled = True
    cudnn.deterministic = True
    torch.cuda.manual_seed(seed)


#set_rdn_seed(200)

def rd_engine(*x,**kargs):
    x =  torch.randn(*x,device='cuda',**kargs)
    x/=  torch.norm(x).sqrt()
    x =  torch.autograd.Variable(x,requires_grad=True)
    return x
rd_engine = None
def cuda_reset():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    #print(f"empty:{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"),

contractor_pool={"torch":torch.einsum,'oe':oe.contract}

def submatrix_test_full_contraction_einsum(D=5,L=5,W=None,B=1,H=None,
                                            backward=True,
                                            path_recorder=None,
                                            contractor='torch',rd_engine=rd_engine):
    contract_engine = contractor_pool[contractor]

    W = L if W is None else W
    H = L if H is None else H
    tn2D_shape_list = [[(D,D)]+[(D,D,D)]*(H-1)]+[[(D,D,D)]+[(D,D,D,D)]*(H-1)]*(W-1)
    # for t in tn2D_shape_list:
    #     for s in t:print(s,end=' ')
    #     print("========")
    path,sublist_list,outlist = get_best_path(tn2D_shape_list,store=path_recorder,type='sub')
    # node_list,sublist_list,outlist = sub_network_tn(tn2D_shape_list)
    # # re-align to get proper index order
    # #last_idx = outlist.pop()
    # #outlist.insert(W-1,last_idx)
    # path,info                      = get_optim_path_by_oe_from_tn(node_list)
    tensor_list     = [rd_engine(B,*l) for t in tn2D_shape_list for l in t]

    assert len(tensor_list)==len(sublist_list)
    operands = structure_operands(tensor_list,sublist_list,outlist,type=contractor)
    out      = contract_engine(*operands,optimize=path).flatten(-H-H,-H-1).flatten(-H,-1)
    if backward:
        loss= out.norm()
        if loss.requires_grad:
            loss.backward()
    return out,f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"

def submatrix_test_sequence_contraction_einsum(D=5,L=5,B=1,
                                                backward=True,
                                                path_recorder=None,
                                                contractor='torch',
                                                recurrent = False,rd_engine=rd_engine):
    contract_engine = contractor_pool[contractor]
    L = L-1
    if recurrent and L>2:
        corn,_= submatrix_test_sequence_contraction_einsum(D=D,L=L,B=B,
        backward=False,
        path_recorder=path_recorder,
        contractor=contractor,
        recurrent=recurrent)
    else:
        corn,_ = submatrix_test_full_contraction_einsum(D=D,L=L,B=B,backward=False,path_recorder=path_recorder,contractor=contractor)

    tn2D_shape_list = [[(D,D,D)]+[(D,D,D,D)]*(L-1)]
    path,sublist_list,outlist = get_best_path(tn2D_shape_list,store=path_recorder,type='sub')
    #if contractor == 'torch':
    tensor_list   = [rd_engine(2,B,*l) for t in tn2D_shape_list for l in t]
    assert len(tensor_list)==len(sublist_list)
    operands    = structure_operands(tensor_list,sublist_list,outlist,type=contractor)
    edge1,edge2 = contract_engine(*operands,optimize=path).flatten(-2*L,-L-1).flatten(-L,-1)
    cent_tensor = rd_engine(B,D,D,D,D)
    equation    = "kab,kcdb,kefcg,kgha->khedf"
    tensor_l    = [corn ,edge1,cent_tensor,edge2]
    #path_final = oe.contract_path(equation, *tensor_l)[0]
    path_final = get_best_path_via_oe(equation,tensor_l,store=path_recorder)
    out        = contract_engine(equation,*tensor_l, optimize=path_final).flatten(-4,-3).flatten(-2,-1)
    if backward:
        loss= out.norm()
        if loss.requires_grad:
            loss.backward()
    return out,f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"

def submatrix_test_recurrent_contraction_einsum(**kargs):
    return submatrix_test_sequence_contraction_einsum(recurrent=True,**kargs)

def submatrix_test_full_contraction_oe(**kargs):
    return submatrix_test_full_contraction_einsum(contractor='oe',**kargs)
def submatrix_test_sequence_contraction_oe(**kargs):
    return submatrix_test_full_contraction_einsum(contractor='oe',**kargs)
def submatrix_test_recurrent_contraction_oe(**kargs):
    return submatrix_test_recurrent_contraction_einsum(contractor='oe',**kargs)


#cuda_reset();print(submatrix_test_full_contraction_einsum(D=5,W=3,H=2)[1])
#cuda_reset();print(  submatrix_test_sequence_contraction_einsum(D=5,L=5)[1])
#cuda_reset();print(submatrix_test_recurrence_contraction_einsum(D=5,L=5)[1])

def fullmatrix_test_with_tn(D=5,L=5):
    from tensornetwork import contractors
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
def fullmatrix_test_directly(D=5,L=5,B=2,path_recorder=None,contractor='torch',rd_engine=rd_engine):
    contract_engine = contractor_pool[contractor]
    top_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [  (D,D)]
    mid_shape_list  =[  [(D,D,D)]+ [(D,D,D,D) for i in range(L-2)] + [(D,D,D)] for _ in range(L-2)]
    bot_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [(D,D)]
    tn2D_shape_list = [top_shape_list]+mid_shape_list+[bot_shape_list]

    path,sublist_list,outlist = get_best_path(tn2D_shape_list,store=path_recorder,type='full')
    # node_list,sublist_list,outlist =  create_templete_2DTN_tn(tn2D_shape_list)
    # path,info     = get_optim_path_by_oe_from_tn(node_list,store = path_recorder)
    operands=[]
    #tensor_list = [t.tensor for t in node_list]
    tensor_list = [rd_engine(B,*l) for t in tn2D_shape_list for l in t]
    operands    = structure_operands(tensor_list,sublist_list,outlist,type=contractor)
    out         = contract_engine(*operands,optimize=path)
    loss = out.norm()
    if loss.requires_grad:
        loss.backward()
    return out,f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"
def fullmatrix_test_corn_4(D=5,L=6,B=2,path_recorder=None,contractor='torch',rd_engine=rd_engine):
    contract_engine = contractor_pool[contractor]
    corn,_ = submatrix_test_full_contraction_einsum(D=D,W=L//2+L%2,H=L//2,B=4*B,backward=False,rd_engine=rd_engine)
    size   = corn.shape[1:]
    corn   = corn.reshape(B,4,*size)
    if L%2 == 0:
        corn   = torch.einsum("xkab,xkbc->xkac",corn[:,[0,2]],corn[:,[1,3]])
        out    = torch.einsum("xab,xba->",corn[:,0],corn[:,1])
    else:
        cent       = rd_engine(B,D,D,D,D)
        equation   = "xiab,xjbc,xkcd,xlda,xijkl->x"
        tensor_l   = [corn[:,0],corn[:,1],corn[:,2],corn[:,3],cent]
        path_final = get_best_path_via_oe(equation,tensor_l,store=path_recorder)
        out        = contract_engine(equation,*tensor_l, optimize=path_final)
    loss   = out.norm()
    if loss.requires_grad:
        loss.backward()
    return corn,f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"
def fullmatrix_test_RG(D=3,B=2,L=7,path_recorder=None,contractor='torch',rd_engine=rd_engine):
    contract_engine = contractor_pool[contractor]
    def get_index(list1,list2):
        x,y=np.meshgrid(list1,list2)
        return y.flatten(),x.flatten()
    #assert L in [2,4,8,16,32,64]
    top_shape_list  = [(D,D,1,1)]  + [(D,D,1,D) for i in range(L-2)] + [(D,1,1,D)]
    mid_shape_list  =[[(D,D,D,1)]+   [(D,D,D,D) for i in range(L-2)] + [(D,1,D,D)] for _ in range(L-2)]
    bot_shape_list  = [(1,D,D,1)]  + [(1,D,D,D) for i in range(L-2)] + [(1,1,D,D)]
    tn2D_shape_list = [top_shape_list]+mid_shape_list+[bot_shape_list]
    tensor_list     = [[rd_engine(B,*l) for l in t] for t in tn2D_shape_list ]

    while True:
        W = len(tensor_list)
        H = len(tensor_list)
        if W<=1:break
        if W%2 ==1:
            last_line_dw = tensor_list.pop()
            last_line_up = tensor_list.pop()
            tensor_list.append([contract_engine("xabcd,xkjal->xkbjcdl",up,dw).flatten(-5,-4).flatten(-2,-1)
                    for up,dw in zip(last_line_up,last_line_dw)])
        if H%2 ==1:
            tensor_list  = list(map(list,zip(*tensor_list)))
            last_line_dw = tensor_list.pop()
            last_line_up = tensor_list.pop()
            tensor_list.append([contract_engine("xabcd,xefgb->xaefcgd",up,dw).flatten(-6,-5).flatten(-3,-2)
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
            equation = "xabcd,xkjal,xefgb,xhiej->xkhficgdl"
            tensor_l = [lu,ld,ru,rd]
            path = get_best_path_via_oe(equation,tensor_l,store=path_recorder)
            out  = contract_engine(equation,*tensor_l, optimize=path)
            out  = out.flatten(3,4).flatten(-4,-3).flatten(1,2).flatten(-2,-1)
            tensor_list.append(out)
        #print(f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M")
        #print([t.shape for t in tensor_list])
        n = W//2
        tensor_list = [tensor_list[i:i + n] for i in range(0, len(tensor_list), n)]
    out = tensor_list[0][0].squeeze()
    loss = out.norm()
    if loss.requires_grad:
        loss.backward()
    return out,f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"
def fullmatrix_test_boundary(D=3,L=7,B=2,W=None,H=None,path_recorder=None,rd_engine=rd_engine):
    # matrix production only, no optimized path
    W = L if W is None else W
    H = L if H is None else H
    def get_index(list1,list2):
        x,y=np.meshgrid(list1,list2)
        return y.flatten(),x.flatten()
    #assert L in [2,4,8,16,32,64]
    top_shape_list  = [(2,D,D,1,1)]  + [(2,D,D,1,D) for i in range(H-2)] + [(2,D,1,1,D)]
    bkl_shape_list  =[[(2,D,D,D,1)]+   [(2,D,D,D,D) for i in range(H-2)] + [(2,D,1,D,D)] for _ in range((W-2)//2)]
    tn2D_shape_list = [top_shape_list]+bkl_shape_list
    if W%2 == 1:
        mid_shape_list   = [  (D,D,D,1)]  + [  (D,D,D,D) for i in range(H-2)] + [(D,1,D,D)]
        tn2D_shape_list += [mid_shape_list]

    tensor_list     = [[rd_engine(B,*l) for l in t] for t in tn2D_shape_list ]
    tensor_list     = tensor_list[::-1]
    tensor          = tensor_list.pop()
    for i in range(1,H//2):
        tensor=[torch.einsum("xzabcd,xzefag->xz e bf c dg",up,dw).flatten(-5,-4).flatten(-2,-1)
                                                        for up,dw in zip(tensor,tensor_list.pop())]
    upper = [t[:,0] for t in tensor]
    lower = [t[:,1] for t in tensor]
    #upper (D,D^n,1,1) (D,D^n,1,D^n) (D,D^n,1,D^n) (D,1,1,D^n)
    #lower (D,D^n,1,1) (D,D^n,1,D^n) (D,D^n,1,D^n) (D,1,1,D^n)
    if len(tensor_list)==1:
        #center (D,D,D,1) (D,D,D,D)  (D,D,D,D) (D,1,D,D)
        upper=[torch.einsum("xabcd,xkjal->x k bj c dl",up,dw).flatten(-5,-4).flatten(-2,-1)
                                                for up,dw in zip(upper,tensor_list.pop())]

    tensor=[torch.einsum("xabcd,xaefg->x f be c dg",up,dw).flatten(-5,-4).flatten(-2,-1)
                                                for up,dw in zip(upper,lower)]
    tensor= [t.squeeze() for t in tensor]
    if B == 1:tensor= [t.unsqueeze(0) for t in tensor]
    out   = tensor[0]
    for t in range(1,len(tensor)-1):
        out = torch.einsum("xa,xab->xb",out,tensor[t])
    out = torch.einsum("xa,xa->x",out,tensor[-1])
    out  = out.squeeze()
    loss = out.norm()
    if loss.requires_grad:
        loss.backward()
    return out,f"{torch.cuda.memory_stats()['reserved_bytes.all.peak']/1024/1024}M"

def fullmatrix_test_directly_oe(**kargs):
    return fullmatrix_test_directly(contractor='oe',**kargs)
def fullmatrix_test_corn_4_oe(**kargs):
    return fullmatrix_test_corn_4(contractor='oe',**kargs)
def fullmatrix_test_RG_oe(**kargs):
    return fullmatrix_test_RG(contractor='oe',**kargs)

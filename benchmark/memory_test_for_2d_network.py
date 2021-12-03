from utils import *
from pytorch_memlab import profile, set_target_gpu,profile_every

def result(D=4,B=1,L=5,O=10):
    top_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [  (D,D)]
    mid_shape_list  =[  [(D,D,D)]+ [(D,D,D,D) for i in range(L-2)] + [(D,D,D)] for _ in range(L-2)]
    bot_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [(D,O,D)]
    tn2D_shape_list = [top_shape_list]+mid_shape_list+[bot_shape_list]

    node_list,sublist_list,outlist =  create_templete_2DTN_tn(tn2D_shape_list)
    path,info = get_optim_path_by_oe_from_tn(node_list)

    tensor_list     = [torch.autograd.Variable(rd_engine(B,*l),requires_grad=True) for t in tn2D_shape_list for l in t]
    assert len(tensor_list)==len(sublist_list)
    operands=[]
    for tensor,sublist in zip(tensor_list,sublist_list):
        operand = [tensor,[...,*sublist]]
        operands+=operand
    operands+= [[...,*outlist]]
    @profile_every(1)
    def func():
        out = torch.einsum(*operands,optimize=path)
        return out
    out = func()
    return out
for D in range(2,20):
    print(f"=============== D={D} ===============")
    torch.cuda.empty_cache()
    out = result(D=D,B=1,L=5,O=10)

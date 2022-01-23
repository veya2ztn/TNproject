from utils import *
from pytorch_memlab import profile, set_target_gpu,profile_every
D=6;B=1;L=5;O=10
for D in [2,3,4,5,6]:
    print(f">>>>>>>>>>>>>>> D={D}")
    top_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [  (D,D)]
    mid_shape_list  =[  [(D,D,D)]+ [(D,D,D,D) for i in range(L-2)] + [(D,D,D)] for _ in range(L-2)]
    bot_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [(D,O,D)]
    tn2D_shape_list = [top_shape_list]+mid_shape_list+[bot_shape_list]

    node_list,sublist_list,outlist =  create_templete_2DTN_tn(tn2D_shape_list)
    path,info = get_optim_path_by_oe_from_tn(node_list)

    #tensor_list     = [torch.autograd.Variable(rd_engine(B,*l),requires_grad=True) for t in tn2D_shape_list for l in t]
    tensor_list     = [rd_engine(B,*l) for t in tn2D_shape_list for l in t]
    assert len(tensor_list)==len(sublist_list)
    operands=[]
    for tensor,sublist in zip(tensor_list,sublist_list):
        operand = [tensor,[...,*sublist]]
        operands+=operand
    operands+= [[...,*outlist]]
    out = torch.einsum(*operands,optimize=path)


    # target = torch.rand_like(out)
    # loss = torch.nn.MSELoss()(out,target)
    #
    # print(out.shape)
    # print(loss)
    # loss.backward()

    # print(torch.cuda.memory_allocated())


    import torch.utils.benchmark as benchmark
    # print("============================")
    # print(benchmark.Timer(
    #     stmt='torch.einsum(*operands, optimize=optimize)',
    #     globals={'operands': operands, 'optimize': path},
    #     sub_label='optimize',
    #     description='torch.einsum opt',
    # ).blocked_autorange())
    # torch.cuda.empty_cache()
    print("============================")
    print(benchmark.Timer(
        stmt='torch.einsum(*operands)',
        globals={'operands': operands},
        sub_label='normal',
        description='torch.einsum opt',
    ).blocked_autorange())

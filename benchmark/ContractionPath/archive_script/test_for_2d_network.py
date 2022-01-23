from utils import *

D=6;B=1;L=5;O=1;D=3
import random
import torch.backends.cudnn as cudnn
seed = 200
random.seed(seed)
np.random.seed(seed)
cudnn.benchmark = False
torch.manual_seed(seed)
cudnn.enabled = True
cudnn.deterministic = True
torch.cuda.manual_seed(seed)

from tensornetwork import contractors

def rd_engine(*x,**kargs):
    x =  torch.randn(*x,device='cuda',**kargs)
    x/=  torch.norm(x).sqrt()
    return x
print(f">>>>>>>>>>>>>>> D={D}")

top_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [  (D,D)]
mid_shape_list  =[  [(D,D,D)]+ [(D,D,D,D) for i in range(L-2)] + [(D,D,D)] for _ in range(L-2)]
bot_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [(D,O,D)]
tn2D_shape_list = [top_shape_list]+mid_shape_list+[bot_shape_list]
node_list,sublist_list,outlist =  create_templete_2DTN_tn(tn2D_shape_list,engine=rd_engine)
ans  = contractors.auto(node_list,output_edge_order=get_all_dangling(node_list))
print(ans.tensor)
#node_list,sublist_list,outlist =  create_templete_2DTN_tn(tn2D_shape_list)
# path,info                      = get_optim_path_by_oe_from_tn(node_list)
# # print(path)
# # print(info)
# #tensor_list     = [torch.autograd.Variable(rd_engine(B,*l),requires_grad=True) for t in tn2D_shape_list for l in t]
# tensor_list     = [rd_engine(B,*l) for t in tn2D_shape_list for l in t]
# assert len(tensor_list)==len(sublist_list)
# operands=[]
# for tensor,sublist in zip(tensor_list,sublist_list):
#     operand = [tensor,[...,*sublist]]
#     operands+=operand
# operands+= [[...,*outlist]]
# out1 = torch.einsum(*operands,optimize=path)
# print(out1)
# out2 = torch.einsum(*operands,optimize=None)
# print(torch.dist(out1,out2))

# target = torch.rand_like(out)
# loss = torch.nn.MSELoss()(out,target)
#
# print(out.shape)
# print(loss)
# loss.backward()

# print(torch.cuda.memory_allocated())

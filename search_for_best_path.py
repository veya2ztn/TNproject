import torch,math
import tensornetwork as tn
import opt_einsum as oe
import json
from models.model_utils import *

json_file ="models/arbitrary_shape_path_recorder.json"
# with open(json_file,'r') as f:
#     path_pool = json.load(f)
# for key in path_pool.keys():
#     path_pool[key]['path'] = oe.paths.ssa_to_linear(path_pool[key]['ssa_path'])
# with open(json_file,'w') as f:
#     json.dump(path_pool,f)
# raise
D=8;O=2;L=W=H=6
LW = W//2
LH = H//2
# top_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [  (D,D)]
# mid_shape_list  =[  [(D,D,D)]+ [(D,D,D,D) for i in range(L-2)] + [(D,D,D)] for _ in range(L-2)]
# bot_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [(D,O,D)]
# tn2D_shape_list = [top_shape_list]+mid_shape_list+[bot_shape_list]
# node_list,sublist_list,outlist =  create_templete_2DTN_tn(tn2D_shape_list)
# tn2D_shape_list                = [ [(D,D,O)]+[  (D,D,D)]*(LH-1) ]+ \
#                                  [ [(D,D,D)]+[(D,D,D,D)]*(LH-1)]*(LW-1)
# node_list,sublist_list,outlist = sub_network_tn(tn2D_shape_list)
# operands = []
# for node in node_list:
#     operands+=[node.tensor,[edge.name for edge in node.edges]]
# operands+= [[edge.name for edge in get_all_dangling(node_list)]]
virtual_bond_config="models/arbitary_shape/arbitary_shape_3.json"
if isinstance(virtual_bond_config,str):
    arbitary_shape_state_dict = torch.load(virtual_bond_config)
else:
    arbitary_shape_state_dict = virtual_bond_config
assert isinstance(arbitary_shape_state_dict,dict)


info_per_group = arbitary_shape_state_dict['node']
info_per_line  = arbitary_shape_state_dict['line']
info_per_point = arbitary_shape_state_dict['element']

center_group    = 0
damgling_num    = len(info_per_group)
info_per_group[center_group]['neighbor'].insert(0,damgling_num)
info_per_line[(center_group,damgling_num)]={'D': 10}

operands = []
sublist_list=[]
outlist  = [list(info_per_line.keys()).index((center_group,damgling_num))]
ranks_list=[]
for group_now in range(len(info_per_group)):
    group_info= info_per_group[group_now]
    neighbors = group_info['neighbor']
    ranks = []
    sublist=[]
    for neighbor_id in neighbors:
        line_tuple = [group_now,neighbor_id]
        line_tuple.sort()
        line_tuple= tuple(line_tuple)
        D = int(info_per_line[line_tuple]['D'])
        idx = list(info_per_line.keys()).index(line_tuple)
        ranks.append(D)
        sublist.append(idx)
    tensor = np.random.randn(*ranks)
    operands+=[tensor,[*sublist]]

    ranks_list.append(ranks)
    sublist_list.append(sublist)
operands+= [[...,*outlist]]
path,info = oe.contract_path(*operands, optimize='optimal')
print(infor)
optimizer=None

if not os.path.exists(json_file):
    path_pool={}
else:
    with open(json_file,'r') as f:
        path_pool = json.load(f)
#shape_array_string = convert_array_shape_to_string(tn2D_shape_list)
shape_array_string = full_size_array_string(*operands)
re_saveQ =True
path_pool[shape_array_string] = path
# optimizer = oe.RandomGreedy(max_time=20, max_repeats=1000)
# for T in [1000,100,10,1,0.1,0.1,0.1]:
#     optimizer.temperature = T
#     path_rand_greedy = oe.contract_path(*operands, optimize=optimizer)
#     print(math.log2(optimizer.best['flops']))
# optimizer.best['path'] = oe.paths.ssa_to_linear(optimizer.best['ssa_path'])
# optimizer.best['outlist']=outlist
# optimizer.best['sublist_list']=sublist_list
# if shape_array_string not in path_pool:
#     path_pool[shape_array_string] = optimizer.best
#     re_saveQ=True
#     print(f"save best:float-({optimizer.best['flops']})")
# else:
#     save_best_now = path_pool[shape_array_string]
#     if optimizer.best['flops'] < save_best_now['flops']:
#         print(f"old best:float-({save_best_now['flops']})")
#         print(f"now best:float-({optimizer.best['flops']})")
#         re_saveQ=True
#         path_pool[shape_array_string] = optimizer.best
#     else:
#         re_saveQ=False
if re_saveQ:
    with open(json_file,'w') as f:
        json.dump(path_pool,f)

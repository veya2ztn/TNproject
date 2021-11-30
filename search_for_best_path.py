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
D=6;O=10;L=6
top_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [  (D,D)]
mid_shape_list  =[  [(D,D,D)]+ [(D,D,D,D) for i in range(L-2)] + [(D,D,D)] for _ in range(L-2)]
bot_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(L-2)] + [(D,O,D)]
tn2D_shape_list = [top_shape_list]+mid_shape_list+[bot_shape_list]
node_list,sublist_list,outlist =  create_templete_2DTN_tn(tn2D_shape_list)
operands = []
for node in node_list:
    operands+=[node.tensor,[edge.name for edge in node.edges]]
operands+= [[edge.name for edge in get_all_dangling(node_list)]]

optimizer = oe.RandomGreedy(max_time=10, max_repeats=1000)
for T in [1000,100,10,1,0.1,0.1,0.1]:
    optimizer.temperature = T
    path_rand_greedy = oe.contract_path(*operands, optimize=optimizer)
    print(math.log2(optimizer.best['flops']))
optimizer.best['path'] = oe.paths.ssa_to_linear(optimizer.best['ssa_path'])

if not os.path.exists(json_file):
    path_pool={}
else:
    with open(json_file,'r') as f:
        path_pool = json.load(f)
shape_array_string = convert_array_shape_to_string(tn2D_shape_list)
if shape_array_string not in path_pool:
    path_pool[shape_array_string] = optimizer.best
    re_saveQ=True
    print(f"save best:float-({optimizer.best['flops']})")
else:
    save_best_now = path_pool[shape_array_string]
    if optimizer.best['flops'] < save_best_now['flops']:
        print(f"old best:float-({save_best_now['flops']})")
        print(f"now best:float-({optimizer.best['flops']})")
        re_saveQ=True
        path_pool[shape_array_string] = optimizer.best
    else:
        re_saveQ=False
if re_saveQ:
    with open(json_file,'w') as f:
        json.dump(path_pool,f)

from .test_functions import *
path_recorder = "path_recorder.json"

import torch.utils.benchmark as benchmark
from tqdm import tqdm
D_list = [2,4,6]
L_list = [3,4,5]
y,x = np.meshgrid(D_list,L_list)
y=y.flatten()
x=x.flatten()
LD_List = [(a,b) for a,b in zip(x,y)]

function_list = [fullmatrix_test_directly,
                 fullmatrix_test_directly_oe,
                 fullmatrix_test_corn_4,
                 fullmatrix_test_corn_4_oe,
                 fullmatrix_test_RG,
                 fullmatrix_test_RG_oe,
                ]

results = []
for L,D in tqdm(LD_List):
    tqdm.write(f"L={L},D={D}")
    for function in function_list:
        try:
            result = benchmark.Timer(
                stmt='cuda_reset();cost=function(D=D,L=L,path_recorder=path_recorder)[1]',
                globals={'D': D, 'L': L,'path_recorder':path_recorder,'cuda_reset':cuda_reset,
                         'function':function},
                sub_label=f"L={L},D={D}",
                description=function.__name__.replace("matrix_test",""),
            ).blocked_autorange()
            results.append(result)
        except:
            pass
compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize(rowwise=True)
compare.print()
# D=10;L=5
# cuda_reset();print(fullmatrix_test_directly(D=D,L=L)[1]);
# cuda_reset();print(fullmatrix_test_corn_4(D=D,L=L)[1]);
# cuda_reset();print(fullmatrix_RG(D=D,L=L)[1]);

from test_functions import *
path_recorder = "path_recorder.json"
import pandas as pd
import torch.utils.benchmark as benchmark
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-D","--D_list", default="3, 4", type=str,help="virtual_bond list")
parser.add_argument("-L","--L_list", default="4, 5", type=str,help="size list")
parser.add_argument("-B","--B_list", default="1, 5", type=str,help="batch size list")
parser.add_argument("-f","--function_list" , default="fullmatrix" , type=str,help="function_list")
parser.add_argument("-o","--output_dir" , default="./result" , type=str,help="output_dir")
parser.add_argument("-m","--mode" , default="memory" , type=str,help="mode")
args = parser.parse_args()


D_list = [int(t) for t in args.D_list.split(',')]
L_list = [int(t) for t in args.L_list.split(',')]
B_list = [int(t) for t in args.B_list.split(',')]

y,x = np.meshgrid(D_list,L_list)
y=y.flatten()
x=x.flatten()
LD_List = [(a,b) for a,b in zip(x,y)]

if args.function_list == "fullmatrix":
    function_list = [
                     fullmatrix_test_directly,
                     fullmatrix_test_directly_oe,
                     fullmatrix_test_corn_4,
                     fullmatrix_test_corn_4_oe,
                     fullmatrix_test_RG,
                     fullmatrix_test_RG_oe,
                     fullmatrix_test_boundary
                    ]
elif args.function_list == "submatrix":
    function_list = [submatrix_test_full_contraction_einsum,
                     submatrix_test_full_contraction_oe,
                     submatrix_test_sequence_contraction_einsum,
                     submatrix_test_sequence_contraction_oe,
                     submatrix_test_recurrent_contraction_einsum,
                     submatrix_test_recurrent_contraction_oe
                    ]
if args.mode == "memory":
    results = {}
    for L,D in tqdm(LD_List):
        for B in tqdm(B_list, leave=False):
            col_name = f"L={L},D={D},B={B}"
            for function in function_list:
                row_name = function.__name__.replace("matrix_test","")
                #tqdm.write(f"{col_name}:{row_name}")
                if row_name not in results:results[row_name]={}

                cuda_reset();
                try:
                    cost = function(D=D,L=L,B=B,path_recorder=path_recorder)[1]
                except:
                    cost = "none"
                results[row_name][col_name]=cost

    data = pd.DataFrame(results)
    data.to_csv(f'result/memory_{args.function_list}.csv')
    print(data)
elif args.mode == "speed":
    results = []
    for L,D in tqdm(LD_List):
        for B in tqdm(B_list, leave=False):
            for function in function_list:

                result = benchmark.Timer(
                    stmt='cuda_reset();cost=function(D=D,L=L,B=B,path_recorder=path_recorder)[1]',
                    globals={'D': D, 'L': L,'B':B,'path_recorder':path_recorder,'cuda_reset':cuda_reset,
                             'function':function},
                    sub_label=f"L={L},D={D},B={B}",
                    description=function.__name__.replace("matrix_test",""),
                ).blocked_autorange()
                results.append(result)

    compare = benchmark.Compare(results)
    compare.trim_significant_figures()
    compare.colorize(rowwise=True)
    compare.print()
    with open(f'result/speed_{args.function_list}.txt','w') as f:
        f.write(compare.__str__())

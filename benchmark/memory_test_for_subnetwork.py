from test_functions import *
path_recorder = "path_recorder.json"
import pandas as pd
import torch.utils.benchmark as benchmark
from tqdm import tqdm
D_list = [2,4,6,8,10,12]
L_list = [3,4,5,6,7,8]
y,x = np.meshgrid(D_list,L_list)
y=y.flatten()
x=x.flatten()
LD_List = [(a,b) for a,b in zip(x,y)]

function_list = [submatrix_test_full_contraction_einsum,
                 submatrix_test_full_contraction_oe,
                 submatrix_test_sequence_contraction_einsum,
                 submatrix_test_sequence_contraction_oe,
                 submatrix_test_recurrent_contraction_einsum,
                 submatrix_test_recurrent_contraction_oe
                ]


results = {}
for L,D in tqdm(LD_List):
    col_name = f"L={L},D={D}"
    for function in tqdm(function_list):
        row_name = function.__name__.replace("matrix_test","")
        tqdm.write(f"{col_name}:{row_name}")
        if row_name not in results:results[row_name]={}
        cuda_reset();
        try:
            cost = function(D=D,L=L,path_recorder=path_recorder)[1]
        except:
            cost = "none"
        results[row_name][col_name]=cost

data = pd.DataFrame(results)
print(data)

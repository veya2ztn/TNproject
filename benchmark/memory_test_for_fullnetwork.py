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

function_list = [fullmatrix_test_directly,
                 fullmatrix_test_directly_oe,
                 fullmatrix_test_corn_4,
                 fullmatrix_test_corn_4_oe,
                 fullmatrix_test_RG,
                 fullmatrix_test_RG_oe,
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

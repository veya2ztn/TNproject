import torch
import torch.nn as nn
import torch.nn.functional as F
from tensornetwork import contractors
import tensornetwork as tn
from tqdm import tqdm
from .mps import MPSLinear_Base,MPSLinear_einsum_arbitary_shape
tn.set_default_backend("pytorch")
class MPSLinear_tn_Base(MPSLinear_Base):
    def __init__(self, in_features,out_features,
                                       in_physics_bond = 2, out_physics_bond=1, virtual_bond_dim=2,
                                       bias=True,label_position='center',init_std=1e-10,contraction_mode = 'recursion'):
        super().__init__()
        if label_position is 'center':
            label_position = in_features//2
        assert type(label_position) is int
        if isinstance(virtual_bond_dim,int):
            virtual_bond_dim = [virtual_bond_dim]*(in_features)
        assert isinstance(virtual_bond_dim,list)
        assert len(virtual_bond_dim)==in_features
        self.in_features   = in_features
        self.out_features  = out_features
        self.ipb           = in_physics_bond
        self.opb           = out_physics_bond
        self.hn            = label_position
        left_num           = self.hn
        right_num          = in_features - left_num

        mps_var    =([self.rde1((self.ipb,virtual_bond_dim[0]),init_std)] +
                     [self.rde1((virtual_bond_dim[i-1],self.ipb,virtual_bond_dim[i]),init_std) for i in range(1,self.hn)] +
                     [self.rde1((virtual_bond_dim[self.hn-1],self.out_features,virtual_bond_dim[self.hn]),init_std)]+
                     [self.rde1((virtual_bond_dim[i-1],self.ipb,virtual_bond_dim[i]),init_std) for i in range(self.hn+1,int(in_features))]+
                     [self.rde1((virtual_bond_dim[-1],self.ipb),init_std)]
                     )
        assert len(mps_var)-1 == int(in_features)
        self.mps_var = [nn.Parameter(v) for v in mps_var]
        self.center  = left_num
        for i, v in enumerate(self.mps_var):
            self.register_parameter(f'mps{i}', param=v)

        self.contraction_mode = contraction_mode

    @staticmethod
    def tensor_network_with_single_input_pure_tn(mps_var,single_batch,center):
        assert len(single_batch) == len(mps_var)-1
        mps_list_1   = mps_var
        mps_nodes_1  = [tn.Node(v, name=f"t{i}") for i,v in enumerate(mps_list_1)]
        mps_edges_1  = [mps_nodes_1[i][-1]^mps_nodes_1[i+1][0] for i in range(len(mps_nodes_1)-1)]
        inp_nodes    = [tn.Node(v, name=f"i{i}") for i,v in enumerate(single_batch)]
        for i,input_node in enumerate(inp_nodes):
            j = i if i < center else i+1
            mps_physicd_edge = mps_nodes_1[j][0] if j==0 else mps_nodes_1[j][1]
            inp_physics_edge = input_node[0]
            tn.connect(mps_physicd_edge,inp_physics_edge,name=f"p_{i}")
        node_list         = mps_nodes_1+inp_nodes
        output_edge_order = [mps_nodes_1[center][1]]
        return node_list,output_edge_order

    @staticmethod
    def tensor_network_with_single_input_partial_tn(mps_var,single_batch,center):
        assert len(single_batch) == len(mps_var)-1
        matrix_train = []
        for i, inp in enumerate(single_batch):
            j = i if i < center else i+1
            if i == center:matrix_train.append(mps_unit)
            mps_unit = mps_var[j]
            if j == 0:
                matrix_train.append(torch.einsum('p,pd->d', inp , mps_unit))
            elif j== len(mps_var)-1:
                matrix_train.append(torch.einsum('p,dp->d', inp , mps_unit))
            else:
                matrix_train.append(torch.einsum('p,apb->ab', inp , mps_unit))

        mps_list_1   = matrix_train
        mps_nodes_1  = [tn.Node(v, name=f"t{i}") for i,v in enumerate(mps_list_1)]
        mps_edges_1  = [mps_nodes_1[i][-1]^mps_nodes_1[i+1][0] for i in range(len(mps_nodes_1)-1)]

        node_list         = mps_nodes_1
        output_edge_order = [mps_nodes_1[center][1]]
        return node_list,output_edge_order

    @staticmethod
    def tensor_network_with_batch_input(mps_var,inputs,center):
        mps_nodes  = [tn.Node(v, name=f"t{i}") for i,v in enumerate(mps_var)]
        # (P,D) <-> (D,P,D) <-> (D,P,D) <-> (D,P,D) <-> (D,P)
        for i in range(len(mps_var)-1):
            tn.connect(mps_nodes[i][-1],mps_nodes[i+1][0],name=f"mps:{i}<->{i+1}")
        # (P,B) <-> (B,P,B) <-> (B,P,B) <-> (B,P,B) <-> (B,P,B)
        inp_nodes=[tn.Node(v, name=f"i{i}") for i,v in enumerate(inputs)]
        for i in range(len(inp_nodes)-1):
            tn.connect(inp_nodes[i][-1],inp_nodes[i+1][0],name=f"inp:{i}<->{i+1}")

        for i,input_node in enumerate(inp_nodes):
            j = i if i < center else i+1
            edge_axis = 0 if j==0 else 1
            tn.connect(mps_nodes[j][edge_axis],input_node[edge_axis],name=f"phy_{i}")
        #  (P,D) <-> (D,P,D) <-> (D,P,D) <-> (D,P,D) <-> (D,P)
        #  |           |           |           |           |
        # (P,B) <-> (B,P,B) <-> (B,P,B) <-> (B,P,B) <-> (B,P,B)
        node_list         = mps_nodes+inp_nodes
        output_edge_order = [inp_nodes[-1][2],mps_nodes[self.center][1]]
        return node_list,output_edge_order

    @staticmethod
    def opt_contract_result(node_list,output_edge_order=[],info=None):
        if info is None:
            for edge in get_all_edges(node_list):
                if edge.name == '__unnamed_edge__':
                    if edge.node1 is not None and edge.node2 is not None:
                        edge.name= f'{edge.node1.name}:{edge.axis1}<->{edge.node2.name}:{edge.axis2}'
                    else:
                        edge.name= f"{edge.node1}:{edge.axis1}"
            operands = []
            for node in node_list:
                operands+=[node.tensor,[edge.name for edge in node.edges]]
                # print(node.name)
                # print([edge.name for edge in node.edges])
            operands+= [[edge.name for edge in output_edge_order]]
            # print([edge.name for edge in output_set])
            # print("=============")
            # raise
            path,info = oe.contract_path(*operands)


        small_cores =[node.tensor for node in node_list]
        sf = ctg.SliceFinder(info, target_size=2**28)
        inds_to_slice, cost_of_slicing = sf.search()
        # def contracte_via_sf(*small_cores):
        #     sc  = sf.SlicedContractor([*small_cores])
        #     ans = sum(sc.contract_slice(i) for i in tqdm.trange(sc.nslices))
        #     return ans
        # ans=checkpoint(contracte_via_sf,*small_cores)
        sc  = sf.SlicedContractor([*small_cores])
        # for i in range(sc.nslices):
        #     print(sc.contract_slice(i).shape)
        # raise
        ans = sum(sc.contract_slice(i) for i in range(sc.nslices))
        return ans

class MPSLinear_tn_loop(MPSLinear_tn_Base):
    '''
    For s simplest MPSLayer(in_features: int, out_features: int,
                            in_physics_bond: int,
                            out_physics_bond: int, virtual_bond_dim:int,
                            bias: bool = True, label_position: int or str):
        input  (B, in_features , in_physics_bond)
        output (B, out_features)
    '''
    path    = None
    def forward(self, inputs):
        out =[]
        for single_input in tqdm(inputs):
            node_list,output_edge_order = self.tensor_network_with_single_input_partial_tn(self.mps_var,single_input,self.hn)
            ans,path = contractors.auto(node_list,output_edge_order=output_edge_order,path=self.path)
            self.path = path
            out.append(ans.tensor)
        out = torch.stack(out)
        return out

import cotengra as ctg
import opt_einsum as oe
from tensornetwork.contractors.opt_einsum_paths.utils import get_subgraph_dangling,get_all_edges

class MPSLinear_tn_batch(MPSLinear_tn_Base):
    path = None
    def forward(self, inputs):
        inputs    = self.batch_together(inputs)
        node_list,output_edge_order = self.tensor_network_with_batch_input(self.mps_var,inputs,self.hn)
        #ans = opt_contract_result(mps_nodes+inp_nodes,output_edge_order=[inp_nodes[-1][2],mps_nodes[self.center][1]])
        ans,path  = contractors.auto(node_list,output_edge_order=output_edge_order,path=self.path)
        self.path = path
        return ans.tensor

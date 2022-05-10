# this script contain the old function like
#   PEPS_uniform_shape_symmetry_any_old,
#   PEPS_uniform_shape_symmetry_deep_model_old
# that use in early stage investigation.
from .tensornetwork_base import *
from .model_utils import *
import opt_einsum as oe
import numpy as np

class PEPS_uniform_shape_symmetry_any_old(TN_Base):
    # this module is compatible for early training result for LinearCombineModel2
    def __init__(self,  W=6,H=6,out_features=10,
                       in_physics_bond = 2, virtual_bond_dim=2,
                       init_std=1e-10):
        super().__init__()
        assert (W % 2 == 0) and (H % 2 == 0) and (W == H)
        self.W             = W
        self.H             = H
        self.out_features  = out_features
        O                  = np.power(16,1/4)
        assert np.ceil(O) == np.floor(O)
        self.O             = O = O.astype('uint')
        self.D             = D = virtual_bond_dim
        self.P             = P = in_physics_bond

        self.corn_tensors = nn.Parameter(self.rde2D( (4,O,P,D,D),init_std, offset=3))
        self.edge_tensors = nn.Parameter(self.rde2D( (2*(W-2)+2*(H-2),P,D,D,D),init_std, offset=2))
        self.bulk_tensors = nn.Parameter(self.rde2D( ((W-2)*(H-2),P,D,D,D,D),init_std, offset=2))
        W = self.W;
        H = self.H
        O = self.O
        P = self.P
        D = self.D
        self.LW = LW = W//2
        self.LH = LH = H//2
        tn2D_shape_list                = [ [(D,D,O)]+[  (D,D,D)]*(LH-1) ]+ \
                                         [ [(D,D,D)]+[(D,D,D,D)]*(LH-1)]*(LW-1)
        path,sublist_list,outlist = get_best_path(tn2D_shape_list,store=path_recorder,type='sub')
        #node_list,sublist_list,outlist = sub_network_tn(tn2D_shape_list)
        #path,info                      = get_optim_path_by_oe_from_tn(node_list)
        last_idx = outlist.pop()
        outlist.insert(LH,last_idx)
        #print(outlist)
        #outlist should be [2, 6, 12, 18, 13, 15, 17] for 3x3
        self.sublist_list = sublist_list
        self.outlist      = outlist
        self.path         = path

        self.path_record  = {}
        self.path_final= None
    # def flatten_image_input(self,batch_image_input):
    #     bulk_input = batch_image_input[...,1:-1,1:-1,:].flatten(1,2)
    #     edge_input = torch.cat([batch_image_input[...,0,1:-1,:],
    #                             batch_image_input[...,1:-1,[0,-1],:].flatten(-3,-2),
    #                             batch_image_input[...,-1,1:-1,:]
    #                            ],1)
    #     corn_input = batch_image_input[...,[0,0,-1],[0,-1,0],:]
    #     cent_input = batch_image_input[...,-1,-1,:]
    #     return bulk_input,edge_input,corn_input,cent_input
    def get_batch_contraction_network(self,input_data):
        bulk_input,edge_input,corn_input = self.flatten_image_input(input_data)
        bulk_tensors = self.einsum_engine("lpabcd,klp->lkabcd",self.bulk_tensors,bulk_input)
        edge_tensors = self.einsum_engine(" lpabc,klp->lkabc" ,self.edge_tensors,edge_input)
        corn_tensors = self.einsum_engine("  lopab,klp->lkabo" ,self.corn_tensors,corn_input)
        return bulk_tensors,edge_tensors,corn_tensors
    def forward(self,input_data):
        # bulk_input,edge_input,corn_input,cent_input = self.flatten_image_input(input_data)
        # corn_input   = torch.cat([corn_input,cent_input.unsqueeze(1)],1)
        # bulk_tensors = self.einsum_engine("lpabcd,klp->lkabcd",self.bulk_tensors,bulk_input)
        # edge_tensors = self.einsum_engine(" lpabc,klp->lkabc" ,self.edge_tensors,edge_input)
        # corn_tensors = self.einsum_engine("  lopab,klp->lkabo" ,self.corn_tensors,corn_input)
        bulk_tensors,edge_tensors,corn_tensors = self.get_batch_contraction_network(input_data)
        L = len(bulk_tensors)
        remain = bulk_tensors.shape[1:]
        bulk_tensors = bulk_tensors.reshape(4,L//4,*remain)

        L = len(edge_tensors)
        remain = edge_tensors.shape[1:]
        edge_tensors = edge_tensors.reshape(4,L//4,*remain)

        LH = self.LH
        LW = self.LW
        tensor_list  =[[corn_tensors]        + list(edge_tensors[:LH-1]) ]+\
                      [[edge_tensors[LH-1+i]]+ list(bulk_tensors[(LH-1)*i:(LH-1)*(i+1)])
                                                                    for i in range(LW-1)]
        tensor_list     = [l for t in tensor_list for l in t]
        assert len(tensor_list)==len(self.sublist_list)
        operands=[]
        for tensor,sublist in zip(tensor_list,self.sublist_list):
            operand = [tensor,[...,*sublist]]
            operands+=operand
        operands+= [[...,*(self.outlist)]]

        # in this case W==H , LW==LH

        quater_contraction = self.einsum_engine(*operands,optimize=self.path).flatten(-2*LW,-LW-1).flatten(-LW,-1)

        tensor  = self.einsum_engine("lkmab,lknbc->lkmnac",
                              quater_contraction[[0,2]],quater_contraction[[1,3]]
                              ).flatten(-4,-3)# -> (2,B,O^2,D^3,D^3)
        tensor  = self.einsum_engine("kmab,knba->kmn",
                              tensor[0],tensor[1]
                              ).flatten(-2,-1)# -> (B,O^4)
        return tensor

class PEPS_uniform_shape_symmetry_base_old(TN_Base):
    def __init__(self, W=6,H=6,out_features=16,
                       in_physics_bond = 3, virtual_bond_dim=3,
                       init_std=1e-10):
        super().__init__()
        assert (W == H)
        self.W             = W
        self.H             = H
        self.out_features  = out_features
        O                  = np.power(out_features,1/4)
        assert np.ceil(O) == np.floor(O)
        self.O             = O = O.astype('uint')
        self.D             = D = virtual_bond_dim
        self.P             = P = in_physics_bond
        self.LW = LW =  int(np.ceil(1.0*W/2))
        self.LH = LH = int(np.floor(1.0*H/2))
        self.corn_tensors = nn.Parameter(self.rde2D( (4,O,P,D,D),init_std, offset=3))
        self.edge_tensors = nn.Parameter(self.rde2D( (2*(W-2)+2*(H-2),P,D,D,D),init_std, offset=2))
        self.bulk_tensors = nn.Parameter(self.rde2D( ((W-2)*(H-2),P,D,D,D,D),init_std, offset=2))

        self.index_matrix = index_matrix = torch.LongTensor([[[i,j] for j in range(W)] for i in range(H)])
        bulk_index,edge_index,corn_index=self.flatten_image_input(index_matrix)
        part_lu_idex = torch.rot90(index_matrix,k=0)[:LW,:LH].flatten(0,1).transpose(1,0)
        part_ru_idex = torch.rot90(index_matrix,k=1)[:LW,:LH].flatten(0,1).transpose(1,0)
        part_rd_idex = torch.rot90(index_matrix,k=2)[:LW,:LH].flatten(0,1).transpose(1,0)
        part_ld_idex = torch.rot90(index_matrix,k=3)[:LW,:LH].flatten(0,1).transpose(1,0)

        flag_matrix = torch.zeros(W,H).long()
        position_matrix = torch.zeros(W,H).long()

        for n,(i,j) in enumerate(corn_index.numpy()):
            flag_matrix[i,j]=0
            position_matrix[i,j]=n
        for n,(i,j) in enumerate(edge_index.numpy()):
            flag_matrix[i,j]=1
            position_matrix[i,j]=n
        for n,(i,j) in enumerate(bulk_index.numpy()):
            flag_matrix[i,j]=2
            position_matrix[i,j]=n

        self.indexrule = torch.stack([
                               position_matrix[part_lu_idex[0],part_lu_idex[1]],
                               position_matrix[part_ru_idex[0],part_ru_idex[1]],
                               position_matrix[part_rd_idex[0],part_rd_idex[1]],
                               position_matrix[part_ld_idex[0],part_ld_idex[1]],
                               ]).transpose(1,0)

        self.partrule        = flag_matrix[part_lu_idex[0],part_lu_idex[1]]
        self.flag_matrix     = flag_matrix
        self.position_matrix = position_matrix
        self.cent_tensor_idx = position_matrix[self.W//2,self.H//2] if self.W%2==1 else None

    def get_batch_contraction_network(self,input_data):
        bulk_input,edge_input,corn_input = self.flatten_image_input(input_data)
        bulk_tensors = self.einsum_engine("lpabcd,klp->lkabcd",self.bulk_tensors,bulk_input)
        edge_tensors = self.einsum_engine(" lpabc,klp->lkabc" ,self.edge_tensors,edge_input)
        corn_tensors = self.einsum_engine(" lopab,klp->lkoab" ,self.corn_tensors,corn_input)
        return bulk_tensors,edge_tensors,corn_tensors

    def weight_init(self,method=None,set_var=1):
        if method is None:return
        if method == "Expecatation_Normalization":
            W          = self.W
            H          = self.H
            D          = self.D
            P          = self.P
            num_of_edge= (W-1)*H+(H-1)*W
            num_of_unit= W*H
            solved_var = np.power(set_var,1/num_of_unit)*np.power(1/D,num_of_edge/(num_of_unit))/P
            solved_std = np.sqrt(solved_var)
            self.corn_tensors = torch.nn.init.normal_(self.corn_tensors,mean=0.0, std=solved_std)
            self.edge_tensors = torch.nn.init.normal_(self.edge_tensors,mean=0.0, std=solved_std)
            self.bulk_tensors = torch.nn.init.normal_(self.bulk_tensors,mean=0.0, std=solved_std)
        else:
            raise NotImplementedError

    @staticmethod
    def pick_tensors(partrule,indexrule,corn_tensors,edge_tensors,bulk_tensors):
        tensor_list = []
        for part,idxes in zip(partrule,indexrule):
            if   part == 0:tensor_list.append(corn_tensors[idxes])
            elif part == 1:tensor_list.append(edge_tensors[idxes])
            elif part == 2:tensor_list.append(bulk_tensors[idxes])
            else:raise NotImplezntedError
        return tensor_list

class PEPS_uniform_shape_symmetry_deep_model_old(PEPS_uniform_shape_symmetry_base_old):
    def __init__(self, nonlinear_layer=nn.Tanh(),
                       normlized_layer_module=nn.InstanceNorm3d,
                       init_std=1e-10,normlized=True, set_var=1,**kargs):
        super().__init__(**kargs)
        H = self.H
        W = self.W
        LW= self.LW
        LH= self.LH
        D = self.D
        O = self.O
        P = self.P
        if normlized:
            self.weight_init(method="Expecatation_Normalization",set_var=set_var)

        flag_matrix     = self.flag_matrix
        position_matrix = self.position_matrix
        index_matrix    = self.index_matrix
        part_lu_idex = torch.rot90(index_matrix,k=0)[:LW,:LH]
        part_ru_idex = torch.rot90(index_matrix,k=1)[:LW,:LH]
        part_rd_idex = torch.rot90(index_matrix,k=2)[:LW,:LH]
        part_ld_idex = torch.rot90(index_matrix,k=3)[:LW,:LH]
        part_idex = torch.stack([part_lu_idex,
                                 part_ru_idex,
                                 part_rd_idex,
                                 part_ld_idex],-2)

        indexrules = []
        partrules  = []
        point_x = [0,0,1,1]
        point_y = [0,1,1,0]
        p       = part_idex[point_x,point_y]#(L,4,2)
        indexrule=position_matrix[p[...,0],p[...,1]]
        partrule =flag_matrix[p[...,0],p[...,1]][:,0]

        indexrules.append(indexrule)
        partrules.append(partrule)
        edge_contraction_path=[]
        for L in range(2,LW):
            indexrule={}
            partrule={}
            tn2D_shape_list = [[(D,D,D)]+[(D,D,D,D)]*(L-1)]
            path,sublist_list,outlist = get_best_path(tn2D_shape_list,store=path_recorder,type='sub')
            edge_contraction_path.append([path,sublist_list,outlist])
            point_x = [[j for j in range(L)],[L for j in range(L)]]
            point_y = [[L for j in range(L)],[j for j in range(L)]]
            p       = part_idex[point_x,point_y]#(2,L,4,2)
            indexrule['edge']= position_matrix[p[...,0],p[...,1]].transpose(0,1)
            partrule['edge'] = flag_matrix[p[...,0],p[...,1]][0][:,0]

            point_x = [L]
            point_y = [L]
            p       = part_idex[point_x,point_y]#(L,4,2)

            indexrule['cent']= position_matrix[p[...,0],p[...,1]]
            partrule['cent'] = flag_matrix[p[...,0],p[...,1]][:,0]

            indexrules.append(indexrule)
            partrules.append(partrule)
        self.indexrules = indexrules
        self.partrules  = partrules
        self.edge_contraction_path = edge_contraction_path
        self.nonlinear_layer = nonlinear_layer
        self.normlized_layers = nn.ModuleList([normlized_layer_module(O,affine=True) for _ in self.partrules])

    def forward(self,input_data):
        LH = self.LH
        LW = self.LW
        D  = self.D
        bulk_tensors,edge_tensors,corn_tensors = self.get_batch_contraction_network(input_data)
        corn_tensors = self.pick_tensors(self.partrules[0],self.indexrules[0],corn_tensors,edge_tensors,bulk_tensors)
        corn = self.einsum_engine("lkoab,lkcdb,lkefcg,lkgah->lkohedf",*corn_tensors).flatten(-4,-3).flatten(-2,-1)
        corn = self.nonlinear_layer(corn)# (4,B,O,D,D)
        corn = self.normlized_layers[0](corn.permute(1,2,0,3,4)).permute(2,0,1,3,4)
        for i,(partrule, indexrule) in enumerate(zip(self.partrules[1:],self.indexrules[1:])):
            path,sublist_list,outlist = self.edge_contraction_path[i]
            edge_tensors= self.pick_tensors(partrule['edge'],indexrule['edge'],corn_tensors,edge_tensors,bulk_tensors)
            cent_tensor = self.pick_tensors(partrule['cent'],indexrule['cent'],corn_tensors,edge_tensors,bulk_tensors)[0]
            L           = len(edge_tensors)
            operands    = structure_operands(edge_tensors,sublist_list,outlist)
            edge1,edge2 = self.einsum_engine(*operands,optimize=path).flatten(-L-L,-L-1).flatten(-L,-1)
            corn = self.einsum_engine("lkoab,lkcdb,lkefcg,lkgah->lkohedf",corn ,edge1,cent_tensor,edge2).flatten(-4,-3).flatten(-2,-1)
            corn = self.nonlinear_layer(corn)
            corn = self.normlized_layers[i+1](corn.permute(1,2,0,3,4)).permute(2,0,1,3,4)
        # corn now is a tensor (B,4,D^(L/2),D^(L/2))
        corn   = corn/D**(LW/3)# basicly should use LW/4 but use LW/3 to avoid gradient vanish
        corn   = self.einsum_engine("kvab,kxbc,kycd,kzda->kvxyz",*corn).flatten(-4,-1)# -> (B,O^4)
        return corn

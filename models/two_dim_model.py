from models.tensornetwork_base import *
import numpy as np
import tensornetwork as tn
tn.set_default_backend("numpy")
import opt_einsum as oe
from tensornetwork.network_components import get_all_nondangling,get_all_dangling
from tensornetwork.contractors.opt_einsum_paths.utils import get_subgraph_dangling,get_all_edges
def create_templete_2DTN_tn(tn2D_shape_list):
    node_array      = []
    W = len(tn2D_shape_list)
    H = len(tn2D_shape_list[0])
    for i in range(W):
        node_line = []
        for j in range(H):
            node = tn.Node(np.random.randn(*tn2D_shape_list[i][j]),name=f"{i}-{j}")
            node_line.append(node)
        node_array.append(node_line)
    #row
    for i in range(W-1):
        for j in range(H-1):
            tn.connect(node_array[i][j][1],node_array[i][j+1][-1],name=f'{i},{j}|{i},{j+1}')
    #last row
    i=W-1
    for j in range(H-1):
            tn.connect(node_array[i][j][0],node_array[i][j+1][-1],name=f'{i},{j}|{i},{j+1}')
    #col
    for j in range(H-1):
        for i in range(W-2):
            tn.connect(node_array[i][j][0],node_array[i+1][j][2],name=f'{i},{j}|{i+1},{j}')
        i=W-2
        tn.connect(node_array[i][j][0],node_array[i+1][j][1],name=f'{i},{j}|{i+1},{j}')
    j = H-1
    for i in range(W-2):
        tn.connect(node_array[i][j][0],node_array[i+1][j][1],name=f'{i},{j}|{i+1},{j}')
    i=W-2
    tn.connect(node_array[i][j][0],node_array[i+1][j][0],name=f'{i},{j}|{i+1},{j}')
    node_list = [item for sublist in node_array for item in sublist]
    for edge in get_all_edges(node_list):
        if edge.name == '__unnamed_edge__':
            if edge.node1 is not None and edge.node2 is not None:
                edge.name= f'{edge.node1.name}:{edge.axis1}<->{edge.node2.name}:{edge.axis2}'
            else:
                edge.name= f"{edge.node1}:{edge.axis1}"
    class edges_name_mapper:
        name_to_idx = {}
        def get_index(self,name):
            if name not in self.name_to_idx:
                self.name_to_idx[name]=len(self.name_to_idx)
            return self.name_to_idx[name]
    mapper = edges_name_mapper()
    sublist_list = [[mapper.get_index(e.name)for e in t.edges] for t in node_list]
    outlist = [mapper.get_index(e.name) for e in get_all_dangling(node_list)]
    return node_list,sublist_list,outlist
def get_optim_path_by_oe_from_tn(node_list):
    operands = []
    for node in node_list:
        operands+=[node.tensor,[edge.name for edge in node.edges]]
    operands+= [[edge.name for edge in get_all_dangling(node_list)]]
    path,info = oe.contract_path(*operands)
    return path,info

class PEPS_einsum_uniform_shape(TN_Base):
    def __init__(self, W,H,out_features,
                       in_physics_bond = 2, virtual_bond_dim=2,
                       bias=True,label_position='center',init_std=1e-10,contraction_mode = 'recursion'):
        super().__init__()
        #label_position at 'corner':
        label_pos_x        = W
        label_pos_y        = H
        self.W             = W
        self.H             = H

        self.out_features  = O = out_features
        self.vbd           = D = virtual_bond_dim
        self.ipb           = P = in_physics_bond
        self.label_pos_x   = label_pos_x
        self.label_pos_y   = label_pos_y

        self.bulk_tensors = nn.Parameter(self.rde2D((     (W-2)*(H-2),P,D,D,D,D),init_std))
        self.edge_tensors = nn.Parameter(self.rde2D( (2*(W-2)+2*(H-2),P,D,D,D),init_std))
        self.corn_tensors = nn.Parameter(self.rde2D(                 (3,P,D,D),init_std))
        self.cent_tensors = nn.Parameter(self.rde2D(                 (O,P,D,D),init_std))
    @staticmethod
    def rde2D(shape,init_std,offset=2):
        size_shape = shape[:offset]
        bias_shape = shape[offset:]
        if len(bias_shape) ==2 :
            bias_mat   = torch.eye(*bias_shape)
        elif len(bias_shape) == 3:
            a,b,c   = bias_shape
            bias_mat   = torch.kron(torch.ones(a),torch.eye(b,c)).reshape(a,b,c)
        elif len(bias_shape) == 4:
            a,b,c,d   = bias_shape
            bias_mat   = torch.kron(torch.ones(a,b),torch.eye(c,d)).reshape(a,b,c,d)

        bias_mat   = bias_mat.repeat(*size_shape,*([1]*len(bias_shape)))
        tensor     = init_std * torch.randn(*shape)+ bias_mat
        return tensor

    def mpo_line(self,i,bulk_tensors,edge_tensors,corn_tensors,cent_tensors):
        W=self.W
        H=self.H
        if i == 0:
            return  corn_tensors[0],edge_tensors[0:W-2],corn_tensors[1]
        elif i == H - 1:
            return  corn_tensors[2],edge_tensors[-(W-2):],cent_tensors
        else:
            return  edge_tensors[W-4+2*i],bulk_tensors[(W-2)*(i-1):(W-2)*i],edge_tensors[W-4+2*i+1]


    @staticmethod
    def batch_contract_mps_mpo(mps_list,mpo_list):
        # mps_list                                    --D--|--D--
        # (D,D)-(D,D,D)-(D,D,D)-...-(D,D,D)-(D,D)          D
        #  -b-    -c-  -b-   -c-  -b-     -b-
        # |a         |a         |a          |a
        # the order i.e. 'abcd' counterclockwise and start from the down index. (so down index must a )
        #  mpo_list
        # (P,D,P)-(D,P,D,P)-(D,P,D,P)-...-(D,P,D,P)-(D,P,P)
        #  |c            |c        |c           |b
        #   -b-      -d-  -b-  -d-  -b-     -c-
        # |a            |a        |a           |a
        assert len(mps_list) > 2
        assert len(mpo_list) > 2
        stack_unit_mps = (len(mps_list)==3 and len(mps_list[0].shape)+ 2 == len(mps_list[1].shape))
        stack_unit_mpo = (len(mpo_list)==3 and len(mpo_list[0].shape)+ 2 == len(mpo_list[1].shape))
        mps_left,mps_rigt = mps_list[0],mps_list[-1]
        mpo_left,mpo_rigt = mpo_list[0],mpo_list[-1]
        new_mps_list= []
        tensor = torch.einsum("  kab,kcda->kcbd",mps_left,mpo_left).flatten(-2,-1)
        new_mps_list.append(tensor)
        if stack_unit_mps and stack_unit_mpo:
            mps_inne = mps_list[1]
            mpo_inne = mpo_list[1]
            tensor = torch.einsum("lkabc,lkdeaf->lkdbecf",mps_inne,mpo_inne).flatten(-4,-3).flatten(-2,-1)
            new_mps_list.append(tensor)
        else:
            if stack_unit_mps:mps_inne = list(*mps_list[1:-1])
            if stack_unit_mpo:mpo_inne = list(*mpo_list[1:-1])
            for mps,mpo in zip(mps_inne,mpo_inne):
                tensor =torch.einsum("kabc,kdeaf->kdebcf",mps,mpo).flatten(-4,-3).flatten(-2,-1)
                new_mps_list.append(tensor)
        tensor = torch.einsum("  kab,kcad->kcbd",mps_rigt,mpo_rigt).flatten(-2,-1)
        new_mps_list.append(tensor)
        return new_mps_list

    @staticmethod
    def batch_contract_mps_mps(mps_list,mpo_list):
        # mps_list                                    --D--|--D--
        # (D,D)-(D,D,D)-(D,D,D)-...-(D,D,D)-(D,D)          D
        # (D,D)-(D,D,D)-(D,D,D)-...-(D,D,D)-(D,D,O)
        #  -b-    -c-  -b-   -c-  -b-     -b-
        # |a         |a         |a          |a
        #
        # |b            |b        |b        |b
        #  -a-      -c-  -a-  -c-  -a-   -c- -a
        assert len(mps_list) > 2
        assert len(mpo_list) > 2
        stack_unit_mps = (len(mps_list)==3 and len(mps_list[0].shape)+ 2 == len(mps_list[1].shape))
        stack_unit_mpo = (len(mpo_list)==3 and len(mpo_list[0].shape)+ 2 == len(mpo_list[1].shape))
        mps_left,mps_rigt = mps_list[0],mps_list[-1]
        mpo_left,mpo_rigt = mpo_list[0],mpo_list[-1]
        new_mps_list= []
        tensor = torch.einsum("  kab,kca->kbc",mps_left,mpo_left).flatten(-2,-1)
        new_mps_list.append(tensor)
        if stack_unit_mps and stack_unit_mpo:
            mps_inne = mps_list[1]
            mpo_inne = mpo_list[1]
            tensor = torch.einsum("lkabc,lkdae->lkbdce",mps_inne,mpo_inne).flatten(-4,-3).flatten(-2,-1)
            new_mps_list.append(tensor)
        else:
            if stack_unit_mps:mps_inne = list(*mps_list[1:-1])
            if stack_unit_mpo:mpo_inne = list(*mpo_list[1:-1])
            for mps,mpo in zip(mps_inne,mpo_inne):
                tensor =torch.einsum("kabc,kdae->kbdce",mps,mpo).flatten(-4,-3).flatten(-2,-1)
                new_mps_list.append(tensor)
        tensor = torch.einsum("  kab,okac->kobc",mps_rigt,mpo_rigt).flatten(-2,-1)
        new_mps_list.append(tensor)
        return new_mps_list

    @staticmethod
    def flatten_image_input(batch_image_input):
        bulk_input = batch_image_input[...,1:-1,1:-1,:].flatten(1,2)
        edge_input = torch.cat([batch_image_input[...,0,1:-1,:],
                                batch_image_input[...,1:-1,[0,-1],:].flatten(-3,-2),
                                batch_image_input[...,-1,1:-1,:]
                               ],1)
        corn_input = batch_image_input[...,[0,0,-1],[0,-1,0],:]
        cent_input = batch_image_input[...,-1,-1,:]
        return bulk_input,edge_input,corn_input,cent_input

    def forward(self, input_data,contraction_mode='top2bot',batch_method='physics_index_first'):
        # the input data shape is (B,L,L,pd)
        if batch_method == 'physics_index_first':
            bulk_input,edge_input,corn_input,cent_input = self.flatten_image_input(input_data)
            bulk_tensors = torch.einsum("lpabcd,klp->lkabcd",self.bulk_tensors,bulk_input)
            edge_tensors = torch.einsum(" lpabc,klp->lkabc" ,self.edge_tensors,edge_input)
            corn_tensors = torch.einsum("  lpab,klp->lkab"  ,self.corn_tensors,corn_input)
            cent_tensors = torch.einsum("   opab,kp->okab"  ,self.cent_tensors,cent_input)
        if contraction_mode == 'top2bot':
            tensor     = self.mpo_line(0,bulk_tensors,edge_tensors,corn_tensors,cent_tensors)
            for i in range(1,self.H-1):
                #print(get_mps_size_list(tensor))
                mpo    = self.mpo_line(i,bulk_tensors,edge_tensors,corn_tensors,cent_tensors)
                #print(get_mps_size_list(mpo))
                #print("=============")
                tensor = self.batch_contract_mps_mpo(tensor,mpo)
                #tensor = right_mps_form(tensor)
                #tensor,scale = approxmate_mps_line(tensor,max_singular_values=100)
            mps    = self.mpo_line(self.H-1,bulk_tensors,edge_tensors,corn_tensors,cent_tensors)

            #print(get_mps_size_list(mps))
            tensor_left,tensor_inne,tensor_rigt = self.batch_contract_mps_mps(tensor,mps)
            tensor_inne = self.get_batch_chain_contraction_fast(tensor_inne)
            tensor  = torch.einsum('ka,kba,kob->ko',tensor_left,tensor_inne,tensor_rigt)
            return tensor

class PEPS_einsum_uniform_shape_6x6_fast(PEPS_einsum_uniform_shape):
    def __init__(self,W=6,H=6,out_features=10,**kargs):
        assert W==6
        assert H==6
        super().__init__(6,6,out_features,**kargs)

    def forward(self,input_data):
        bulk_input,edge_input,corn_input,cent_input = self.flatten_image_input(input_data)
        bulk_tensors = torch.einsum("lpabcd,klp->lkabcd",self.bulk_tensors,bulk_input)
        edge_tensors = torch.einsum(" lpabc,klp->lkabc" ,self.edge_tensors,edge_input)
        corn_tensors = torch.einsum("  lpab,klp->lkab"  ,self.corn_tensors,corn_input)
        cent_tensors = torch.einsum("   opab,kp->okab"  ,self.cent_tensors,cent_input)
        #corner_contraction
        W=H=6;
        L=4              ;corn_index = [0,1,2,3]
        L=2*(W-2)+2*(H-2);edge_index1=[0,(W-2)+1,W-2+2*(H-3),L-1] # [0,5,10,15] for 6x6
        L=(W-2)*(H-2)    ;bulk_index = [0,W-3,(W-2)*(H-3),L-1]# [0,3,12,15] for 6x6
        L=2*(W-2)+2*(H-2);edge_index2=[(W-2),(W-2)-1,L-(W-2),L-(W-2)-1] # [4,3,12,11] for 6x6
        corner123_contraction = torch.einsum("lkab,lkcdb,lkefcg,lkgah->lkhedf",
                                       corn_tensors[corn_index[:3]],
                                       edge_tensors[edge_index1[:3]],
                                       bulk_tensors[bulk_index[:3]],
                                       edge_tensors[edge_index2[:3]],
                                      ).flatten(-4,-3).flatten(-2,-1)
        corner4_contraction = torch.einsum("okab,kceb,khicg,kfga->okfhei" ,
                                       cent_tensors,
                                       edge_tensors[edge_index1[-1]],
                                       bulk_tensors[bulk_index[-1]],
                                       edge_tensors[edge_index2[-1]],
                                      ).flatten(-4,-3).flatten(-2,-1)

        L=2*(W-2)+2*(H-2);egde_index = [1,W-4,W,W+1,-(W-2)-4,-(W-2)-3,L-3,L-2]
        # [1,2,6,7,8,9,13,14] for 6x6
        L=(W-2)*(H-2);bulk_index = [1,W-4,W-2,W-2+W-2-1,L-(W-2)-(W-3)-1,L-(W-2)-1,L-3,L-2]
        # [1,2,4,7,8,11,13,14] for 6x6
        edge_fast_contraction = torch.einsum("lkabc,lkefah->lkebfch" ,
                                               edge_tensors[egde_index],
                                               bulk_tensors[bulk_index],
                                              ).flatten(-4,-3).flatten(-2,-1)

        L=(W-2)*(H-2);bulk_index = [W-1,2*(W-2)-2,L-W-(W-2)+3,L-(W)]# [5,6,9,10] for 6x6
        edge_index1 = [0,3,4,7]
        edge_index2 = [2,1,6,5]
        corner123_contraction = torch.einsum("lkab,lkcdb,lkefcg,lkgah->lkhedf" ,
                                       corner123_contraction,
                                       edge_fast_contraction[edge_index1[:3]],
                                       bulk_tensors[bulk_index[:3]],
                                       edge_fast_contraction[edge_index2[:3]],
                                      ).flatten(-4,-3).flatten(-2,-1)

        corner4_contraction = torch.einsum("okab,kcdb,kefcg,kgah->okhedf",
                                       corner4_contraction,
                                       edge_fast_contraction[edge_index1[-1]],
                                       bulk_tensors[bulk_index[-1]],
                                       edge_fast_contraction[edge_index2[-1]],
                                      ).flatten(-4,-3).flatten(-2,-1)
        tensor  = torch.einsum("kab,kbc->kac",corner123_contraction[0],corner123_contraction[1])
        tensor  = torch.einsum("kab,kbc->kac",tensor,corner123_contraction[2])
        tensor  = torch.einsum("kab,okba->ko",tensor,corner4_contraction)
        return tensor

class PEPS_einsum_uniform_shape_6x6_fast2(TN_Base):
    def __init__(self, W=6, H=6 ,out_features=10,
                       in_physics_bond = 2, virtual_bond_dim=2,
                       bias=True,label_position='center',init_std=1e-10,contraction_mode = 'recursion'):
        super().__init__()
        assert W==6
        assert H==6
        P = in_physics_bond
        O = out_features
        if isinstance(virtual_bond_dim,list):
            assert len(virtual_bond_dim)==W
            for t in virtual_bond_dim:
                assert isinstance(t,list)
                assert len(t) == H
        elif isinstance(virtual_bond_dim,int):
            D = virtual_bond_dim
            top_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(H-2)] + [(D,D)]
            mid_shape_list  =[  [(D,D,D)]+ [(D,D,D,D) for i in range(H-2)] + [(D,D,D)] for _ in range(W-2)]
            bot_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(H-2)] + [(D,D)]
            virtual_bond_dim= [top_shape_list]+mid_shape_list+[bot_shape_list]
        label_x,label_y = W-1,H-1
        # may cause problem for label at edge, be care
        a,b = list(virtual_bond_dim[label_x][label_y])
        virtual_bond_dim[label_x][label_y] = (a,O,b)

        node_list,sublist_list,outlist =  create_templete_2DTN_tn(virtual_bond_dim)
        path,info = get_optim_path_by_oe_from_tn(node_list)
        self.sublist_list = sublist_list
        self.outlist      = outlist
        self.path         = path

        self.bulk_tensors = nn.Parameter(self.rde2D((     (W-2)*(H-2),P,D,D,D,D),init_std))
        self.edge_tensors = nn.Parameter(self.rde2D( (2*(W-2)+2*(H-2),P,D,D,D),init_std))
        self.corn_tensors = nn.Parameter(self.rde2D(                 (3,P,D,D),init_std))
        self.cent_tensors = nn.Parameter(self.rde2D(                 (O,P,D,D),init_std))

    def forward(self,input_data):
        bulk_input,edge_input,corn_input,cent_input = PEPS_einsum_uniform_shape.flatten_image_input(input_data)
        bulk_tensors = torch.einsum("lpabcd,klp->lkabcd",self.bulk_tensors,bulk_input)
        edge_tensors = torch.einsum(" lpabc,klp->lkabc" ,self.edge_tensors,edge_input)
        corn_tensors = torch.einsum("  lpab,klp->lkab"  ,self.corn_tensors,corn_input)
        cent_tensors = torch.einsum("   opab,kp->okab"  ,self.cent_tensors,cent_input)
        batch_input  = [corn_tensors[ 0],*edge_tensors[ 0: 4],corn_tensors[ 1],
                        edge_tensors[ 4],*bulk_tensors[ 0: 4],edge_tensors[ 5],
                        edge_tensors[ 6],*bulk_tensors[ 4: 8],edge_tensors[ 7],
                        edge_tensors[ 8],*bulk_tensors[ 8:12],edge_tensors[ 9],
                        edge_tensors[10],*bulk_tensors[12:16],edge_tensors[11],
                        corn_tensors[ 2],*edge_tensors[12:16],cent_tensors.permute(1,2,0,3)]

        operands=[]
        for tensor,sublist in zip(batch_input,self.sublist_list):
            operand = [tensor,[...,*sublist]]
            operands+=operand
        operands+= [[...,*self.outlist]]
        return torch.einsum(*operands,optimize=self.path)


class PEPS_einsum_uniform_shape_6x6_fast_one_step(PEPS_einsum_uniform_shape_6x6_fast):
    def __init__(self, out_features,**kargs):
        super().__init__(out_features,**kargs)

    def forward(self,input_data):
        bulk_input,edge_input,corn_input,cent_input = self.flatten_image_input(input_data)
        bulk_tensors = torch.einsum("lpabcd,klp->lkabcd",self.bulk_tensors,bulk_input)
        edge_tensors = torch.einsum(" lpabc,klp->lkabc" ,self.edge_tensors,edge_input)
        corn_tensors = torch.einsum("  lpab,klp->lkab"  ,self.corn_tensors,corn_input)
        cent_tensors = torch.einsum("   opab,kp->okab"  ,self.cent_tensors,cent_input)
        #corner_contraction
        corner123_contraction = torch.einsum(
            "xyab,xycdb,xyefd,xygha,xyijch,xyklej,xymng,xyopin,xyqrkp->xymoqflr",
             corn_tensors[[ 0, 1, 2]] ,
             edge_tensors[[ 0, 5,10]],
             edge_tensors[[ 1, 7, 8]],
             edge_tensors[[ 4, 3,12]],
             bulk_tensors[[ 0, 3,12]],
             bulk_tensors[[ 1, 7, 8]],
             edge_tensors[[ 6, 2,13]],
             bulk_tensors[[ 4, 2,13]],
             bulk_tensors[[ 5, 6, 9]]
            ).flatten(-6,-4).flatten(-3,-1)

        corner4_contraction = torch.einsum(
            "zyab,ycdb,yefd,ygha,yijch,yklej,ymng,yopin,yqrkp->zymoqflr",
             cent_tensors,#corn_tensors[ 3] ,
             edge_tensors[15],
             edge_tensors[14],
             edge_tensors[11],
             bulk_tensors[15],
             bulk_tensors[14],
             edge_tensors[ 9],
             bulk_tensors[11],
             bulk_tensors[10]
            ).flatten(-6,-4).flatten(-3,-1)

        tensor  = torch.einsum("yab,ybc,ycd,oyda->yo",
                               corner123_contraction[0],
                               corner123_contraction[1],
                               corner123_contraction[2],
                               corner4_contraction)

        return tensor


class PEPS_einsum_arbitrary_shape_fast(TN_Base):
    def __init__(self, W, H ,out_features=10,
                       in_physics_bond = 2, virtual_bond_dim=2,
                       bias=True,label_position='center',init_std=1e-10,contraction_mode = 'recursion'):
        super().__init__()
        P = in_physics_bond
        O = out_features
        if isinstance(virtual_bond_dim,list):
            assert len(virtual_bond_dim)==W
            for t in virtual_bond_dim:
                assert isinstance(t,list)
                assert len(t) == H
        elif isinstance(virtual_bond_dim,int):
            D = virtual_bond_dim
            top_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(H-2)] + [(D,D)]
            mid_shape_list  =[  [(D,D,D)]+ [(D,D,D,D) for i in range(H-2)] + [(D,D,D)] for _ in range(W-2)]
            bot_shape_list  =   [(D,D)]  + [  (D,D,D) for i in range(H-2)] + [(D,D)]
            virtual_bond_dim= [top_shape_list]+mid_shape_list+[bot_shape_list]
        label_x,label_y = W-1,H-1
        # may cause problem for label at edge, be care
        origin_shape = list(virtual_bond_dim[label_x][label_y])
        origin_shape.insert(1,O)
        virtual_bond_dim[label_x][label_y] = tuple(origin_shape)

        node_list,sublist_list,outlist =  create_templete_2DTN_tn(virtual_bond_dim)
        path,info = get_optim_path_by_oe_from_tn(node_list)
        self.sublist_list = sublist_list
        self.outlist      = outlist
        self.path         = path

        unit_list      = [self.rde2D((P,*l),init_std,offset=1) for t in virtual_bond_dim for l in t]
        # the last one get out put feature, so initial independent.
        p,a,o,b = unit_list[-1].shape
        unit_list[-1]  = self.rde2D((p,o,a,b),init_std,offset=2).permute(0,2,1,3)
        assert len(unit_list)==len(sublist_list)
        self.unit_list = [nn.Parameter(v) for v in unit_list]
        for i, v in enumerate(self.unit_list):
            self.register_parameter(f'unit_{i//W}-{i%W}', param=v)
    def forward(self,input_data):
        #input data shape B,W,H,P
        input_data  = input_data.flatten(1,2)
        batch_input = []
        for _input,unit in zip(input_data.permute(1,0,2),self.unit_list):
            if len(unit.shape)==3:
                batch_unit = torch.einsum("kp,pab->kab",_input,unit)
            elif len(unit.shape)==4:
                batch_unit = torch.einsum("kp,pabc->kabc",_input,unit)
            elif len(unit.shape)==5:
                batch_unit = torch.einsum("kp,pabcd->kabcd",_input,unit)
            batch_input.append(batch_unit)

        operands=[]
        for tensor,sublist in zip(batch_input,self.sublist_list):
            operand = [tensor,[...,*sublist]]
            operands+=operand
        operands+= [[...,*self.outlist]]
        return torch.einsum(*operands,optimize=self.path)

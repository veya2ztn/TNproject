from .tensornetwork_base import *
from .model_utils import *
import opt_einsum as oe
import numpy as np
from itertools import permutations

class PEPS_einsum_uniform_shape(TN_Base):
    def __init__(self, W,H,out_features,
                       in_physics_bond = 2, virtual_bond_dim=2,
                       bias=True,label_position='center',init_std=1e-10,
                       contraction_mode = 'recursion'):
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
        self.bulk_tensors = nn.Parameter(self.rde2D((     (W-2)*(H-2),P,D,D,D,D),init_std, offset=2))
        self.edge_tensors = nn.Parameter(self.rde2D( (2*(W-2)+2*(H-2),P,D,D,D),init_std, offset=2))
        self.corn_tensors = nn.Parameter(self.rde2D(                 (3,P,D,D),init_std, offset=2))
        self.cent_tensors = nn.Parameter(self.rde2D(                 (O,P,D,D),init_std, offset=2))
    def mpo_line(self,i,bulk_tensors,edge_tensors,corn_tensors,cent_tensors):
        W=self.W
        H=self.H
        if i == 0:
            return  corn_tensors[0],edge_tensors[0:W-2],corn_tensors[1]
        elif i == H - 1:
            return  corn_tensors[2],edge_tensors[-(W-2):],cent_tensors
        else:
            return  edge_tensors[W-4+2*i],bulk_tensors[(W-2)*(i-1):(W-2)*i],edge_tensors[W-4+2*i+1]

    def get_batch_contraction_network(self,input_data):
        bulk_input,edge_input,corn_input = self.flatten_image_input(input_data)
        cent_input   = corn_input[..., -1,:]
        corn_input   = corn_input[...,:-1,:]
        bulk_tensors = self.einsum_engine("lpabcd,klp->lkabcd",self.bulk_tensors,bulk_input)
        edge_tensors = self.einsum_engine(" lpabc,klp->lkabc" ,self.edge_tensors,edge_input)
        corn_tensors = self.einsum_engine("  lpab,klp->lkab"  ,self.corn_tensors,corn_input)
        cent_tensors = self.einsum_engine("   opab,kp->okab"  ,self.cent_tensors,cent_input)
        return bulk_tensors,edge_tensors,corn_tensors,cent_tensors

    def forward(self, input_data,contraction_mode='top2bot',batch_method='physics_index_first'):
        # the input data shape is (B,L,L,pd)
        if batch_method == 'physics_index_first':
            bulk_tensors,edge_tensors,corn_tensors,cent_tensors = self.get_batch_contraction_network(input_data)
        if contraction_mode == 'top2bot':
            tensor     = self.mpo_line(0,bulk_tensors,edge_tensors,corn_tensors,cent_tensors)
            for i in range(1,self.H-1):
                #print(get_mps_size_list(tensor))
                mpo    = self.mpo_line(i,bulk_tensors,edge_tensors,corn_tensors,cent_tensors)
                #print(get_mps_size_list(mpo))
                #print("=============")
                tensor = batch_contract_mps_mpo(tensor,mpo)
                #tensor = right_mps_form(tensor)
                #tensor,scale = approxmate_mps_line(tensor,max_singular_values=100)
            mps    = self.mpo_line(self.H-1,bulk_tensors,edge_tensors,corn_tensors,cent_tensors)

            #print(get_mps_size_list(mps))
            tensor_left,tensor_inne,tensor_rigt = batch_contract_mps_mps(tensor,mps)
            tensor_inne = self.get_batch_chain_contraction_fast(tensor_inne)
            tensor  = self.einsum_engine('ka,kba,kob->ko',tensor_left,tensor_inne,tensor_rigt)
            return tensor
class PEPS_einsum_uniform_shape_boundary_2direction(TN_Base):
    def __init__(self, W,H,out_features,
                       in_physics_bond = 2, virtual_bond_dim=2,
                       bias=True,label_position='center',init_std=1e-10,contraction_mode = 'recursion'):
        super().__init__()
        assert (W%2 == 1) and (H%2 == 1)
        self.W             = W
        self.H             = H

        self.out_features  = O = out_features
        self.vbd           = D = virtual_bond_dim
        self.ipb           = P = in_physics_bond

        top_shape_list  =[  [(2,D,D)]  + [  (2,D,D,D) for i in range(H-2)] + [(2,D,D)]]
        bul_shape_list  =[  [(2,D,D,D)]+ [(2,D,D,D,D) for i in range(H-2)] + [(2,D,D,D)] for _ in range((W-2)//2)]
        cen_shape_list  =[  [(  D,D,D)]+ [(  D,D,D,D) for i in range(H-2)] + [(O,D,D,D)]]
        virtual_bond_dim= top_shape_list+bul_shape_list+cen_shape_list
        unit_list = []
        for line in virtual_bond_dim[:-1]:
            unit_list.append([nn.Parameter(self.rde2D((P,*l),init_std,offset=2)) for l in line])
        line = virtual_bond_dim[-1]
        unit_list.append([nn.Parameter(self.rde2D((P,*l),init_std,offset=1)) for l in line[:-1]])
        l = line[-1]
        unit_list[-1].append(nn.Parameter(self.rde2D((P,*l),init_std,offset=2)))

        self.unit_list = unit_list
        for i,line in enumerate(unit_list):
            for j,v in enumerate(line):
                self.register_parameter(f'unit_{i}-{j}', param=v)

    def forward(self, input_data,contraction_mode='top2bot',batch_method='physics_index_first'):
        input_data  = input_data#(B,W,H,P)
        batch_input = []
        for i,(input_line,unit_line) in enumerate(zip(input_data.permute(1,2,0,3),self.unit_list)):
            batch_line=[]
            for _input,unit in zip(input_line,unit_line):
                if len(unit.shape)==3:
                    batch_unit = self.einsum_engine("kp,pab->kab",_input,unit)
                elif len(unit.shape)==4:
                    batch_unit = self.einsum_engine("kp,pabc->kabc",_input,unit)
                elif len(unit.shape)==5:
                    batch_unit = self.einsum_engine("kp,pabcd->kabcd",_input,unit)
                elif len(unit.shape)==6:
                    batch_unit = self.einsum_engine("kp,pabcde->kabcde",_input,unit)
                else:
                    print(unit.shape)
                    raise NotImplementedError
                if i < len(self.unit_list)-1:
                    batch_unit = batch_unit.flatten(0,1)
                batch_line.append(batch_unit)
            batch_input.append(batch_line)

        tensor     = batch_input[0]
        for i in range(1,self.H//2):
            mpo    = batch_input[i]
            tensor = batch_contract_mps_mpo(tensor,mpo)

        shapel= [t.shape[1:] for t in tensor]
        tensor= [t.reshape(-1,2,*s) for s,t in zip(shapel,tensor)]
        upper = [t[:,0] for t in tensor  ]
        cente = batch_input[-1]
        lower = [t[:,1] for t in tensor  ]
        tensor = batch_contract_mps_mpo(upper,cente)
        mps_left,mps_inne,mps_rigt =  lower[0], lower[1:-1], lower[-1]
        mpo_left,mpo_inne,mpo_rigt = tensor[0],tensor[1:-1],tensor[-1]
        new_mps_list= []
        tensor = self.einsum_engine("kab,kac->kbc",mps_left,mpo_left).flatten(-2,-1)
        new_mps_list.append(tensor)
        for mps,mpo in zip(mps_inne,mpo_inne):
            tensor =self.einsum_engine("kabc,kade->kbdce",mps,mpo).flatten(-4,-3).flatten(-2,-1)
            new_mps_list.append(tensor)
        tensor = self.einsum_engine("kab,koac->kobc",mps_rigt,mpo_rigt).flatten(-2,-1)
        new_mps_list.append(tensor)
        tensor = new_mps_list
        tensor_left,tensor_inne,tensor_rigt = tensor[0],torch.stack(tensor[1:-1]),tensor[-1]
        tensor_inne = self.get_batch_chain_contraction_fast(tensor_inne)
        tensor  = self.einsum_engine('ka,kba,kob->ko',tensor_left,tensor_inne,tensor_rigt)
        return tensor
class PEPS_einsum_uniform_shape_6x6_fast(PEPS_einsum_uniform_shape):
    def __init__(self,W=6,H=6,out_features=10,**kargs):
        assert W==6
        assert H==6
        super().__init__(6,6,out_features,**kargs)

    def forward(self,input_data):
        bulk_tensors,edge_tensors,corn_tensors,cent_tensors = self.get_batch_contraction_network(input_data)

        W=H=6;
        L=4              ;corn_index = [0,1,2,3]
        L=2*(W-2)+2*(H-2);edge_index1=[0,(W-2)+1,W-2+2*(H-3),L-1] # [0,5,10,15] for 6x6
        L=(W-2)*(H-2)    ;bulk_index = [0,W-3,(W-2)*(H-3),L-1]# [0,3,12,15] for 6x6
        L=2*(W-2)+2*(H-2);edge_index2=[(W-2),(W-2)-1,L-(W-2),L-(W-2)-1] # [4,3,12,11] for 6x6
        corner123_contraction = self.einsum_engine("lkab,lkcdb,lkefcg,lkgah->lkhedf",
                                       corn_tensors[corn_index[:3]],
                                       edge_tensors[edge_index1[:3]],
                                       bulk_tensors[bulk_index[:3]],
                                       edge_tensors[edge_index2[:3]],
                                      ).flatten(-4,-3).flatten(-2,-1)
        corner4_contraction   = self.einsum_engine("okab,kceb,khicg,kfga->okfhei" ,
                                       cent_tensors,
                                       edge_tensors[edge_index1[-1]],
                                       bulk_tensors[bulk_index[-1]],
                                       edge_tensors[edge_index2[-1]],
                                      ).flatten(-4,-3).flatten(-2,-1)

        L=2*(W-2)+2*(H-2);egde_index = [1,W-4,W,W+1,-(W-2)-4,-(W-2)-3,L-3,L-2]
        # [1,2,6,7,8,9,13,14] for 6x6
        L=(W-2)*(H-2);bulk_index = [1,W-4,W-2,W-2+W-2-1,L-(W-2)-(W-3)-1,L-(W-2)-1,L-3,L-2]
        # [1,2,4,7,8,11,13,14] for 6x6
        edge_fast_contraction = self.einsum_engine("lkabc,lkefah->lkebfch" ,
                                               edge_tensors[egde_index],
                                               bulk_tensors[bulk_index],
                                              ).flatten(-4,-3).flatten(-2,-1)

        L=(W-2)*(H-2);bulk_index = [W-1,2*(W-2)-2,L-W-(W-2)+3,L-(W)]# [5,6,9,10] for 6x6
        edge_index1 = [0,3,4,7]
        edge_index2 = [2,1,6,5]
        corner123_contraction = self.einsum_engine("lkab,lkcdb,lkefcg,lkgah->lkhedf" ,
                                       corner123_contraction,
                                       edge_fast_contraction[edge_index1[:3]],
                                       bulk_tensors[bulk_index[:3]],
                                       edge_fast_contraction[edge_index2[:3]],
                                      ).flatten(-4,-3).flatten(-2,-1)

        corner4_contraction = self.einsum_engine("okab,kcdb,kefcg,kgah->okhedf",
                                       corner4_contraction,
                                       edge_fast_contraction[edge_index1[-1]],
                                       bulk_tensors[bulk_index[-1]],
                                       edge_fast_contraction[edge_index2[-1]],
                                      ).flatten(-4,-3).flatten(-2,-1)
        tensor  = self.einsum_engine("kab,kbc->kac",corner123_contraction[0],corner123_contraction[1])
        tensor  = self.einsum_engine("kab,kbc->kac",tensor,corner123_contraction[2])
        tensor  = self.einsum_engine("kab,okba->ko",tensor,corner4_contraction)
        return tensor
class PEPS_einsum_uniform_shape_6x6_fast_one_step(PEPS_einsum_uniform_shape_6x6_fast):
    def __init__(self, W=6,H=6,out_features=10,virtual_bond_dim=2,**kargs):
        self.path_1    = None
        self.path_1    = None
        self.path_final= None
        super().__init__(W=W,H=H,out_features=out_features,virtual_bond_dim=virtual_bond_dim,**kargs)

    def forward(self,input_data):
        bulk_tensors,edge_tensors,corn_tensors,cent_tensors = self.get_batch_contraction_network(input_data)
        #corner_contraction
        corner123_contraction = self.einsum_engine("xyab,xycdb,xyefd,xygha,xyijch,xyklej,xymng,xyopin,xyqrkp->xymoqflr",
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
        corner4_contraction = self.einsum_engine("zyab,ycdb,yefd,ygha,yijch,yklej,ymng,yopin,yqrkp->zymoqflr",
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
        tensor  = self.einsum_engine("yab,ybc,ycd,oyda->yo",
                corner123_contraction[0],
                corner123_contraction[1],
                corner123_contraction[2],
                corner4_contraction)
        return tensor
class PEPS_einsum_uniform_shape_6x6_fast_one_step_usesublist(PEPS_einsum_uniform_shape_6x6_fast):
    def __init__(self, W=6,H=6,out_features=10,virtual_bond_dim=2,**kargs):
        O = out_features
        D = virtual_bond_dim
        assert isinstance(virtual_bond_dim,int)

        L=3
        tn2D_shape_list = [[(D,D)]+[(D,D,D)]*(L-1)]+[[(D,D,D)]+[(D,D,D,D)]*(L-1)]*(L-1)
        node_list,sublist_list,outlist = sub_network_tn(tn2D_shape_list)
        path,info                      = get_optim_path_by_oe_from_tn(node_list)
        last_idx = outlist.pop()
        outlist.insert(L-1,last_idx)
        #outlist should be [5, 11, 17, 12, 14, 16] for 3x3
        #print(outlist)
        self.sublist_list_1 = sublist_list
        self.outlist_1      = outlist
        self.path_1         = path

        tn2D_shape_list = [[(D,D,O)]+[(D,D,D)]*(L-1)]+[[(D,D,D)]+[(D,D,D,D)]*(L-1)]*(L-1)
        node_list,sublist_list,outlist =  sub_network_tn(tn2D_shape_list)
        path,info = get_optim_path_by_oe_from_tn(node_list)
        last_idx = outlist.pop()
        outlist.insert(L,last_idx)
        #print(outlist)
        #outlist should be [2, 6, 12, 18, 13, 15, 17] for 3x3
        self.sublist_list_2 = sublist_list
        self.outlist_2      = outlist
        self.path_2         = path

        self.path_final     = None
        super().__init__(W=W,H=H,out_features=out_features,virtual_bond_dim=virtual_bond_dim,**kargs)

    def forward(self,input_data):
        bulk_tensors,edge_tensors,corn_tensors,cent_tensors = self.get_batch_contraction_network(input_data)
        #corner_contraction
        batch_input = [ corn_tensors[[ 0, 1, 2]] ,
                        edge_tensors[[ 0, 5,10]],
                        edge_tensors[[ 1, 7, 8]],
                        edge_tensors[[ 4, 3,12]],
                        bulk_tensors[[ 0, 3,12]],
                        bulk_tensors[[ 1, 7, 8]],
                        edge_tensors[[ 6, 2,13]],
                        bulk_tensors[[ 4, 2,13]],
                        bulk_tensors[[ 5, 6, 9]]
                      ]
        operands=[]
        for tensor,sublist in zip(batch_input,self.sublist_list_1):
            operand = [tensor,[...,*sublist]]
            operands+=operand
        operands+= [[...,*self.outlist_1]]
        corner123_contraction =  self.einsum_engine(*operands,optimize=self.path_1
                                            ).flatten(-6,-4).flatten(-3,-1)
        # corner123_contraction = self.einsum_engine(
        #     "xyab,xycdb,xyefd,xygha,xyijch,xyklej,xymng,xyopin,xyqrkp->xymoqflr",
        #      corn_tensors[[ 0, 1, 2]] ,
        #      edge_tensors[[ 0, 5,10]],
        #      edge_tensors[[ 1, 7, 8]],
        #      edge_tensors[[ 4, 3,12]],
        #      bulk_tensors[[ 0, 3,12]],
        #      bulk_tensors[[ 1, 7, 8]],
        #      edge_tensors[[ 6, 2,13]],
        #      bulk_tensors[[ 4, 2,13]],
        #      bulk_tensors[[ 5, 6, 9]]
        #     ).flatten(-6,-4).flatten(-3,-1)
        batch_input = [ cent_tensors,#corn_tensors[ 3] ,
                        edge_tensors[15],
                        edge_tensors[14],
                        edge_tensors[11],
                        bulk_tensors[15],
                        bulk_tensors[14],
                        edge_tensors[ 9],
                        bulk_tensors[11],
                        bulk_tensors[10]
                      ]
        operands=[]
        for tensor,sublist in zip(batch_input,self.sublist_list_2):
            operand = [tensor,[...,*sublist]]
            operands+=operand
        operands+= [[...,*self.outlist_2]]
        corner4_contraction =  self.einsum_engine(*operands,optimize=self.path_2
                                            ).flatten(-6,-4).flatten(-3,-1)
        # corner4_contraction = self.einsum_engine(
        #     "zyab,ycdb,yefd,ygha,yijch,yklej,ymng,yopin,yqrkp->zymoqflr",
        #      cent_tensors,#corn_tensors[ 3] ,
        #      edge_tensors[15],
        #      edge_tensors[14],
        #      edge_tensors[11],
        #      bulk_tensors[15],
        #      bulk_tensors[14],
        #      edge_tensors[ 9],
        #      bulk_tensors[11],
        #      bulk_tensors[10]
        #     ).flatten(-6,-4).flatten(-3,-1)
        equation = "yab,ybc,ycd,yoda->yo"
        tensor_l = [ corner123_contraction[0],
                     corner123_contraction[1],
                     corner123_contraction[2],
                     corner4_contraction]
        if self.path_final is None:
            self.path_final = oe.contract_path(equation, *tensor_l)[0]
        tensor  = self.einsum_engine(equation,*tensor_l, optimize=self.path_final)

        return tensor
class PEPS_einsum_uniform_shape_6x6_one_contracting(PEPS_einsum_uniform_shape):
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

        self.bulk_tensors = nn.Parameter(self.rde2D((     (W-2)*(H-2),P,D,D,D,D),init_std, offset=2))
        self.edge_tensors = nn.Parameter(self.rde2D( (2*(W-2)+2*(H-2),P,D,D,D),init_std, offset=2))
        self.corn_tensors = nn.Parameter(self.rde2D(                 (3,P,D,D),init_std, offset=2))
        self.cent_tensors = nn.Parameter(self.rde2D(                 (O,P,D,D),init_std, offset=2))

    def forward(self,input_data):
        bulk_tensors,edge_tensors,corn_tensors,cent_tensors = self.get_batch_contraction_network(input_data)
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
        return self.einsum_engine(*operands,optimize=self.path)

class PEPS_uniform_shape_symmetry_base(TN_Base):
    def __init__(self, W=6,H=6,out_features=16,
                       in_physics_bond = 3, virtual_bond_dim=3,
                       init_std=1e-10,symmetry = None,init_method=None,init_set_var=1,**kargs):
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


        index_matrix = torch.LongTensor([[[i,j] for j in range(W)] for i in range(H)])
        bulk_index,edge_index,corn_index=self.flatten_image_input(index_matrix)
        flag_matrix     = torch.zeros(W,H).long()
        position_matrix = torch.zeros(W,H).long()
        symmetry_map    = self.generate_symmetry_map(W,H,symmetry=symmetry)
        tensor_needed   = []
        active_index    = []
        for flag, types in enumerate([corn_index,edge_index,bulk_index]):
            res= 0
            tensor_order_for_this_type=[]
            for (i,j) in types.numpy():
                flag_matrix[i,j]=flag
                now_pos = (i,j)
                while (now_pos in symmetry_map) and (isinstance(symmetry_map[now_pos],tuple )):
                    now_pos = symmetry_map[now_pos]
                if now_pos not in symmetry_map:
                    symmetry_map[now_pos] = res
                    res += 1
                position_matrix[i,j]= symmetry_map[now_pos]
                tensor_order_for_this_type.append(position_matrix[i,j])
            active_index.append(tensor_order_for_this_type)
            tensor_needed.append(res)
        assert W*H == len(symmetry_map)
        # flag_matrix record the tensor type information: is corner(a,b) or edge(a,b,c) or bulk(a,b,c,d)
        # position_matrix record a position index for (i,j) should be where in the select rule self.flatten_image_input
        #    this rule is also the order in this weight.
        #    for example, the (x,y) = [0,-1] is the second tensor in corn and should be the second tensor after
        #    contract self.corn_tensors and corn_input
        #    the symmetry map indicate the symmetry group:
        #        if "R4" symmetry, then corn [0,0] [0,-1] [-1,-1] [-1,0] will share same active kernel
        #        so there only end tensor_needed[0] = 1 corn-weight
        #    [Update]: however, indices rotation is need since
        #      TODO: realize right indices rotation for symmetry element.
        self.active_index = active_index

        self.corn_tensors = nn.Parameter(self.rde2D( (tensor_needed[0],O,P,D,D)  ,init_std, offset=3))
        self.edge_tensors = nn.Parameter(self.rde2D( (tensor_needed[1],P,D,D,D)  ,init_std, offset=2))
        self.bulk_tensors = nn.Parameter(self.rde2D( (tensor_needed[2],P,D,D,D,D),init_std, offset=2))
        self.weight_init(method = init_method,set_var=init_set_var)
        part_lu_idex = torch.rot90(index_matrix,k=0)[:LW,:LH].flatten(0,1).transpose(1,0)
        part_ru_idex = torch.rot90(index_matrix,k=1)[:LW,:LH].flatten(0,1).transpose(1,0)
        part_rd_idex = torch.rot90(index_matrix,k=2)[:LW,:LH].flatten(0,1).transpose(1,0)
        part_ld_idex = torch.rot90(index_matrix,k=3)[:LW,:LH].flatten(0,1).transpose(1,0)
        self.indexrule = torch.stack([
                               position_matrix[part_lu_idex[0],part_lu_idex[1]],
                               position_matrix[part_ru_idex[0],part_ru_idex[1]],
                               position_matrix[part_rd_idex[0],part_rd_idex[1]],
                               position_matrix[part_ld_idex[0],part_ld_idex[1]],
                               ]).transpose(1,0)
        self.partrule        = flag_matrix[part_lu_idex[0],part_lu_idex[1]]
        # self.indexrule maintain the contraction engine (sublist)
        # for each tensor should be located in where.
        # in this module, the first 1/4 part is the left-upper corner. the next is its 90 degree rotation, then 180 then 270.

        # in real contraction, we first get the sublist of the tensors' position (i,j),
        # then use partrule and position_matrix get the tensor from the map position_matrix[i,j] = index, partrule[i,j] = 0,1,2
        self.flag_matrix     = flag_matrix
        self.position_matrix = position_matrix
        self.cent_tensor_idx = position_matrix[self.W//2,self.H//2] if self.W%2==1 else None

    def get_batch_contraction_network(self,input_data):
        bulk_inputs,edge_inputs,corn_inputs = self.flatten_image_input(input_data)
        # the input data would divide in same rule as flatten_image_input: NxN -> corn/edge/bulk input list
        corn_tensors = self.einsum_engine(" lopab,klp->lkoab" ,self.corn_tensors[[self.active_index[0]]],corn_inputs)
        edge_tensors = self.einsum_engine(" lpabc,klp->lkabc" ,self.edge_tensors[[self.active_index[1]]],edge_inputs)
        bulk_tensors = self.einsum_engine("lpabcd,klp->lkabcd",self.bulk_tensors[[self.active_index[2]]],bulk_inputs)
        return bulk_tensors,edge_tensors,corn_tensors

    def generate_symmetry_map(self,W,H,symmetry=None):
        if symmetry is None:return {}
        LW =  int(np.ceil(1.0*W/2))
        LH = int(np.floor(1.0*H/2))
        index_matrix = torch.LongTensor([[[i,j] for j in range(W)] for i in range(H)])
        if symmetry == 'P4':
            part_lu_idex = torch.rot90(index_matrix,k=0)[:LW,:LH].flatten(0,1).numpy()
            part_ru_idex = torch.rot90(index_matrix,k=1)[:LW,:LH].flatten(0,1).numpy()
            part_rd_idex = torch.rot90(index_matrix,k=2)[:LW,:LH].flatten(0,1).numpy()
            part_ld_idex = torch.rot90(index_matrix,k=3)[:LW,:LH].flatten(0,1).numpy()
            symmetry_map = {}
            for n,(a,b,c) in enumerate(zip(part_ru_idex,part_rd_idex,part_ld_idex)):
                i = part_lu_idex[n][0]
                j = part_lu_idex[n][1]
                symmetry_map[a[0],a[1]] = (i,j)
                symmetry_map[b[0],b[1]] = (i,j)
                symmetry_map[c[0],c[1]] = (i,j)
            return symmetry_map
        elif symmetry == 'P4Z2':
            symmetry_map = self.generate_symmetry_map(W,H,symmetry='P4')
            part_lu_idex = torch.rot90(index_matrix,k=0)[:LW,:LH].flatten(0,1).numpy()
            for i,j in part_lu_idex:
                if (j,i) not in symmetry_map and j>i:symmetry_map[j,i] =(i,j)

            return symmetry_map
        else:
            raise NotImplementedError

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

class PEPS_uniform_shape_symmetry_6x6(PEPS_uniform_shape_symmetry_base):
    '''
    same performance as PEPS_uniform_shape_symmetry_any(W=6,H=6    )
    '''
    def __init__(self, W=6,H=6,**kargs):
        assert (W == 6) and (H == 6)
        super().__init__(W=6,H=6,**kargs)
    def forward(self,input_data):
        bulk_tensors,edge_tensors,corn_tensors = self.get_batch_contraction_network(input_data)

        quater_contraction = self.einsum_engine(
            "xyzab,xycdb,xyefd,xygha,xyijch,xyklej,xymng,xyopin,xyqrkp->xyzmoqflr",
                corn_tensors[[ 0, 1, 2, 3]] ,
                edge_tensors[[ 0, 5,10,15]],
                edge_tensors[[ 1, 7, 8,14]],
                edge_tensors[[ 4, 3,12,11]],
                bulk_tensors[[ 0, 3,12,15]],
                bulk_tensors[[ 1, 7, 8,14]],
                edge_tensors[[ 6, 2,13, 9]],
                bulk_tensors[[ 4, 2,13,11]],
                bulk_tensors[[ 5, 6, 9,10]]
            ).flatten(-6,-4).flatten(-3,-1)
        # this is slighly different from the symmetry_any
        # the quater order is left-up , right_up, left-down,right-down
        # the real contraction order should ab-bc-da-cd
        # However, here is ab-bc-cd-da, this may cost problem.
        # However, when the provius layer is CNN, this problem could automatively eraser.

        tensor  = self.einsum_engine("lkmab,lknbc->lkmnac",
                              quater_contraction[[0,2]],quater_contraction[[1,3]]
                              ).flatten(-4,-3)# -> (2,B,O^2,D^3,D^3)
        tensor  = self.einsum_engine("kmab,knba->kmn",
                              tensor[0],tensor[1]
                              ).flatten(-2,-1)# -> (B,O^4)
        return tensor

class PEPS_uniform_shape_symmetry_any(PEPS_uniform_shape_symmetry_base):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        LW= self.LW
        LH= self.LH
        D = self.D
        O = self.O
        # if Wand
        tn2D_shape_list           = [ [(D,D,O)]+[  (D,D,D)]*(LH-1) ]+ \
                                    [ [(D,D,D)]+[(D,D,D,D)]*(LH-1)]*(LW-1)
        path,sublist_list,outlist = get_best_path(tn2D_shape_list,store=path_recorder,type='sub')
        last_idx = outlist.pop()
        outlist.insert(LH,last_idx)
        #print(outlist)
        #outlist should be [2, 6, 12, 18, 13, 15, 17] for 3x3
        self.sublist_list = sublist_list
        self.outlist      = outlist
        self.path         = path

        self.path_final= None
    def forward(self,input_data):
        LH = self.LH
        LW = self.LW
        bulk_tensors,edge_tensors,corn_tensors = self.get_batch_contraction_network(input_data)
        corn_tensors  = corn_tensors.permute(0,1,3,4,2) #(4BODD)->(4BDDO)
        tensor_list = self.pick_tensors(self.partrule,self.indexrule,corn_tensors,edge_tensors,bulk_tensors)
        assert len(tensor_list)==len(self.sublist_list)
        operands=[]
        for tensor,sublist in zip(tensor_list,self.sublist_list):
            operand = [tensor,[...,*sublist]]
            operands+=operand
        operands+= [[...,*(self.outlist)]]

        if self.cent_tensor_idx is not None:
            quater_contraction = self.einsum_engine(*operands,optimize=self.path).flatten(-2*LH-1,-LH-2).flatten(-LH,-1)
            #print(quater_contraction.shape)
            tensor  = self.einsum_engine("kvaub,kxbic,kycod,kzdpa,kuiop->kvxyz",
                                  *quater_contraction,bulk_tensors[self.cent_tensor_idx]
                                  ).flatten(-4,-1)# -> (2,B,O^4)
        else:
            quater_contraction = self.einsum_engine(*operands,optimize=self.path).flatten(-2*LW,-LW-1).flatten(-LW,-1)
            # now it is odd case
            tensor  = self.einsum_engine("kvab,kxbc,kycd,kzda->kvxyz",
                                  *quater_contraction
                                  ).flatten(-4,-1)# -> (B,O^4)
        return tensor

class PEPS_einsum_arbitrary_shape_optim(TN_Base):
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
        path  = read_path_from_offline(virtual_bond_dim)
        if path is None:
            print("no offline path find, we generate one, note not gurantee best")
            path,info = get_optim_path_by_oe_from_tn(node_list)
        else:
            print("offline path finded~!" )
            path = path['path']
        self.path         = path
        self.sublist_list = sublist_list
        self.outlist      = outlist


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
            batch_unit = torch.tensordot(_input,unit,dims=([-1], [0]))
            batch_input.append(batch_unit)

        operands=[]
        for tensor,sublist in zip(batch_input,self.sublist_list):
            operand = [tensor,[...,*sublist]]
            operands+=operand
        operands+= [[...,*self.outlist]]
        return self.einsum_engine(*operands,optimize=self.path)

from collections import OrderedDict
def reorder_neighbor(patch_json):
    #for patch_json in patch_json_list:
    info_per_point = patch_json['element']
    info_per_group = patch_json['node']
    info_per_line  = patch_json['line']

    x_max,y_max = np.array(list(info_per_point.keys())).max(0)
    processed_group    ={}
    for i in range(x_max + 1):
        for j in range(y_max + 1):
            real_group = info_per_point[i,j]['group']
            if real_group not in processed_group:
                ordered_group = len(processed_group)
                processed_group[real_group] = ordered_group
            else:
                ordered_group = processed_group[real_group]

    new_info_per_point ={}
    for key,val in info_per_point.items():
        new_val = copy.deepcopy(val)
        new_val['group'] = processed_group[new_val['group']]
        new_info_per_point[key]=new_val

    new_info_per_group ={}
    for key,val in info_per_group.items():
        new_val = copy.deepcopy(val)
        new_val['neighbor'] = [processed_group[g] for g in new_val['neighbor']]
        new_info_per_group[key]=new_val

    # we would share a same
    #
    new_info_per_line = {}
    for key,val in info_per_line.items():
        new_val = copy.deepcopy(val)
        new_key = (processed_group[key[0]],processed_group[key[1]])
        new_info_per_line[new_key]=new_val

    new_info_per_line_order = OrderedDict()
    for i in range(len(new_info_per_group) + 1):
        for j in range(len(new_info_per_group) + 1):
            if (i,j) in new_info_per_line:
                new_info_per_line_order[i,j]=new_info_per_line[i,j]
    return {'element':new_info_per_point,
             'node':new_info_per_group,
             'line':new_info_per_line_order}


class PEPS_einsum_arbitrary_partition_optim(TN_Base):
    def __init__(self,out_features=None,
                      virtual_bond_dim=None,#"models/arbitary_shape/arbitary_shape_16x9_2.json",
                      label_position=None,#=(8,4),
                       fixed_virtual_dim=None,
                       patch_engine=torch.nn.Identity,
                       symmetry=None,#"Z2_16x9",
                       seted_variation=10,
                       init_std=1e-10,
                       solved_std=None,
                       convertPeq1=False):
        super().__init__()
        assert out_features is not None
        assert virtual_bond_dim is not None
        assert label_position is not None
        self.convertPeq1  = convertPeq1
        self.out_features = out_features
        if isinstance(virtual_bond_dim,str):
            arbitary_shape_state_dict = torch.load(virtual_bond_dim)
        else:
            arbitary_shape_state_dict = virtual_bond_dim
        assert isinstance(arbitary_shape_state_dict,dict)


        info_per_group = arbitary_shape_state_dict['node']
        info_per_line  = arbitary_shape_state_dict['line']
        info_per_point = arbitary_shape_state_dict['element']



        #symmetry_map = FFT_Z2_symmetry_16x9_point
        symmetry_map,symmetry_map_point = self.generate_symmetry_map(symmetry,info_per_point)

        # if we use the symmetry, we need to make sure not only the tensor shape but also the
        # object that each tensor leg point. That is we need to realign the neighbor for each point.
        for group_now in range(len(info_per_group)):
            neighbors = info_per_group[group_now]['neighbor']# neighbors is also a group
            unique_id = [min(symmetry_map[neighbor_id]) for neighbor_id in neighbors]
            new_order = np.argsort(unique_id)
            #print(f"group-{group_now}: old neighbors{neighbors} real neighbor {unique_id} new order {new_order}")
            info_per_group[group_now]['neighbor'] = [neighbors[i] for i in new_order]
            elements  =  info_per_group[group_now]['element']
            unique_id = [min(symmetry_map_point[element]) for element in elements]
            new_order = np.argsort(unique_id)
            info_per_group[group_now]['element'] = [elements[i] for i in new_order]

        # if the element in group is symmetric, for example, if (15,0) and (1,0) in the same group, then the regist weight A(#Element,D,D,D,D)
        # should hold same symmetry along #Element dimension. That is, if we set (15,0) to the A[0](D,D,D,D) and (1,0) to the A[1](D,D,D,D),
        # then should have A[0]==A[1]
        for group_now,pool in info_per_group.items():
            in_symmetry_idx     = {}
            for idx, point in enumerate(pool['element']):
                in_symmetry_idx[idx]= [idx]
                for string_id in symmetry_map_point[point]:# the symmetry_map_point return string point position, for example '15-0'
                    real_point_pos = tuple([int(s) for s in string_id.split('-')])
                    if real_point_pos != point and real_point_pos in pool['element']: # exclude the itself, and its symmetry part in element list
                        the_idx_for_symmetry_point_in_element=pool['element'].index(real_point_pos)
                        in_symmetry_idx[idx].append(the_idx_for_symmetry_point_in_element)

            the_symmetry_permutation=[]
            has_symmetry_point = False
            for idx, point in enumerate(pool['element']):
                if len(in_symmetry_idx[idx])>1:has_symmetry_point=True
                if len(in_symmetry_idx[idx])>2:
                    raise NotImplezntedError("we now don't support the case that have more than 2 symmetric point in one group")
                the_symmetry_permutation.append(in_symmetry_idx[idx][-1])
            if has_symmetry_point:
                pool['in_group_symmetry_permutation'] = the_symmetry_permutation
            else:
                pool['in_group_symmetry_permutation'] = None

        if fixed_virtual_dim is not None:
            for key in info_per_line.keys():
                info_per_line[key]['D'] = fixed_virtual_dim
        #since we use symmetry than the center group could not in the symmetry part
        center_group = info_per_point[label_position]['group']
        damgling_num = len(info_per_group)


        info_per_group[center_group]['neighbor'].insert(0,damgling_num)
        info_per_line[(center_group,damgling_num)]={'D': out_features}
        symmetry_map[damgling_num]=set([damgling_num])
        operands = []
        sublist_list=[]

        ranks_list=[]

        #absolute_line_id = {}
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
                #if line_tuple not in absolute_line_id:absolute_line_id[line_tuple] = len(absolute_line_id)
                #idx = absolute_line_id[line_tuple]
                idx = list(info_per_line.keys()).index(line_tuple)

                ranks.append(D)
                sublist.append(idx)
            tensor = np.random.randn(*ranks)
            operands+=[tensor,[*sublist]]

            ranks_list.append(ranks)
            sublist_list.append(sublist)
        outlist  = [list(info_per_line.keys()).index((center_group,damgling_num))]
        # line_tuple = (center_group,damgling_num)
        # #if line_tuple not in absolute_line_id:absolute_line_id[line_tuple] = len(absolute_line_id)
        # idx = absolute_line_id[line_tuple]
        # outlist = [idx]
        operands+= [[*outlist]]


        self.operands_string = full_size_array_string(*operands)
        #path,info = oe.contract_path(*operands,optimize='random-greedy-128')
        path = get_best_path_via_oe(*operands,store="models/arbitrary_shape_path_recorder.json",re_saveQ=False)
        self.path         = path
        self.sublist_list = sublist_list
        self.outlist      = outlist


        unique_unit_list = []
        # the unique unit list is the real weight for training of the model
        # if there is no symmetry assign, then each group/block would allocate a weight
        # then the len(unique_unit_list) == len(info_per_group)
        # if there is symmetry, then some groups share the weight
        # then the len(unique_unit_list) < len(info_per_group)
        for i in range(len(sublist_list)):

            for the_symmetry_counterpart in symmetry_map[i]:
                if 'weight_idx' in info_per_group[the_symmetry_counterpart]:
                    info_per_group[i]['weight_idx']=info_per_group[the_symmetry_counterpart]["weight_idx"]
                    info_per_group[i]['symmetry_indices']=info_per_group[the_symmetry_counterpart]["symmetry_indices"]
                    break
            if 'weight_idx' in info_per_group[i]:
                continue
            # assume all element for the tensornetwork is indenpendent.
            # The bond (include physics) list is l0,l1,l2,...,ln
            # All element follow normal distribution X - sigma(0,alpha)
            # where alpha is the variation we need to calculated.
            # the output after contracting is also a tensor (may 1-rank scalar, 2-rank matrix, etc)
            # the element of the output follow the composite normal distribution Y - sigma(0,beta)
            # where beta = l0 x l1 x l2 x ... x ln x alpha^(# of tensor)

            # if we active the Z2 symmetry, not only the left and right part should be reflected, but also
            # the center part. Since the center part link two same region, then it should be the symmetry
            # matrix for the bond it connected.
            # we would firstly check the neighbor of the group
            unique_neighbor_group = [min(symmetry_map[neighbor_id]) for neighbor_id in info_per_group[i]['neighbor']]
            symmetry_indices=None
            for j in set(unique_neighbor_group):
                if unique_neighbor_group.count(j)>1:
                    symmetry_leg_pos = list(np.where(np.array(unique_neighbor_group)==j)[0]+1)# plus 1 since we would add physics dim
                    if symmetry_indices is not None:
                        raise NotImplementedError("we now only support one symmetry indices")
                    symmetry_indices = list(permutations((symmetry_leg_pos)))
            info_per_group[i]['symmetry_indices'] = symmetry_indices
            info_per_group[i]['weight_idx']       = len(unique_unit_list)


            shape = ranks_list[i]
            P     = len(info_per_group[i]['element'])
            if self.convertPeq1:
                if self.convertPeq1 == 'all_convert':
                    P = 2*P
                else:
                    P     = 2 if P==1 else P

            #control_mat = self.rde2D((P,*shape),0,physics_index=0,offset= 2 if i==center_group else 1)
            #bias_mat    = torch.normal(0,solved_std,(P,*shape))
            #bias_mat[control_mat.nonzero(as_tuple=True)]=0
            #unique_unit_list.append(control_mat+bias_mat)
            unique_unit_list.append(self.rde2D((P,*shape),init_std,offset=1))
            #unique_unit_list.append(torch.normal(mean=0,std=solved_std,size=(P,*shape)))
        #assert len(unit_list)==len(sublist_list)

        self.unique_unit_list         = [nn.Parameter(v) for v in unique_unit_list]
        self.unique_groupwise_backend = nn.ModuleList()
        center_group_unique_idx = info_per_group[center_group]["weight_idx"]
        for i,v in enumerate(unique_unit_list):
            if i == center_group_unique_idx:
                offset = 2
                inchan = out_features
            else:
                offset = 1
                inchan = 1
            self.unique_groupwise_backend.append(patch_engine(v.shape[offset:],inchan))
        for i, v in enumerate(self.unique_unit_list):self.register_parameter(f'unit_{i}', param=v)
        #put the tensor into the AD graph.
        #assert len(self.unit_list)==len(sublist_list)
        self.info_per_group=info_per_group
        self.info_per_line =info_per_line
        self.info_per_point=info_per_point
        self.symmetry_map  =symmetry_map
        self.symmetry_map_point =symmetry_map_point
        self.pre_activate_layer = nn.Identity()
        self.debugQ=False

    def generate_symmetry_map(self,symmetry,info_per_point):
        symmetry_map       = {}
        symmetry_map_point = {}
        for i,j in info_per_point.keys():
            group_now  = info_per_point[i,j]['group']
            symmetry_map[group_now]= set([group_now])
            symmetry_map_point[i,j]= set([f"{i}-{j}"])
        if symmetry is not None:
            if symmetry == "Z2_16x9":
                for i in list(range(1,8))+list(range(9,16)):
                    for j in range(9):
                        group_now  = info_per_point[i,j]['group']
                        group_sym  = info_per_point[16-i,j]['group']
                        symmetry_map[group_now]=symmetry_map[group_now]|set([group_sym])
                        symmetry_map_point[i,j]=symmetry_map_point[i,j]|set([f"{16-i}-{j}"])
            else:
                raise NotImplementedError
        return symmetry_map,symmetry_map_point

    def weight_init(self,method=None,set_var=1):
        if method is None:return
        if method == "Expecatation_Normalization":
            num_of_tensor   = len(self.info_per_group)
            list_of_virt_dim= [t['D'] for t in self.info_per_line.values()]
            # notice, we cannot count the class label
            list_of_virt_dim.append(1/self.out_features)
            list_of_phys_dim= [len(t['element']) for t in self.info_per_group.values()]
            divider         = np.prod([np.power(t,1/num_of_tensor) for t in list_of_phys_dim+list_of_virt_dim])
            solved_var      = np.power(set_var,1/num_of_tensor)/divider
            solved_std      = np.sqrt(solved_var)
            for i in range(len(self.unique_unit_list)):
                self.unique_unit_list[i] = torch.nn.init.normal_(self.unique_unit_list[i],mean=0.0, std=solved_std)
        elif method == "Expecatation_Normalization2":
            num_of_tensor   = len(self.info_per_group)
            list_of_virt_dim= [t['D'] for t in self.info_per_line.values()]
            # notice, we cannot count the class label
            list_of_virt_dim.append(1/self.out_features)
            list_of_phys_dim= [len(t['element']) for t in self.info_per_group.values()]
            divider         = np.prod([np.power(t,1/num_of_tensor) for t in list_of_virt_dim])
            solved_var      = np.power(set_var,1/num_of_tensor)/divider
            solved_std      = np.sqrt(solved_var)
            for i in range(len(self.unique_unit_list)):
                self.unique_unit_list[i] = torch.nn.init.normal_(self.unique_unit_list[i],mean=0.0, std=solved_std)
        elif method == "fixpoint_start":
            for i in range(len(self.unique_unit_list)):
                shape = self.unique_unit_list[i].shape
                offset= 1
                size_shape = shape[:offset]
                bias_shape = shape[offset:]
                max_dim    = max(bias_shape)
                full_rank  = len(bias_shape)
                half_rank  = full_rank//2
                rest_rank  = full_rank-half_rank
                bias_mat   = torch.eye(max_dim**half_rank,max_dim**rest_rank)
                bias_mat   = bias_mat.reshape(*([max_dim]*full_rank))
                for j,d in enumerate(bias_shape):
                    bias_mat   = torch.index_select(bias_mat, j,torch.arange(d))
                norm   = np.sqrt(np.prod(bias_shape))
                norm  *= np.sqrt(size_shape[0])
                #print(norm)

                bias_mat  /= norm
                bias_mat   = bias_mat.repeat(*size_shape,*([1]*len(bias_shape)))

                with torch.no_grad():
                    self.unique_unit_list[i] = torch.nn.init.normal_(self.unique_unit_list[i],mean=0.0, std=np.sqrt(set_var))
                    self.unique_unit_list[i].add_(bias_mat)
        elif method == "normlization_to_one":
            for i in range(len(self.unique_unit_list)):
                with torch.no_grad():
                    self.unique_unit_list[i].div_(torch.norm(self.unique_unit_list[i]))
        else:
            raise NotImplementedError(f"we dont have init option:{method}")

    def forward(self,input_data, only_return_input=False):
        #input data shape B,1,W,H
        assert len(input_data.shape)==4
        assert np.prod(input_data.shape[-2:])==len(self.info_per_point)
        input_data  = input_data.squeeze(1)

        _input = []
        _units = []
        #for i,((unit,symmetry_indices),point_idx) in enumerate(zip(unit_list,self.point_of_group)):
        for i in range(len(self.info_per_group)):
            #patch_idx  = self.info_per_group[i]['element_idx']
            #print(f"processing unit {i}: conclude point{patch_idx}")
            #batch_input= input_data[...,patch_idx] # B,P
            pool = self.info_per_group[i]
            point_idx = np.array(pool['element']).transpose()
            #print(info_per_group[i]["weight_idx"])
            unit             = self.unique_unit_list[pool["weight_idx"]]
            if pool['in_group_symmetry_permutation'] is not None:
                unit = (unit + unit[pool['in_group_symmetry_permutation']])/2
            unit_engine      = self.unique_groupwise_backend[pool["weight_idx"]]
            symmetry_indices = pool['symmetry_indices']

            x,y = point_idx
            batch_input= input_data[...,x,y] # B,P
            if self.convertPeq1:
                if self.convertPeq1 == 'all_convert':
                    batch_input = torch.cat([batch_input,1-batch_input],-1)
                elif batch_input.shape[-1]==1: #self.convertPeq1 == 'only convert for P=1':
                    batch_input = torch.cat([batch_input,1-batch_input],-1)
                else:
                    pass
            # the correct way to follow the sprit is symmetry the weight
            # however, if there is no further processing, it is same to so contractrion than do symmetry.
            # what's more, if we use further processing, it much more convience to do symmetriy later rather than
            # create a symmetric further processing for symmetric weight.
            #if symmetry_indices is not None:
            #    unit_sys = unit
            #    for symmetry_indice in symmetry_indices[1:]:
            #        unit_sys = unit_sys + unit.transpose(*symmetry_indice)
            #    unit = unit_sys/len(symmetry_indices)
            #_units.append(unit)
            #_input.append(batch_input)

            batch_unit = torch.tensordot(batch_input,unit,dims=([-1], [0]))
            batch_unit = self.pre_activate_layer(batch_unit)
            #print(f"{batch_unit.shape}->",end=' ')
            batch_unit = unit_engine(batch_unit)

            #print(f"{batch_unit.shape}",end='\n')
            #print(f"{batch_input.norm()}-{unit.norm()}->{batch_unit.norm()}")
            #print(batch_unit.shape)
            if symmetry_indices is not None:
                unit_sys = batch_unit
                for symmetry_indice in symmetry_indices[1:]:
                    unit_sys = unit_sys + batch_unit.transpose(*symmetry_indice)
                batch_unit = unit_sys/len(symmetry_indices)
            if self.debugQ:
                std,mean = torch.std_mean(batch_unit)
                print(f'patch_{i}: std:{std.item()} mean:{mean.item()}')
            _input.append(batch_unit)
        if only_return_input:
            return _input
        #return _input,_units
        operands=[]
        for tensor,sublist in zip(_input,self.sublist_list):
            operand = [tensor,[...,*sublist]]
            operands+=operand
        operands+= [[...,*self.outlist]]
        return self.einsum_engine(*operands,optimize=self.path)

    def set_alpha(self,alpha):
        for backend in self.unique_groupwise_backend:
            #if hasattr(backend,'set_alpha'):
            backend.set_alpha(alpha)
import copy
class PEPS_aggregation_model(TN_Base):
    def __init__(self,out_features=None,virtual_bond_dim=None,label_position=None,#=(8,4),
                      fixed_virtual_dim=None,
                       patch_engine=torch.nn.Identity,
                       alpha_list=[],
                       symmetry=None,#"Z2_16x9",
                       seted_variation=10,
                       init_std=1e-10,
                       solved_std=None,
                       convertPeq1=False):
        super().__init__()
        assert out_features is not None
        assert virtual_bond_dim is not None
        assert label_position is not None
        assert isinstance(fixed_virtual_dim,int)
        if isinstance(virtual_bond_dim,str):
            virtual_bond_dim = torch.load(virtual_bond_dim)
        else:
            virtual_bond_dim = virtual_bond_dim
        assert isinstance(virtual_bond_dim,list)
        assert isinstance(virtual_bond_dim[0],dict)
        self.out_features  = out_features
        self.label_position= label_position
        self.fixed_virtual_dim=fixed_virtual_dim
        self.patch_engine     =patch_engine
        self.alpha_list       =alpha_list
        self.symmetry         =symmetry
        self.seted_variation  =seted_variation
        self.init_std         =init_std
        self.solved_std       =solved_std
        self.convertPeq1      =convertPeq1

        self.modules_list  = torch.nn.ModuleList()

        # if you want to use aggregation model, then all the patch config should follow the
        # same group order, for example, group should be arranged like  group0, group1,group2
        #                                                               group3, group4,group5
        #                                                               .....
        # otherwise, they won't share the optim path and won't share same _input list

        ### reorder group
        virtual_bond_dim = [reorder_neighbor(p) for p in virtual_bond_dim]
        # for patch_json in virtual_bond_dim:
        #     info_per_point = patch_json['element']
        #     info_per_group = patch_json['node']
        #
        #     x_max,y_max = np.array(list(info_per_point.keys())).max(0)
        #     info_per_group_new = {}
        #     processed_group={}
        #     for i in range(x_max + 1):
        #         for j in range(y_max + 1):
        #             real_group = info_per_point[i,j]['group']
        #             if real_group not in processed_group:
        #                 ordered_group = len(info_per_group_new)
        #                 info_per_group_new[ordered_group] = info_per_group[real_group]
        #                 processed_group[real_group] = ordered_group
        #             else:
        #                 ordered_group = processed_group[real_group]
        #             info_per_point[i,j]['group'] = ordered_group
        #     patch_json['node'] =  info_per_group_new

        self.virtual_bond_dim = virtual_bond_dim
        if isinstance(alpha_list,int):alpha_list=[alpha_list]*len(virtual_bond_dim)
        if len(alpha_list) ==0: alpha_list=[3]*len(virtual_bond_dim)
        assert len(alpha_list) == len(virtual_bond_dim)
        self.alpha_list = alpha_list
        for alpha,virtual_bond in zip(alpha_list,virtual_bond_dim):
            self.modules_list.append(PEPS_einsum_arbitrary_partition_optim(out_features=out_features,
                                                                           fixed_virtual_dim=fixed_virtual_dim,
                                                                           virtual_bond_dim = copy.deepcopy(virtual_bond),
                                                                           label_position = label_position,
                                                                           patch_engine   = lambda *arg:patch_engine(*arg,alpha=alpha),
                                                                           symmetry=symmetry,
                                                                           seted_variation=seted_variation,
                                                                           init_std=init_std,
                                                                           solved_std=solved_std,
                                                                           convertPeq1=convertPeq1))
        path_array_string_store={}
        for layer_id,layer in enumerate(self.modules_list):
            if layer.operands_string not in path_array_string_store:path_array_string_store[layer.operands_string]=[]
            path_array_string_store[layer.operands_string].append(layer_id)
        # if len(path_array_string_store)>1:
        #     print("the submodel you setup have different operands path: see below:")
        #     for string, layer_id_list in path_array_string_store.items():
        #         print("==========================")
        #         print(f"for layer {layer_id_list}, you get oprands string:")
        #         print(string)
        #         print("==========================")
            # usually, the operands string for different module is different.
            # but you will find, they actually are same since we have symmetry.
            #raise NotImplementedError


    def weight_init(self,*args,**kargs):
        for sub_model in self.modules_list:
            sub_model.weight_init(*args,**kargs)
    def forward(self,input_data,do_average=True):
        assert len(input_data.shape)==4
        assert np.prod(input_data.shape[-2:])==len(self.modules_list[0].info_per_point)
        all_inputs  = [module(input_data,only_return_input=True) for module in self.modules_list]
        all_inputs = [torch.stack([t[i] for t in all_inputs],1) for i in range(len(all_inputs[0]))]

        operands=[]
        for tensor,sublist in zip(all_inputs,self.modules_list[0].sublist_list):
            operand = [tensor,[...,*sublist]]
            operands+=operand
        operands+= [[...,*self.modules_list[0].outlist]]
        out = self.einsum_engine(*operands,optimize=self.modules_list[0].path)
        if do_average:
            out = out.mean(1)
        return out
    def get_alpha_list(self):
        alpha_list = np.linspace(0.2,2,20)
        for config_id,structure_config in enumerate(copy.deepcopy(self.virtual_bond_dim)):
            std_record=[]
            for alpha in alpha_list:
                model=PEPS_einsum_arbitrary_partition_optim(out_features    = self.out_features,
                                                           fixed_virtual_dim= self.fixed_virtual_dim,
                                                           virtual_bond_dim = copy.deepcopy(structure_config),
                                                           label_position   = self.label_position,
                                                           patch_engine     = lambda *arg:self.patch_engine(*arg,alpha=alpha),
                                                           symmetry         = self.symmetry,
                                                           seted_variation  = self.seted_variation,
                                                           init_std         = self.init_std,
                                                           solved_std       = self.solved_std,
                                                           convertPeq1      = self.convertPeq1
                                                           )
                model.weight_init(method="Expecatation_Normalization2")
                device = 'cuda'
                model  = model.to(device).eval()
                with torch.no_grad():
                    std_list = []
                    for _ in range(10):
                         std_list.append(torch.std(model(torch.randn(100,1,16,9).cuda())).item())
                std_record.append(np.mean(std_list))
            headers_str = [str(b) for b in alpha_list]
            data = np.array([std_record])
           #tp.table(data, headers_str)
            x = alpha_list
            y = std_record
            z1 = np.polyfit(x, y, 6) #用3次多项式拟合，输出系数从高到0
            p1 = np.poly1d(z1) #使用次数合成多项式
            y_pre = p1(x)
            #print("拟合结果: y = %10.5f x + %10.5f " % (a,b) )
            print(f"id:{config_id}:best alpha {[t.real for t in np.roots(z1-2) if np.isreal(t) and t>0][0]}")

from .tensornetwork_base import *
from .model_utils import *
import opt_einsum as oe
import numpy as np

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
        input_data = input_data.flatten(1,2)
        bulk_input = batch_image_input[...,bulk_idxes,:]
        edge_input = batch_image_input[...,edge_idxes,:]
        corn_input = batch_image_input[...,corn_idxes,:]

        cent_input   = corn_input[:, -1]
        corn_input   = corn_input[:,:-1]
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
        #corner_contraction
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
                       init_std=1e-10,symmetry = None):
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
        self.active_index = active_index
        self.corn_tensors = nn.Parameter(self.rde2D( ([tensor_needed[0]],O,P,D,D)  ,init_std, offset=3))
        self.edge_tensors = nn.Parameter(self.rde2D( ([tensor_needed[1]],P,D,D,D)  ,init_std, offset=2))
        self.bulk_tensors = nn.Parameter(self.rde2D( ([tensor_needed[2]],P,D,D,D,D),init_std, offset=2))

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
        corn_tensors = self.einsum_engine(" lopab,klp->lkoab" ,self.corn_tensors[[self.active_index[0]]],corn_input)
        edge_tensors = self.einsum_engine(" lpabc,klp->lkabc" ,self.edge_tensors[[self.active_index[1]]],edge_input)
        bulk_tensors = self.einsum_engine("lpabcd,klp->lkabcd",self.bulk_tensors[[self.active_index[2]]],bulk_input)
        return bulk_tensors,edge_tensors,corn_tensors

    def generate_symmetry_map(self,W,H,symmetry='R4'):
        assert W%2 == 0 and W == H
        LW =  int(np.ceil(1.0*W/2))
        LH = int(np.floor(1.0*H/2))
        index_matrix = torch.LongTensor([[[i,j] for j in range(W)] for i in range(H)])
        if symmetry == 'R4':
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
        elif symmetry == 'R4Z2':
            symmetry_map = self.generate_symmetry_map(W,H,symmetry='R4')
            part_lu_idex = torch.rot90(index_matrix,k=0)[:LW,:LH].flatten(0,1).numpy()
            for i,j in part_lu_idex:
                if (j,i) not in symmetry_map and j>i:symmetry_map[j,i] =(i,j)
            return symmetry_map

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

# class PEPS_uniform_shape_symmetry_any_old(PEPS_uniform_shape_symmetry_base):
#     def __init__(self, **kargs):
#         super().__init__(**kargs)
#         W = self.W;
#         H = self.H
#         O = self.O
#         P = self.P
#         D = self.D
#         self.LW = LW = W//2
#         self.LH = LH = H//2
#         tn2D_shape_list                = [ [(D,D,O)]+[  (D,D,D)]*(LH-1) ]+ \
#                                          [ [(D,D,D)]+[(D,D,D,D)]*(LH-1)]*(LW-1)
#         path,sublist_list,outlist = get_best_path(tn2D_shape_list,store=path_recorder,type='sub')
#         #node_list,sublist_list,outlist = sub_network_tn(tn2D_shape_list)
#         #path,info                      = get_optim_path_by_oe_from_tn(node_list)
#         last_idx = outlist.pop()
#         outlist.insert(LH,last_idx)
#         #print(outlist)
#         #outlist should be [2, 6, 12, 18, 13, 15, 17] for 3x3
#         self.sublist_list = sublist_list
#         self.outlist      = outlist
#         self.path         = path
#
#         self.path_record  = {}
#         self.path_final= None
#     def forward(self,input_data):
#         bulk_input,edge_input,corn_input,cent_input = self.flatten_image_input(input_data)
#         corn_input   = torch.cat([corn_input,cent_input.unsqueeze(1)],1)
#         bulk_tensors = self.einsum_engine("lpabcd,klp->lkabcd",self.bulk_tensors,bulk_input)
#         edge_tensors = self.einsum_engine(" lpabc,klp->lkabc" ,self.edge_tensors,edge_input)
#         corn_tensors = self.einsum_engine("  lopab,klp->lkabo" ,self.corn_tensors,corn_input)
#         L = len(bulk_tensors)
#         remain = bulk_tensors.shape[1:]
#         bulk_tensors = bulk_tensors.reshape(4,L//4,*remain)
#
#         L = len(edge_tensors)
#         remain = edge_tensors.shape[1:]
#         edge_tensors = edge_tensors.reshape(4,L//4,*remain)
#
#         LH = self.LH
#         LW = self.LW
#         tensor_list  =[[corn_tensors]        + list(edge_tensors[:LH-1]) ]+\
#                       [[edge_tensors[LH-1+i]]+ list(bulk_tensors[(LH-1)*i:(LH-1)*(i+1)])
#                                                                     for i in range(LW-1)]
#         tensor_list     = [l for t in tensor_list for l in t]
#         assert len(tensor_list)==len(self.sublist_list)
#         operands=[]
#         for tensor,sublist in zip(tensor_list,self.sublist_list):
#             operand = [tensor,[...,*sublist]]
#             operands+=operand
#         operands+= [[...,*(self.outlist)]]
#
#         # in this case W==H , LW==LH
#
#         quater_contraction = self.einsum_engine(*operands,optimize=self.path).flatten(-2*LW,-LW-1).flatten(-LW,-1)
#
#         tensor  = self.einsum_engine("lkmab,lknbc->lkmnac",
#                               quater_contraction[[0,2]],quater_contraction[[1,3]]
#                               ).flatten(-4,-3)# -> (2,B,O^2,D^3,D^3)
#         tensor  = self.einsum_engine("kmab,knba->kmn",
#                               tensor[0],tensor[1]
#                               ).flatten(-2,-1)# -> (B,O^4)
#         return tensor


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

class PEPS_uniform_shape_symmetry_deep_model(PEPS_uniform_shape_symmetry_base):
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

class PEPS_uniform_shape_rotation90_even(TN_Base):
    def __init__(self, W=6,H=6,out_features=16,
                       in_physics_bond = 3, virtual_bond_dim=3,
                       init_std=1e-10):
        super().__init__()
        assert (W == H)
        assert (W%2 == 0)
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


        self.index_matrix = index_matrix = torch.LongTensor([[[i,j] for j in range(W)] for i in range(H)])
        bulk_index,edge_index,corn_index=self.flatten_image_input(index_matrix)
        flag_matrix     = torch.zeros(W,H).long()
        position_matrix = torch.zeros(W,H).long()
        symmetry_map    = self.generate_symmetry_map(W,H,symmetry='R4')
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

        self.corn_tensors = nn.Parameter(self.rde2D( ([tensor_needed[0]],O,P,D,D)  ,init_std, offset=3))
        self.edge_tensors = nn.Parameter(self.rde2D( ([tensor_needed[1]],P,D,D,D)  ,init_std, offset=2))
        self.bulk_tensors = nn.Parameter(self.rde2D( ([tensor_needed[2]],P,D,D,D,D),init_std, offset=2))

        self.active_index = active_index
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
        self.flag_matrix     = flag_matrix
        self.position_matrix = position_matrix
        self.real_pos_for_id = real_pos_for_id
        self.cent_tensor_idx = position_matrix[self.W//2,self.H//2] if self.W%2==1 else None


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

class PEPS_einsum_arbitrary_partition_optim(TN_Base):
    def __init__(self,out_features=10,in_physics_bond = 2, virtual_bond_config="models/arbitary_shape/arbitary_shape_1.json",
                       bias=True,label_position='center',contraction_mode = 'recursion',
                 init_std=1e-10,
                 seted_variation=10,
                 solved_std=None):
        super().__init__()
        if isinstance(virtual_bond_config,str):
            arbitary_shape_state_dict = torch.load(virtual_bond_config)
        else:
            arbitary_shape_state_dict = virtual_bond_config
        assert isinstance(arbitary_shape_state_dict,dict)


        info_per_group = arbitary_shape_state_dict['node']
        info_per_line  = arbitary_shape_state_dict['line']
        info_per_point = arbitary_shape_state_dict['element']

        if solved_std is None:
            num_of_tensor   = len(info_per_group)
            list_of_virt_dim= [t['D'] for t in info_per_line.values()]
            list_of_phys_dim= [len(t['element']) for t in info_per_group.values()]
            divider         = np.prod([np.power(t,1/num_of_tensor) for t in list_of_phys_dim+list_of_virt_dim])
            solved_var      = np.power(seted_variation,1/num_of_tensor)/divider
            solved_std      = np.sqrt(solved_var)


        center_group    = 0
        damgling_num    = len(info_per_group)
        info_per_group[center_group]['neighbor'].insert(0,damgling_num)
        info_per_line[(center_group,damgling_num)]={'D': out_features}

        self.info_per_group=info_per_group
        self.info_per_line =info_per_line
        self.info_per_point=info_per_point

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
        path = self.get_best_contracting_path(*operands)
        #path,info = oe.contract_path(*operands,optimize='random-greedy-128')
        self.path         = path
        self.sublist_list = sublist_list
        self.outlist      = outlist

        # assume all element for the tensornetwork is indenpendent.
        # The bond (include physics) list is l0,l1,l2,...,ln
        # All element follow normal distribution X - sigma(0,alpha)
        # where alpha is the variation we need to calculated.
        # the output after contracting is also a tensor (may 1-rank scalar, 2-rank matrix, etc)
        # the element of the output follow the composite normal distribution Y - sigma(0,beta)
        # where beta = l0 x l1 x l2 x ... x ln x alpha^(# of tensor)

        unit_list = []
        for i in range(len(sublist_list)):
            shape = ranks_list[i]
            P        = len(info_per_group[i]['element'])
            #control_mat = self.rde2D((P,*shape),0,physics_index=0,offset= 2 if i==center_group else 1)
            #bias_mat    = torch.normal(0,solved_std,(P,*shape))
            #bias_mat[control_mat.nonzero(as_tuple=True)]=0
            #unit_list.append(control_mat+bias_mat)
            #unit_list.append(self.rde2D((P,*shape),init_std,offset=1))
            unit_list.append(torch.normal(mean=0,std=solved_std,size=(P,*shape)))
        assert len(unit_list)==len(sublist_list)

        self.unit_list = [nn.Parameter(v) for v in unit_list]
        for i, v in enumerate(self.unit_list):
            self.register_parameter(f'unit_{i}', param=v)

    def forward(self,input_data):
        #input data shape B,1,W,H
        assert len(input_data.shape)==4
        input_data  = input_data.flatten(-3,-1)

        _input = []
        for i,unit in enumerate(self.unit_list):
            patch_idx  = self.info_per_group[i]['element_idx']
            batch_input= input_data[...,patch_idx] # B,P
            batch_unit = torch.tensordot(batch_input,unit,dims=([-1], [0]))
            #print(f"{batch_input.norm()}-{unit.norm()}->{batch_unit.norm()}")
            _input.append(batch_unit)

        operands=[]
        for tensor,sublist in zip(_input,self.sublist_list):
            operand = [tensor,[...,*sublist]]
            operands+=operand
        operands+= [[...,*self.outlist]]
        return self.einsum_engine(*operands,optimize=self.path)

from models.tensornetwork_base import *

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

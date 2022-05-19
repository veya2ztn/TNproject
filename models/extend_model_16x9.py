from .extend_model import *

class PEPS_16x9_Z2_Binary_Wrapper:
    def __init__(self,module,structure_path,alpha=3,fixed_virtual_dim=None):
        self.module  = module
        self.structure_path = structure_path
        self.fixed_virtual_dim = fixed_virtual_dim
        self.__name__ = f"PEPS_16x9_Z2_Binary_{module.__class__.__name__}"
        self.alpha = alpha
    def __call__(self,alpha=None,**kargs):
        if alpha is None:alpha = self.alpha
        module = lambda *args:self.module(*args,alpha=alpha)
        model=PEPS_einsum_arbitrary_partition_optim(virtual_bond_dim=self.structure_path,
                                                    label_position=(8,4),fixed_virtual_dim=self.fixed_virtual_dim,
                                                    symmetry="Z2_16x9",
                                                    patch_engine=module,
                                                    **kargs
                                               )
        model.weight_init(method="Expecatation_Normalization2")
        return model
class PEPS_16x9_Z2_Binary_Aggregation_Wrapper:
    def __init__(self,module,structure_path,alpha_list=1,fixed_virtual_dim=5,convertPeq1=True):
        self.module  = module
        self.structure_path = structure_path
        self.fixed_virtual_dim = fixed_virtual_dim
        self.convertPeq1 = convertPeq1
        self.__name__ = f"PEPS_16x9_Z2_Binary_{module.__class__.__name__}"
        self.alpha_list = alpha_list
    def __call__(self,alpha_list=None,**kargs):
        if alpha_list is None:alpha_list = self.alpha_list
        model=PEPS_aggregation_model(
                                   virtual_bond_dim=self.structure_path,
                                   label_position=(8,4),
                                   symmetry="Z2_16x9",
                                   patch_engine=self.module,
                                   alpha_list=alpha_list,
                                   fixed_virtual_dim=self.fixed_virtual_dim,
                                   convertPeq1=self.convertPeq1,**kargs
                                  )
        model.weight_init(method="Expecatation_Normalization2")
        return model


PEPS_16x9_Z2_Binary_CNNS_2    = PEPS_16x9_Z2_Binary_Wrapper(TensorNetConvND_Single,#ops=104,paras=169121
                                                            "models/arbitary_shape/arbitary_shape_16x9_2.json",fixed_virtual_dim=None,alpha=3)
PEPS_16x9_Z2_Binary_CNNS_2_v4 = PEPS_16x9_Z2_Binary_Wrapper(TensorNetConvND_Single,#ops=104,paras=22180
                                                            "models/arbitary_shape/arbitary_shape_16x9_2.json",fixed_virtual_dim=4,alpha=1.1)
PEPS_16x9_Z2_Binary_CNNS_3    = PEPS_16x9_Z2_Binary_Wrapper(TensorNetConvND_Single,#ops=130,paras=172408
                                                            "models/arbitary_shape/arbitary_shape_16x9_3.json",fixed_virtual_dim=None,alpha=3)
PEPS_16x9_Z2_Binary_CNNS_3_v5 = PEPS_16x9_Z2_Binary_Wrapper(TensorNetConvND_Single,#ops=130,paras=55360
                                                            "models/arbitary_shape/arbitary_shape_16x9_3.json",fixed_virtual_dim=5,alpha=4)
PEPS_16x9_Z2_Binary_CNNS_4    = PEPS_16x9_Z2_Binary_Wrapper(TensorNetConvND_Single,#ops=56,paras=181175
                                                            "models/arbitary_shape/arbitary_shape_16x9_4.json",fixed_virtual_dim=None,alpha=1.5)
PEPS_16x9_Z2_Binary_CNNS_5    = PEPS_16x9_Z2_Binary_Wrapper(TensorNetConvND_Single,#ops=70,paras=113788
                                                            "models/arbitary_shape/arbitary_shape_16x9_5.json",fixed_virtual_dim=None,alpha=1.5)
PEPS_16x9_Z2_Binary_CNNS_6    = PEPS_16x9_Z2_Binary_Wrapper(TensorNetConvND_Single,#ops=70,paras=113788
                                                            "models/arbitary_shape/arbitary_shape_16x9_5.json",fixed_virtual_dim=None,alpha=2.5)
PEPS_16x9_Z2_Binary_CNNS_7    = PEPS_16x9_Z2_Binary_Wrapper(TensorNetConvND_Single,#ops=104,paras=172160
                                                            "models/arbitary_shape/arbitary_shape_16x9_7.json",fixed_virtual_dim=None)
PEPS_16x9_Z2_Binary_CNNS_13_v4= PEPS_16x9_Z2_Binary_Wrapper(TensorNetConvND_Single,#ops=70,paras=14883
                                                "models/arbitary_shape/arbitary_shape_16x9_7.json",fixed_virtual_dim=None,alpha=1)

# compate for old model name
PEPS_16x9_Z2_Binary_CNN_7    = PEPS_16x9_Z2_Binary_CNNS_7
PEPS_16x9_Z2_Binary_CNN_0    = PEPS_16x9_Z2_Binary_CNNS_2
PEPS_16x9_Z2_Binary_CNN_0_v4 = PEPS_16x9_Z2_Binary_CNNS_2

PEPS_16x9_Z2_Binary_CNNA_2    = PEPS_16x9_Z2_Binary_Wrapper(TensorNetConvND_Block_a,"models/arbitary_shape/arbitary_shape_16x9_2.json",fixed_virtual_dim=None,alpha=3.5)
# compate for old model name
PEPS_16x9_Z2_Binary_CNN_1    = PEPS_16x9_Z2_Binary_CNNA_2

PEPS_16x9_Z2_Binary_CNN_Aggregation_19_3    = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorNetConvND_Single,#ops=1064,paras= 691885
                                                                "models/arbitary_shape/patch_partions_3colum_max45raw_json_list.pt", alpha_list = 1)
PEPS_16x9_Z2_Binary_CNN_Aggregation_12_3    = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorNetConvND_Single,#ops= 672,paras= 410730
                                                                "models/arbitary_shape/patch_partions_3column_12units.pt", alpha_list = 1)
PEPS_16x9_Z2_Binary_CNN_Aggregation_12_3_v3 = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorNetConvND_Single,#ops= 672,paras=  68850
                                                                "models/arbitary_shape/patch_partions_3column_12units.pt", alpha_list = 0.6,fixed_virtual_dim=3)
PEPS_16x9_Z2_Binary_CNN_Aggregation_28_3    = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorNetConvND_Single,#ops=1568,paras=1019620
                                                                "models/arbitary_shape/patch_partions_3column_28units.pt", alpha_list = 1)
PEPS_16x9_Z2_Binary_CNN_Aggregation_19_5    = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorNetConvND_Single,#ops=1976,paras=1114447
                                                                "models/arbitary_shape/patch_partions_5colum_max45raw_json_list.pt", alpha_list = 2)
PEPS_16x9_Z2_Binary_CNN_Aggregation_12_5    = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorNetConvND_Single,#ops=1248,paras=668006
                                                                "models/arbitary_shape/patch_partions_5column_12units.pt", alpha_list = 2)
PEPS_16x9_Z2_Binary_CNN_Aggregation_28_5    = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorNetConvND_Single,#ops=2912,paras=1644514
                                                                "models/arbitary_shape/patch_partions_5column_28units.pt", alpha_list = 2)

PEPS_16x9_Z2_Binary_CNN_Aggregation = PEPS_16x9_Z2_Binary_CNN_Aggregation_19_3

PEPS_16x9_Z2_Binary_TAT_2     = PEPS_16x9_Z2_Binary_TA_0    = PEPS_16x9_Z2_Binary_Wrapper(TensorAttention,"models/arbitary_shape/arbitary_shape_16x9_2.json",fixed_virtual_dim=5,alpha=0.05)


PEPS_16x9_Z2_Binary_TAT_2_v3 = PEPS_16x9_Z2_Binary_Wrapper(TensorAttention,#ops=260,paras= 26217
                                                          "models/arbitary_shape/arbitary_shape_16x9_2.json",fixed_virtual_dim=3,alpha=0.1)
PEPS_16x9_Z2_Binary_TAT_2_v4 = PEPS_16x9_Z2_Binary_Wrapper(TensorAttention,#ops=260,paras= 79104
                                                          "models/arbitary_shape/arbitary_shape_16x9_2.json",fixed_virtual_dim=4,alpha=0.05)
PEPS_16x9_Z2_Binary_TAT_3_v4 = PEPS_16x9_Z2_Binary_Wrapper(TensorAttention,#ops=325,paras= 95424
                                                          "models/arbitary_shape/arbitary_shape_16x9_3.json",fixed_virtual_dim=3,alpha=0.06)
PEPS_16x9_Z2_Binary_TAT_3_v5 = PEPS_16x9_Z2_Binary_Wrapper(TensorAttention,#ops=325,paras=227625
                                                          "models/arbitary_shape/arbitary_shape_16x9_3.json",fixed_virtual_dim=4,alpha=0.02)
PEPS_16x9_Z2_Binary_TAT_4_v4 = PEPS_16x9_Z2_Binary_Wrapper(TensorAttention,#ops=156,paras= 50688
                                                          "models/arbitary_shape/arbitary_shape_16x9_4.json",fixed_virtual_dim=4,alpha=0.2)
PEPS_16x9_Z2_Binary_TAT_5_v4 = PEPS_16x9_Z2_Binary_Wrapper(TensorAttention,#ops=195,paras= 59328
                                                          "models/arbitary_shape/arbitary_shape_16x9_5.json",fixed_virtual_dim=4,alpha=0.13)
PEPS_16x9_Z2_Binary_TAT_6_v4 = PEPS_16x9_Z2_Binary_Wrapper(TensorAttention,#ops=260,paras= 80640
                                                          "models/arbitary_shape/arbitary_shape_16x9_6.json",fixed_virtual_dim=4,alpha=0.07)
PEPS_16x9_Z2_Binary_TAT_7_v4 = PEPS_16x9_Z2_Binary_Wrapper(TensorAttention,#ops=260,paras= 80640
                                                          "models/arbitary_shape/arbitary_shape_16x9_7.json",fixed_virtual_dim=4,alpha=0.05)
PEPS_16x9_Z2_Binary_TAT_13_v4= PEPS_16x9_Z2_Binary_Wrapper(TensorAttention,#ops=195,paras= 59328
                                                          "models/arbitary_shape/arbitary_shape_16x9_13.json",fixed_virtual_dim=4,alpha=0.15)


PEPS_16x9_Z2_Binary_TAT_Aggregation_12_3    = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorAttention,#ops=1872,paras=1443450
                                                                "models/arbitary_shape/patch_partions_3column_12units.pt", alpha_list = 0.13)
PEPS_16x9_Z2_Binary_TAT_Aggregation_12_3_v2 = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorAttention,#ops=1872,paras=  46272
                                                                "models/arbitary_shape/patch_partions_3column_12units.pt", alpha_list = 0.23,fixed_virtual_dim=2)
PEPS_16x9_Z2_Binary_TAT_Aggregation_12_3_v3 = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorAttention,#ops=1872,paras= 204930
                                                                "models/arbitary_shape/patch_partions_3column_12units.pt", alpha_list = 0.2,fixed_virtual_dim=3)
PEPS_16x9_Z2_Binary_TAT_Aggregation_19_3    = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorAttention,#ops=2964,paras=2327025
                                                                "models/arbitary_shape/patch_partions_3colum_max45raw_json_list.pt", alpha_list = 0.15)
PEPS_16x9_Z2_Binary_TAT_Aggregation_19_3_v3 = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorAttention,#ops=2964,paras= 330885
                                                                "models/arbitary_shape/patch_partions_3colum_max45raw_json_list.pt", alpha_list = 0.15,fixed_virtual_dim=3)
PEPS_16x9_Z2_Binary_TAT_Aggregation_28_3    = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorAttention,#ops=4368,paras=3429300
                                                                "models/arbitary_shape/patch_partions_3column_28units.pt", alpha_list = 0.1)
PEPS_16x9_Z2_Binary_TAT_Aggregation_28_3_v3 = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorAttention,#ops=4368,paras= 487620
                                                                "models/arbitary_shape/patch_partions_3column_28units.pt", alpha_list = 0.17,fixed_virtual_dim=3)


PEPS_16x9_Z2_Binary_TAT_Aggregation_19_5    = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorAttention,
                                                                "models/arbitary_shape/patch_partions_5colum_max45raw_json_list.pt", alpha_list = 1)
PEPS_16x9_Z2_Binary_TAT_Aggregation_19_5_v3 = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorAttention,
                                                                "models/arbitary_shape/patch_partions_5colum_max45raw_json_list.pt", alpha_list = 1,fixed_virtual_dim=3)
PEPS_16x9_Z2_Binary_TAT_Aggregation_12_5    = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorAttention,
                                                                "models/arbitary_shape/patch_partions_5column_12units.pt", alpha_list = 2)
PEPS_16x9_Z2_Binary_TAT_Aggregation_28_5    = PEPS_16x9_Z2_Binary_Aggregation_Wrapper(TensorAttention,
                                                                "models/arbitary_shape/patch_partions_5column_28units.pt", alpha_list = 2)


def PEPS_16x9_Z2_Binary_CNN_full(**kargs):
    model=PEPS_einsum_arbitrary_partition_optim(out_features=1,
                                            virtual_bond_dim="models/arbitary_shape/arbitary_shape_16x9_full.json",
                                            label_position=(8,4),
                                            symmetry="Z2_16x9",
                                            #patch_engine=self.module,
                                            fixed_virtual_dim=2, # if D=3 require 18G for inference
                                            convertPeq1=True)
    model.weight_init(method="Expecatation_Normalization")
    model.pre_activate_layer =scale_sigmoid()
    return model

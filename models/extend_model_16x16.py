from .extend_model import *

class PEPS_16x16_Z2_Binary_Wrapper:
    def __init__(self,module,structure_path,alpha=3,fixed_virtual_dim=None):
        self.module  = module
        self.structure_path = structure_path
        self.fixed_virtual_dim = fixed_virtual_dim
        self.__name__ = f"PEPS_16x16_Z2_Binary_{module.__class__.__name__}"
        self.alpha = alpha
    def __call__(self,alpha=None,**kargs):
        if alpha is None:alpha = self.alpha
        module = lambda *args:self.module(*args,alpha=alpha)
        model=PEPS_einsum_arbitrary_partition_optim(virtual_bond_dim=self.structure_path,
                                                    label_position="no_center",fixed_virtual_dim=self.fixed_virtual_dim,
                                                    symmetry="Z2_16x16",
                                                    patch_engine=module,
                                                    **kargs
                                               )
        model.weight_init(method="Expecatation_Normalization2")
        return model
class PEPS_16x16_Z2_Binary_Aggregation_Wrapper:
    def __init__(self,module,structure_path,alpha_list=1,fixed_virtual_dim=5,convertPeq1=True):
        self.module  = module
        self.structure_path = structure_path
        self.fixed_virtual_dim = fixed_virtual_dim
        self.convertPeq1 = convertPeq1
        self.__name__ = f"PEPS_16x16_Z2_Binary_{module.__class__.__name__}"
        self.alpha_list = alpha_list
    def __call__(self,alpha_list=None,**kargs):
        if alpha_list is None:alpha_list = self.alpha_list
        model=PEPS_aggregation_model(
                                   virtual_bond_dim=self.structure_path,
                                   label_position="no_center",
                                   symmetry="Z2_16x16",
                                   patch_engine=self.module,
                                   alpha_list=alpha_list,
                                   fixed_virtual_dim=self.fixed_virtual_dim,
                                   convertPeq1=self.convertPeq1,**kargs
                                  )
        model.weight_init(method="Expecatation_Normalization2")
        return model


PEPS_16x16_Z2_Binary_CNN_6x6_0_v4 = PEPS_16x16_Z2_Binary_Wrapper(TensorNetConvND_Single,#ops=44,paras=11497
                                        "models/arbitary_shape/arbitary_shape_16x16_6x6.json",fixed_virtual_dim=4   ,alpha=1)
PEPS_16x16_Z2_Binary_CNN_8x8_0_v4 = PEPS_16x16_Z2_Binary_Wrapper(TensorNetConvND_Single,#ops=82,paras=17124
                                        "models/arbitary_shape/arbitary_shape_16x16_8x8.json",fixed_virtual_dim=4   ,alpha=1.5)
PEPS_16x16_Z2_Binary_CNN_Aggregation_6x6_28_v3    = PEPS_16x16_Z2_Binary_Aggregation_Wrapper(TensorNetConvND_Single,#ops=1232,paras=112518
                                                                "models/arbitary_shape/patch_partions_28_6x6_Z2_json_list.pt",fixed_virtual_dim=3, alpha_list = 0.7)
PEPS_16x16_Z2_Binary_CNN_Aggregation_6x6_28_v4    = PEPS_16x16_Z2_Binary_Aggregation_Wrapper(TensorNetConvND_Single,
                                                                "models/arbitary_shape/patch_partions_28_6x6_Z2_json_list.pt",fixed_virtual_dim=4, alpha_list = 1)
PEPS_16x16_Z2_Binary_CNN_Aggregation_6x6_28_v2    = PEPS_16x16_Z2_Binary_Aggregation_Wrapper(TensorNetConvND_Single,#ops=1232,paras=35864
                                                                "models/arbitary_shape/patch_partions_28_6x6_Z2_json_list.pt",fixed_virtual_dim=2, alpha_list = 0.35)

PEPS_16x16_Z2_Binary_TAT_6x6_0_v4 = PEPS_16x16_Z2_Binary_Wrapper(TensorAttention,#ops=113,paras=34736
                                        "models/arbitary_shape/arbitary_shape_16x16_6x6.json",fixed_virtual_dim=4   ,alpha=0.02)
PEPS_16x16_Z2_Binary_TAT_6x6_0_v5 = PEPS_16x16_Z2_Binary_Wrapper(TensorAttention,#ops=113,paras=82240
                                        "models/arbitary_shape/arbitary_shape_16x16_6x6.json",fixed_virtual_dim=5   ,alpha=0.015)
PEPS_16x16_Z2_Binary_TAT_8x8_0_v4 = PEPS_16x16_Z2_Binary_Wrapper(TensorAttention,#ops=204,paras=59936
                                        "models/arbitary_shape/arbitary_shape_16x16_8x8.json",fixed_virtual_dim=4   ,alpha=0.005)

PEPS_16x16_Z2_Binary_TAT_Aggregation_6x6_28_v3    = PEPS_16x16_Z2_Binary_Aggregation_Wrapper(TensorAttention,#ops=3164,paras=320082
                                                                "models/arbitary_shape/patch_partions_28_6x6_Z2_json_list.pt",fixed_virtual_dim=3, alpha_list = 0.03)
PEPS_16x16_Z2_Binary_TAT_Aggregation_6x6_28_v2    = PEPS_16x16_Z2_Binary_Aggregation_Wrapper(TensorAttention,#ops=3164,paras= 73020
                                                                "models/arbitary_shape/patch_partions_28_6x6_Z2_json_list.pt",fixed_virtual_dim=2, alpha_list =  0.05)
PEPS_16x16_Z2_Binary_TAT_Aggregation_6x6_28_v4    = PEPS_16x16_Z2_Binary_Aggregation_Wrapper(TensorAttention,#ops=3164,paras=952112
                                                                "models/arbitary_shape/patch_partions_28_6x6_Z2_json_list.pt",fixed_virtual_dim=4, alpha_list = 0.02)

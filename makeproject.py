import time
from config import*
import numpy as np


######################
#### train config ####
######################
trainbases= [Train_Base_Default.copy({'accu_list':[#'MSError',
                                                   #'MSError_for_RDN','MSError_for_PTN','MSError_for_PLG',
                                                   #"ClassifierA","CELoss",#"ClassifierP","ClassifierN",
                                                   'BinaryAL','BCEWithLogitsLoss',#'BinaryNL'
                                                    ],
                                     "grad_clip":None,
                                     'warm_up_epoch':100,
                                     'epoches': 1000,
                                     'use_swa':False,
                                     'swa_start':20,
                                     'BATCH_SIZE':10800,
                                     'drop_rate':None,
                                     'do_extra_phase':False,
                                     'doearlystop':True,
                                     'doanormaldt':True,
                                     #'use_metatrain':True,
                                     'optuna_limit_trials':12})]

hypertuner= [Optuna_Train_Default.copy({'hypertuner_config':{'n_trials':3},'not_prune':True,
                                       'optimizer_list':{
                                                'Adam':{'lr':[0.0005,0.005],  'betas':[[0.5,0.9],0.999]},
                                                #'Adabelief':{'lr':[0.0005,0.005],'eps':[1e-11,1e-7],'weight_decouple': True,'rectify':True,'print_change_log':False}
                                                },
                                       #'drop_rate_range':[0,0.5],
                                       #'grad_clip_list':[None,1,5],
                                       #'alpha_list':[3,3.5,4],
                                       #'convertPeq1_list':[0,
                                        #                    1,
                                                            #"all_convert"
                                        #                    ],
                                       'batch_size_list':[200,10800],
                                                        })]
#hypertuner= [Normal_Train_Default]
schedulers= [Scheduler_None]
#schedulers= [Scheduler_CosALR_Default.copy({"config":{"T_max":32}})]
optimizers= [Optimizer_Adam.copy({"config":{"lr":0.001}})]
#optimizers= [Optimizer_lbfgs.copy({"config":{"lr":0.01,'max_iter':20}})]
earlystops= [Earlystop_NMM_Default.copy({"_TYPE_":"no_min_more","config":{"es_max_window":40}})]
anormal_detect= [Anormal_D_DC_Default.copy({"_TYPE_":"decrease_counting",
                                      "config":{"stop_counting":30,
                                                    "wall_value":0.8,
                                                    "delta_unit" :1e-8,
                                                    }})]
train_config_list = [ConfigCombine({"base":[b,h], "scheduler":[s], "earlystop":[e],"optimizer":[o],"anormal_detect":[a]})
                        for b in trainbases for h in hypertuner for s in schedulers
                        for o in optimizers for e in earlystops
                        for a in anormal_detect]

MNIST_DATA_Config=Config({'dataset_TYPE':'datasets.FashionMNIST','dataset_args':{'root':DATAROOT+f"/FashionMNIST",'download':True}})
msdataT_RDNfft   =msdataT_RDN.copy({'image_transformer':'fft16x9_norm'})
msdataT_RDNM55   =msdataT_RDN.copy({'image_transformer':'-0.5-0.5'})
dmlist=[
           # (MNIST_DATA_Config,
           #  backbone_templete.copy({'backbone_TYPE':'LinearCombineModel2',
           #                           'backbone_config':{'virtual_bond_dim':6,'init_std':1},
           #                           'train_batches':1000
           #                           })
           # ),
        # [msdataT_RDNfft,  backbone_templete.copy({'criterion_type':"CELoss",#'criterion_config':{'reduction':'sum'},
        #     'backbone_TYPE':'PEPS_16x9_Z2_Binary_TAT_13_v4','backbone_config':{"out_features":2},
        #     'backbone_alias':'PEPS_16x9_Z2_Binary_TAT_13_v4_CE',
        # })],
        # [msdataT_RDNfft,  backbone_templete.copy({'criterion_type':"CELoss",#'criterion_config':{'reduction':'sum'},
        #     'backbone_TYPE':'PEPS_16x9_Z2_Binary_TAT_3_v4','backbone_config':{"out_features":2},
        #     'backbone_alias':'PEPS_16x9_Z2_Binary_TAT_3_v4_CE','valid_batch':3000
        # })],
        # [msdataT_RDNfft,  backbone_templete.copy({'criterion_type':"CELoss",#'criterion_config':{'reduction':'sum'},
        #     'backbone_TYPE':'PEPS_16x9_Z2_Binary_TAT_2_v4','backbone_config':{"out_features":2},
        #     'backbone_alias':'PEPS_16x9_Z2_Binary_TAT_2_v4_CE','valid_batch':3000
        # })],
        #
        #
        # [msdataT_RDNfft,  backbone_templete.copy({'criterion_type':"CELoss",#'criterion_config':{'reduction':'sum'},
        #     'backbone_TYPE':'PEPS_16x9_Z2_Binary_CNNS_2_v4','backbone_config':{"out_features":2},
        #     'backbone_alias':'PEPS_16x9_Z2_Binary_CNNS_2_v4_CE','valid_batch':3000
        # })],
        # [msdataT_RDNfft,  backbone_templete.copy({'criterion_type':"CELoss",#'criterion_config':{'reduction':'sum'},
        #     'backbone_TYPE':'PEPS_16x9_Z2_Binary_CNNS_13_v4','backbone_config':{"out_features":2},
        #     'backbone_alias':'PEPS_16x9_Z2_Binary_CNNS_13_v4_CE','valid_batch':3000
        # })],

            # [msdataT_RDNfft,  backbone_templete.copy({'criterion_type':"BCEWithLogitsLoss",#'criterion_config':{'reduction':'sum'},
            # 'backbone_TYPE':'PEPS_16x9_Z2_Binary_CNN_Aggregation_12_3_v3','backbone_config':{"out_features":1},
            # 'backbone_alias':'PEPS_16x9_Z2_Binary_CNN_Aggregation_12_3_v3','valid_batch':3000
            # })],

            [msdataT_RDNfft,  backbone_templete.copy({'criterion_type':"BCEWithLogitsLoss",#'criterion_config':{'reduction':'sum'},
            'backbone_TYPE':'PEPS_16x9_Z2_Binary_TAT_Aggregation_12_3_v2','backbone_config':{"out_features":1},
            'backbone_alias':'PEPS_16x9_Z2_Binary_TAT_Aggregation_12_3_v2','valid_batch':3000
            })],

            # [msdataT_RDNfft,  backbone_templete.copy({'criterion_type':"BCEWithLogitsLoss",#'criterion_config':{'reduction':'sum'},
            # 'backbone_TYPE':'PEPS_16x9_Z2_Binary_TAT_2_v4','backbone_config':{"out_features":1},
            # 'backbone_alias':'PEPS_16x9_Z2_Binary_TAT_2_v4','valid_batch':3000
            # })],
            #
            # [msdataT_RDNM55,  backbone_templete.copy({'criterion_type':"BCEWithLogitsLoss",#'criterion_config':{'reduction':'sum'},
            # 'backbone_TYPE':'PEPS_16x16_Z2_Binary_CNN_Aggregation_6x6_28_v3','backbone_config':{"out_features":1},
            # 'backbone_alias':'PEPS_16x16_Z2_Binary_CNN_Aggregation_6x6_28_v3','valid_batch':3000
            # })],
            # [msdataT_RDNM55,  backbone_templete.copy({'criterion_type':"BCEWithLogitsLoss",#'criterion_config':{'reduction':'sum'},
            #  'backbone_TYPE':'PEPS_16x16_Z2_Binary_TAT_8x8_0_v4','backbone_config':{"out_features":1},
            #  'backbone_alias':'PEPS_16x16_Z2_Binary_TAT_8x8_0_v4','valid_batch':3000
            # })],
            # [msdataT_RDNM55,  backbone_templete.copy({'criterion_type':"BCEWithLogitsLoss",#'criterion_config':{'reduction':'sum'},
            #   'backbone_TYPE':'PEPS_16x16_Z2_Binary_CNN_Aggregation_6x6_28_v3','backbone_config':{"out_features":1},
            #   'backbone_alias':'PEPS_16x16_Z2_Binary_CNN_Aggregation_6x6_28_v3','valid_batch':3000
            # })],
            #
            # [msdataT_RDNM55,  backbone_templete.copy({'criterion_type':"BCEWithLogitsLoss",#'criterion_config':{'reduction':'sum'},
            #   'backbone_TYPE':'PEPS_16x16_Z2_Binary_TAT_6x6_0_v4','backbone_config':{"out_features":1},
            #   'backbone_alias':'PEPS_16x16_Z2_Binary_TAT_6x6_0_v4','valid_batch':3000
            # })],
            # [msdataT_RDNM55,  backbone_templete.copy({'criterion_type':"BCEWithLogitsLoss",#'criterion_config':{'reduction':'sum'},
            #   'backbone_TYPE':'PEPS_16x16_Z2_Binary_TAT_8x8_0_v4','backbone_config':{"out_features":1},
            #   'backbone_alias':'PEPS_16x16_Z2_Binary_TAT_8x8_0_v4','valid_batch':3000
            # })],


          # [msdataT_RDNM55,  backbone_templete.copy({'criterion_type':"BCEWithLogitsLoss",#'criterion_config':{'reduction':'sum'},
          #     'backbone_TYPE':'PEPS_16x16_Z2_Binary_TAT_Aggregation_6x6_28_v3','backbone_config':{"out_features":1},
          #     'backbone_alias':'PEPS_16x16_Z2_Binary_TAT_Aggregation_6x6_28_v3',
          # })],


       # [msdataT_RDNfft,  backbone_templete.copy({'criterion_type':"BCEWithLogitsLoss",#'criterion_config':{'reduction':'sum'},
       #     'backbone_TYPE':'PEPS_16x9_Z2_Binary_CNN_7','backbone_config':{"alpha":4,"out_features":1,"convertPeq1":True},
       #     'backbone_alias':'PEPS_16x9_Z2_Binary_CNN_7',
       # })],
      # [msdataT_RDNfft,  backbone_templete.copy({'criterion_type':"BCEWithLogitsLoss",#'criterion_config':{'reduction':'sum'},
      #     'backbone_TYPE':'PEPS_16x9_Z2_Binary_TA_0','backbone_config':{"alpha":4,"out_features":1,"convertPeq1":True},
      #     'backbone_alias':'PEPS_16x9_Z2_Binary_TA_0',
      # })],
      # [msdataT_RDNfft,  backbone_templete.copy({'criterion_type':"BCEWithLogitsLoss",#'criterion_config':{'reduction':'sum'},
      #     'backbone_TYPE':'PEPS_16x9_Z2_Binary_CNN_Aggregation_19_3','backbone_config':{"out_features":1},
      #     'backbone_alias':'PEPS_16x9_Z2_Binary_CNN_Aggregation_19_3',
      # })],
           # (MNIST_DATA_Config.copy({'crop':24,'reverse':True,'divide':4}),
           #  backbone_templete.copy({'backbone_TYPE':'PEPS_uniform_shape_symmetry_any',
           #                           'backbone_config':{'W':6,'H':6,'virtual_bond_dim':6,'in_physics_bond':16,'init_std': 1e-5},
           #                           'train_batches':1400
           #                           })
           #  ),
            # (MNIST_DATA_Config.copy({'p_norm':DATAROOT+f"/MNIST/MNIST/statisitc_stdmean.pt",'crop':24}),
            #  backbone_templete.copy({'backbone_TYPE':'TensorNetworkDeepModel1',
            #                           'backbone_config':{'virtual_bond_dim':5,'init_std':1e-5,'normlized':True, 'set_var':1},
            #                           'train_batches':2000
            #                           })
            #  ),
            #  (MNIST_DATA_Config.copy({'reverse':True}),
            #   backbone_templete.copy({'backbone_TYPE':'PEPS_einsum_arbitrary_partition_optim',
            #                        'backbone_alias':'PEPS_einsum_arbitrary_partition_optim_shape_3',
            #                        'backbone_config':{'virtual_bond_config':"models/arbitary_shape/arbitary_shape_3.json",'solved_std':0.024},
            #                        'train_batches':600
            #                        })
            # ),
        ]
# dmlist=[(MNIST_DATA_Config,
#              backbone_templete.copy({'backbone_TYPE':'LinearCombineModel2',
#                                       'backbone_config':{'virtual_bond_dim':vd,'init_std':3e-1},
#                                       'train_batches':Bs
#                                       })
#              ) for vd,Bs in [
#              #[3,20000],[4,10000],[6,2000],[7,900],[8,300],[9,200],
#              #[10,100]]
#
#         ]




#### generate config
for train_cfg in train_config_list:
    for data_cfg,model_cfg in dmlist:
            cfg = Merge(data_cfg, train_cfg, model_cfg)
            TIME_NOW        = time.strftime("%m_%d_%H_%M_%S")
            cfg.create_time = TIME_NOW
            del data_cfg
            cfg.save()

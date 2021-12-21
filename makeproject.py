
import time
from config import*
import numpy as np


######################
#### train config ####
######################
trainbases= [Train_Base_Default.copy({"grad_clip":None,
                                     'warm_up_epoch':100,
                                     'epoches': 400,
                                     'use_swa':False,
                                     'swa_start':20,
                                     #'BATCH_SIZE':4000,
                                     'drop_rate':None,
                                     'do_extra_phase':False,
                                     'doearlystop':False})]
hypertuner= [Normal_Train_Default]
hypertuner= [Optuna_Train_Default.copy({'hypertuner_config':{'n_trials':3},'not_prune':True,
                                       'optimizer_list':{
                                                'Adam':{'lr':[0.001,0.01],  'betas':[[0.5,0.9],0.999]},
                                                #'Adabelief':{'lr':[0.0005,0.005],'eps':[1e-11,1e-7],'weight_decouple': True,'rectify':True,'print_change_log':False}
                                                },
                                       #'drop_rate_range':[0.1,0.25],
                                       'grad_clip_list':[None,5],
                                                        })]

schedulers= [Scheduler_None]
schedulers= [Scheduler_CosALR_Default.copy({"config":{"T_max":32}})]
optimizers= [Optimizer_Adam.copy({"config":{"lr":0.01}})]
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
# dmlist=[
#            # (MNIST_DATA_Config,
#            #  backbone_templete.copy({'backbone_TYPE':'LinearCombineModel2',
#                                      # 'backbone_config':{'virtual_bond_dim':8,'init_std':1},
#                                      # 'train_batches':1000
#                                      # })
#            # ),
#            # (MNIST_DATA_Config,
#            #  backbone_templete.copy({'backbone_TYPE':'LinearCombineModel3',
#            #                           'backbone_config':{'virtual_bond_dim':5,'init_std':1e-5},
#            #                           'train_batches':3500
#            #                           })
#            #  ),
#             (MNIST_DATA_Config.copy({'crop':24}),
#              backbone_templete.copy({'backbone_TYPE':'TensorNetworkDeepModel1',
#                                       'backbone_config':{'virtual_bond_dim':5,'init_std':1e-5},
#                                       'train_batches':4000
#                                       })
#              ),
#              (MNIST_DATA_Config.copy({'reverse':True}),
#               backbone_templete.copy({'backbone_TYPE':'PEPS_einsum_arbitrary_partition_optim',
#                                    'backbone_config':{'virtual_bond_dim':"models/arbitary_shape/arbitary_shape_2.json",'init_std':1e-2},
#                                    'train_batches':4000
#                                    })
#             ),
#         ]


dmlist=[(MNIST_DATA_Config.copy({'crop':24,'divide':4}),
         backbone_templete.copy({'backbone_TYPE':'PEPS_einsum_uniform_shape_6x6_fast',
                                      'backbone_config':{'virtual_bond_dim':vd,'init_std':4e-2,'in_physics_bond':16},
                                      'train_batches':Bs
                                      })
             ) for vd,Bs in [
             #[3,20000],[4,10000],[5,3000],
             [6,1400]]

        ]

#### generate config
for train_cfg in train_config_list:
    for data_cfg,model_cfg in dmlist:
            cfg = Merge(data_cfg, train_cfg, model_cfg)
            TIME_NOW        = time.strftime("%m_%d_%H_%M_%S")
            cfg.create_time = TIME_NOW
            del data_cfg
            cfg.save()

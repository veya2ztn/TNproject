import time
from config import*
import numpy as np


######################
#### train config ####
######################
trainbases= [Train_Base_Default.copy({"grad_clip":None,
                                     'warm_up_epoch':20,
                                     'epoches': 300,
                                     'use_swa':False,
                                     'swa_start':20,
                                     #'BATCH_SIZE':4000,
                                     'drop_rate':None,
                                     'do_extra_phase':False,
                                     'doearlystop':True,
                                     'doanormaldt':True})]
hypertuner= [Normal_Train_Default]
# hypertuner= [Optuna_Train_Default.copy({'hypertuner_config':{'n_trials':20},'not_prune':True,
#                                        'optimizer_list':{
#                                                 'Adam':{'lr':[0.0001,0.005],  'betas':[[0.5,0.9],0.999]},
#                                                 #'Adabelief':{'lr':[0.0005,0.005],'eps':[1e-11,1e-7],'weight_decouple': True,'rectify':True,'print_change_log':False}
#                                                 },
#                                        #'drop_rate_range':[0.1,0.25],
#                                        'grad_clip_list':[None],
#                                                         })]

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

MNIST_DATA_Config=Config({'dataset_TYPE':'datasets.MNIST','dataset_args':{'root':DATAROOT+f"/MNIST",'download':True}})
dmlist=[
           # (MNIST_DATA_Config,
           #  backbone_templete.copy({'backbone_TYPE':'LinearCombineModel2',
           #                           'backbone_config':{'virtual_bond_dim':10,'init_std':0.3},
           #                           'train_batches':100
           #                           })
           # ),
           # (MNIST_DATA_Config.copy({'crop':24,'reverse':True,'divide':4}),
           #  backbone_templete.copy({'backbone_TYPE':'PEPS_uniform_shape_symmetry_any',
           #                           'backbone_config':{'W':6,'H':6,'virtual_bond_dim':6,'in_physics_bond':16,'init_std': 1e-5},
           #                           'train_batches':1400
           #                           })
           #  ),
            (MNIST_DATA_Config.copy({'p_norm':DATAROOT+f"/MNIST/MNIST/statisitc_stdmean.pt",'crop':24}),
             backbone_templete.copy({'backbone_TYPE':'TensorNetworkDeepModel1',
                                      'backbone_config':{'virtual_bond_dim':5,'init_std':1e-5,'normlized':True, 'set_var':1},
                                      'train_batches':2000
                                      })
             ),
            #  (MNIST_DATA_Config.copy({'reverse':True}),
            #   backbone_templete.copy({'backbone_TYPE':'PEPS_einsum_arbitrary_partition_optim',
            #                        'backbone_config':{'virtual_bond_dim':"models/arbitary_shape/arbitary_shape_2.json",'init_std':1e-2},
            #                        'train_batches':4000
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

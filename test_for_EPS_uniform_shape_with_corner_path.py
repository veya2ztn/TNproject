import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from utils import preprocess_sincos, preprocess_binary,aggregation_patch
from models.two_dim_model import *
import random,time,os
from optuna.trial import TrialState
def preprocess_images(x):
    return (1-(aggregation_patch(x[...,3:27,3:27],divide=4)))/50
import numpy as  np
from mltool.dataaccelerate import DataSimfetcher
from mltool.loggingsystem import LoggingSystem
import json
DATAROOTPATH = f"{os.path.dirname(os.path.realpath(__file__))}/.DATARoot.json"
print(DATAROOTPATH)
if os.path.exists(DATAROOTPATH):
    with open(DATAROOTPATH,'r') as f:
        RootDict=json.load(f)

DATAROOT  = RootDict['DATAROOT']
SAVEROOT  = RootDict['SAVEROOT']
EXP_HUB   = RootDict['EXP_HUB']
MODEL_NAME= "PEPS_einsum_uniform_shape_6x6_fast"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-bs","--batch_size"  , default=64, type=int,help="batch_size")
parser.add_argument("-vd","--virtual_bond", default=7 , type=int,help="virtual_bond dimension")
parser.add_argument("-gpu","--gpu"        , default=0 , type=int,help="the gpu number")
parser.add_argument("-W","--width"         , default=6 , type=int,help="the gpu number")
parser.add_argument("-H","--height"        , default=6 , type=int,help="the gpu number")
args = parser.parse_args()


batch_size   = args.batch_size
virtual_bond = args.virtual_bond
gpu          = args.gpu
job_gpu     = str(args.gpu)
W = args.width
H = args.height
os.environ["CUDA_VISIBLE_DEVICES"] = job_gpu
# lr           = 0.1
# init_std     = 1

device       = 'cuda'
DB_NAME      = MODEL_NAME
TASK_NAME    = MODEL_NAME+f".vd={virtual_bond}.W={W}.H={H}"
transform = transforms.Compose([transforms.ToTensor()])

mnist_train = datasets.MNIST(DATAROOT, train=True, download=False, transform=transform)
mnist_test  = datasets.MNIST(DATAROOT, train=False,download=False, transform=transform)
train_loader= torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=mnist_test,  batch_size=batch_size, shuffle=False)

def do_train(model,config_pool,logsys,trial=None,**kargs):

    hparam_dict   = config_pool['hparam']
    epoches       = config_pool['epoches']
    accu_list     = ['error']
    doearlystop   = False
    valid_per_epoch=1
    start_epoch   = 0
    lr            = hparam_dict['lr']
    logsys.regist({'task':config_pool['project_name'],
                  'model':config_pool['model_name']})

    metric_dict   = logsys.initial_metric_dict(accu_list)
    metric_dict   = metric_dict.metric_dict
    _             = logsys.create_recorder(hparam_dict=hparam_dict,metric_dict=metric_dict)

    save_accu_list    = accu_list[0:1]
    _                 = logsys.create_model_saver(accu_list=save_accu_list,epoches=epoches)
    FULLNAME          = config_pool['project_name']
    banner            = logsys.banner_initial(epoches,FULLNAME)
    master_bar        = logsys.create_master_bar(epoches)
    logsys.banner_show(start_epoch,FULLNAME)
    logsys.train_bar  = logsys.create_progress_bar(1,unit=' img',unit_scale=train_loader.batch_size)
    logsys.valid_bar  = logsys.create_progress_bar(1,unit=' img',unit_scale=valid_loader.batch_size)
    logsys.Q_batch_loss_record=True
    device     = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in master_bar:
        #if epoch < start_epoch:continue
        ### training phase ########
        model.train()
        logsys.train()
        batches    = len(train_loader)
        infiniter  = DataSimfetcher(train_loader, device=device)
        inter_b    = logsys.create_progress_bar(len(train_loader))
        train_loss = []
        while inter_b.update_step():
            image,label= infiniter.next()
            optimizer.zero_grad()
            data       = preprocess_images(image)
            logits     = model(data).squeeze(-1)
            loss       = torch.nn.CrossEntropyLoss()(logits,label)
            loss.backward()
            #for p in model.parameters():nn.utils.clip_grad_norm_(p, max_norm=1)
            optimizer.step()
            logsys.batch_loss_record([loss])
            loss=loss.cpu().item()
            train_loss.append(loss)
            outstring="Batch:{:3}/{} loss:{:.4f} ".format(inter_b.now,batches,loss)
            inter_b.lwrite(outstring, end="\r")
        train_loss=np.array(train_loss).mean()
        if np.isnan(loss):raise NanValueError
        logsys.record('the_lr_use_now', optimizer.param_groups[0]['lr'] , epoch)
        logsys.record('training_loss', loss, epoch)

        bad_condition_happen = logsys.save_latest_ckpt( {"model": model,
                                epoch,train_loss,saveQ=True,
                                optimizer=optimizer,doearlystop=False)
        ### valid phase ########
        if epoch%valid_per_epoch ==0:
            model.eval()
            logsys.eval()
            prefetcher = DataSimfetcher(valid_loader, device=device)
            inter_b    = logsys.create_progress_bar(len(valid_loader))
            labels     = []
            logits     = []
            with torch.no_grad():
                while inter_b.update_step():
                    image,label= prefetcher.next()
                    data       = preprocess_images(image)
                    logit      = model(data).squeeze(-1)
                    labels.append(label)
                    logits.append(logit)
            labels  = torch.cat(labels)
            logits  = torch.cat(logits)
            pred_labels  = torch.argmax(logits,-1)
            accu =  torch.sum(pred_labels == labels)/len(labels)
            valid_acc_pool = {'error':1-accu.item()}
            update_accu    = logsys.metric_dict.update(valid_acc_pool,epoch)
            metric_dict    = logsys.metric_dict.metric_dict
            for accu_type in accu_list:
                logsys.record(accu_type, valid_acc_pool[accu_type], epoch)
                logsys.record('best_'+accu_type, metric_dict['best_'+accu_type][accu_type], epoch)
            earlystopQ  = logsys.save_best_ckpt(model,metric_dict,epoch,doearlystop=doearlystop)

        if trial:
            trial.report(metric_dict[accu_list[0]], epoch)
            # if trial.should_prune():
            #     raise optuna.TrialPruned()



import optuna
def objective(trial):
    #drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', 0.0, 1.0, 0.1)
    random_seed = random.randint(1, 100000)
    TIME_NOW    = time.strftime("%m_%d_%H_%M_%S")
    TRIAL_NOW   = '{}-seed-{}'.format(TIME_NOW,random_seed)
    save_checkpoint   = os.path.join(SAVEROOT,'checkpoints',MODEL_NAME,TRIAL_NOW)
    logsys            = LoggingSystem(True,save_checkpoint,
                                      bar_log_path=f"runtime_log/bar_for_job_on_GPU{gpu}",
                                      seed=random_seed)
    if not os.path.exists(save_checkpoint):os.makedirs(save_checkpoint)



    lr      = trial.suggest_uniform(f"learning_rate", 0.001,0.01)
    init_std= trial.suggest_loguniform(f"init_std", 1e-5,1)
    trial.set_user_attr('trial_name', TRIAL_NOW)


    model = eval(MODEL_NAME)(W,H,out_features=10,in_physics_bond=16,virtual_bond_dim=virtual_bond,init_std=init_std)
    #print([p.shape for p in model.parameters()])
    device = 'cuda'
    model = model.to(device)
    config_pool= {"project_name":"Uniform_shape_finite_PEPS",
                    'model_name':MODEL_NAME,
                    "epoches":100,
                    'hparam':{"lr":lr,"init_std":init_std,"vd":virtual_bond,"batch_size":batch_size}
    }
    result = do_train(model,config_pool,logsys,trial=trial)
    del model
    torch.cuda.empty_cache()
    return result
#study = optuna.create_study(direction="minimize")

study = optuna.create_study(study_name=TASK_NAME, storage=f'sqlite:///optuna_database/{DB_NAME}.db',
                            load_if_exists=True,
                            sampler=optuna.samplers.CmaEsSampler(),
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=28)
                            )
optuna_limit_trials = 30
if len([t.state for t in study.trials if t.state== TrialState.COMPLETE])>optuna_limit_trials:raise
#study.optimize(objective, n_trials=50, timeout=600,pruner=optuna.pruners.MedianPruner())
hypertuner_config =  {'n_trials':10}
study.optimize(objective, **hypertuner_config)
del model
torch.cuda.empty_cache()
project_root_dir = None

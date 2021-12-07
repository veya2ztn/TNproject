import os,json,traceback,time,re,shutil,random,sys
import numpy as np
import torch
import torch.nn as nn

import optuna
from optuna.trial import TrialState

from mltool.dataaccelerate import DataLoaderX,DataLoader,DataPrefetcher,DataSimfetcher
from mltool.tableprint.printer import summary_table_info
from mltool.loggingsystem import LoggingSystem
from mltool import tableprint as tp
from mltool import lr_schedulers as lr_scheduler
from mltool import lr_schedulers as lrschdl
from mltool import optim as optim

from torchvision import datasets, transforms

from models import NanValueError
import models as mdl

MEMORYLIMIT = 0.9
GPU_MEMORY_CONFIG_FILE="projects/GPU_Memory_Config.json"

from config import SAVEROOT,EXP_HUB
class TestCodeError(NotImplementedError):pass
class Project:pass

def custom_2d_patch(x,divide=4):
    B, dim0, dim1 = tuple(x.shape)
    assert dim0%divide==0
    assert dim1%divide==0
    x     = x.reshape((B, dim0//divide,divide,dim1//divide,divide)).permute(0,1,3,2,4)
    x     = x.flatten(start_dim=-2,end_dim=-1)
    return 1-x

#torch.autograd.set_detect_anomaly(True)
def struct_model(project_config,dataset_train,cuda=True):
    MODEL_TYPE       = project_config.model.backbone_TYPE
    MODEL_CONIFG     = project_config.model.backbone_config

    W,H,P            = dataset_train[0][0].shape[-3:]
    model            = eval(f"mdl.{MODEL_TYPE}")(W=W,H=H,in_physics_bond=P,**MODEL_CONIFG)
    model            = model.cuda() if cuda else model
    if hasattr(project_config.model,'pre_train_weight'):
        pre_train_weight  = project_config.model.pre_train_weight
        if pre_train_weight is not None:
            print("----------------------------")
            print(f"load pre weight at {pre_train_weight}")
            print("----------------------------")
            weight = torch.load(pre_train_weight)
            if 'state_dict' in weight:weight= weight['state_dict']
            if pre_train_weight is not None:model.load_state_dict(weight)
    return model

def struct_dataloader(project_config,only_valid=False,verbose = True):
    DATASET_TYPE= project_config.data.dataset_TYPE
    DATASETargs = project_config.data.dataset_args
    transform   = [transforms.ToTensor()]
    if hasattr(project_config.data,'crop') and project_config.data.crop is not None:
        transform.append(transforms.CenterCrop(project_config.data.crop))
    if hasattr(project_config.data,'divide') and project_config.data.divide is not None:
        transform.append(lambda x:custom_2d_patch(x,project_config.data.divide))
    transform   = transforms.Compose(transform)
    if only_valid:
        assert DATANORMER == "none"
        dataset_valid= eval(f"{DATASET_TYPE}")(transform=transform,**DATASETargs)
        return dataset_valid
    dataset_train= eval(f"{DATASET_TYPE}")(train=True,transform=transform,**DATASETargs)
    dataset_valid= eval(f"{DATASET_TYPE}")(train=False,transform=transform,**DATASETargs)

    db = Project()
    db.dataset_train= dataset_train
    db.dataset_valid= dataset_valid
    db.data_summary = project_config.data.to_dict()
    return db
def struct_config(project_config,db = None,build_model=True,verbose=True):
    PROJECTNAME    = project_config.project_name #
    PROJECTFULLNAME= project_config.project_json_name #
    EPOCHES        = project_config.train.epoches   #100
    TRIALS         = project_config.train.trials if hasattr(project_config.train,'trials') else 1
    BATCH_SIZE     = project_config.train.BATCH_SIZE if hasattr(project_config.train,'BATCH_SIZE') \
                                                 else project_config.model.train_batches
    [memory_k,memory_b] = project_config.model.memory_para if hasattr(project_config.model,'memory_para') else [None,None]
    # judge whethere reload dataset

    if db is None or project_config.data.to_dict() != db.data_summary:
        if verbose:print("======== use new dataset! ========")
        db = struct_dataloader(project_config,verbose=verbose)
    else:
        if verbose:print("======== inherit dataset! ========")
    DATA_VOLUME = project_config.train.volume if hasattr(project_config.train,'volume') else None
    ORIGIN_SIZE = len(db.dataset_train)
    if not isinstance(BATCH_SIZE,int):
        if verbose:print("the BATCH_SIZE is not a int")
        raise

    BATCH_SIZE = min(len(db.dataset_train),BATCH_SIZE)
    if BATCH_SIZE<0:BATCH_SIZE=100
    if verbose:print("==== the batch size now set {}".format(BATCH_SIZE))
    project_config.train.BATCH_SIZE = BATCH_SIZE

    #train_loader = DataLoaderX(dataset=db.dataset_train,num_workers=1,batch_size=BATCH_SIZE,pin_memory=True,shuffle=True,collate_fn=db.dataset_train._collate)
    train_loader = DataLoader(dataset=db.dataset_train,batch_size=BATCH_SIZE,pin_memory=True,shuffle=True)
    valid_loader = DataLoader(dataset=db.dataset_valid,batch_size=BATCH_SIZE)
    project = Project()

    project.trials_num   = TRIALS
    project.train_epoches= EPOCHES
    project.project_name = PROJECTNAME+".on{}".format(len(db.dataset_train)) if DATA_VOLUME is None else PROJECTNAME
    #project.project_name = PROJECTNAME
    project.train_loader = train_loader
    project.valid_loader = valid_loader

    project.full_config  = project_config
    model = struct_model(project_config,db.dataset_train) if build_model else None
    return model,project,db

def train_epoch(model,dataloader,logsys,Fethcher=DataSimfetcher,test_mode=False,detail_log=False):
    '''
    train model
    '''
    model.train()
    logsys.train()
    batches    = len(dataloader)
    device     = next(model.parameters()).device
    prefetcher = Fethcher(dataloader,device)
    criterion  = model.criterion
    inter_b    = logsys.create_progress_bar(batches)
    train_loss = []
    train_accu = []
    while inter_b.update_step():
        image,label= prefetcher.next()
        model.optimizer.zero_grad()
        logit  = model(image.squeeze())
        logit  = logit.squeeze()
        loss   = torch.nn.CrossEntropyLoss()(logit,label)
        loss.backward()
        if hasattr(model.optimizer,"grad_clip") and (model.optimizer.grad_clip is not None):
            nn.utils.clip_grad_norm_(model.parameters(), model.optimizer.grad_clip)
        model.optimizer.step()
        if model.scheduler is not None:model.scheduler.step()

        loss = loss.cpu().item()
        accu = (torch.sum(torch.argmax(logit,-1) == label)/len(label)).cpu().item()
        logsys.batch_loss_record([loss,accu])
        train_loss.append(loss)
        train_accu.append(accu)
        outstring="Batch:{:3}/{} loss:{:.4f} accu:{:.3f}".format(inter_b.now,batches,loss,accu)
        inter_b.lwrite(outstring, end="\r")
        if test_mode:return
    if not detail_log:
        train_loss=np.array(train_loss).mean()
        train_accu=np.array(train_accu).mean()
    return train_loss,train_accu
def test_epoch(model,dataloader,logsys,accu_list=None,Fethcher=DataSimfetcher,inference=False):
    model.eval()
    logsys.eval()
    device     = next(model.parameters()).device
    prefetcher = Fethcher(dataloader,device)
    batches    = len(dataloader)
    inter_b    = logsys.create_progress_bar(batches)
    labels = []
    logits = []
    with torch.no_grad():
        while inter_b.update_step():
            image,label= prefetcher.next()
            logit  = model(image.squeeze())
            logit  = logit.squeeze()
            labels.append(label)
            logits.append(logit)
    labels  = torch.cat(labels)
    logits  = torch.cat(logits)
    pred_labels  = torch.argmax(logits,-1)
    accu =  torch.sum(pred_labels == labels)/len(labels)
    valid_acc_pool = {'error':1-accu.item()}
    return valid_acc_pool


def get_hparam_dict(config):
    #due to the bad performance of tensorboard Hyper Parameter, we need list all HYPER at beginning
    hparam_dict     = {'model':config.model.str_backbone_TYPE,
                       'criterion':config.model.criterion_type,
                       'optimer':config.train.optimizer.str_optimizer_TYPE if hasattr(config.train.optimizer,"str_optimizer_TYPE") \
                                 else config.train.optimizer._TYPE_,
                       'lr': config.train.optimizer.config['lr'],
                       'normf':config.data.dataset_norm,
                     }
    if hasattr(config.train,'volume'):hparam_dict['volume']=config.train.volume
    if hasattr(config.train,'drop_rate'):hparam_dict['drop_rate']=config.train.drop_rate
    if hasattr(config.model,'criterion_config'):
        for key, val in config.model.criterion_config.items():
            hparam_dict[f'criterion_config_{key}']=val
    if hasattr(config,"optuna_hparam"):
        for key,val in config.optuna_hparam.items():
            hparam_dict[key]=val
    return hparam_dict
def enter_into_SGD_phase(model,args,logsys,last_ckpt=None):
    ## save the old ckpt
    logsys.info("start new phase")
    if last_ckpt is not None:
        state_dict = torch.load(last_ckpt)
        state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
        model.load_state_dict(state_dict)
    optimizer_config    = args.train.optimizer.config
    optimizer           = torch.optim.SGD(model.parameters(), lr=optimizer_config['lr'],momentum=0.9)
    scheduler           = lr_scheduler.CosinePowerAnnealing(optimizer, **{'max_epoch' : 20 ,'cycles': 1,
                                                            'power' : 1,'min_lr': 0.0001,'cycle_decay':0.8})
    model.optimizer     = optimizer
    model.scheduler     = scheduler
def swa_update_bn(loader, model, device=None):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[1]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

def do_train(model,project,train_loader,valid_loader,logsys,trial=False):
    return one_complete_train(model,project,train_loader,valid_loader,logsys,trial=trial)
def one_complete_train(model,project,train_loader,valid_loader,logsys,trial=False):

    logsys.regist({'task':project.project_name,'model':project.full_config.model.backbone_TYPE})
    logsys.Q_batch_loss_record=True
    train_mode      = project.train_mode
    args            = project.full_config
    if hasattr(args,'comment') and args.comment != "":
        with open(os.path.join(logsys.ckpt_root,'README'),'w') as f:
            f.write(args.comment)
    show_start_status   = args.train.show_start_status
    warm_up_epoch       = args.train.warm_up_epoch
    valid_per_epoch     = args.train.valid_per_epoch
    infer_epoch         = args.train.infer_epoch
    do_inference        = args.train.do_inference
    epoches             = args.train.epoches
    doearlystop         = args.train.doearlystop
    doanormaldt         = args.train.doanormaldt
    do_extra_phase      = args.train.do_extra_phase
    drop_rate           = args.train.drop_rate if hasattr(args.train,'drop_rate') else None

    optimizer_config    = args.train.optimizer.config
    optimizer_TYPE      = args.train.optimizer._TYPE_
    optimizer           = eval(f"optim.{optimizer_TYPE}")(model.parameters(), **optimizer_config)
    optimizer.grad_clip = args.train.grad_clip

    criterion           = torch.nn.CrossEntropyLoss()

    scheduler_config    = args.train.scheduler.config
    scheduler_TYPE      = args.train.scheduler._TYPE_
    scheduler           = eval(f"lrschdl.{scheduler_TYPE}")(optimizer, **scheduler_config) if args.train.scheduler._TYPE_ else None
    if hasattr(args.train,'use_swa') and args.train.use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        print("active experiment feature: SWA")
        swa_model           = None
        swa_scheduler       = None



    hparam_dict         = get_hparam_dict(args)
    logsys.info(hparam_dict)
    accu_list     = ['error']
    metric_dict   = logsys.initial_metric_dict(accu_list)
    metric_dict   = metric_dict.metric_dict
    _             = logsys.create_recorder(hparam_dict=hparam_dict,metric_dict=metric_dict)
    epoches       = project.train_epoches
    save_accu_list= accu_list[0:1] if train_mode == "optuna" else accu_list
    _             = logsys.create_model_saver(accu_list=save_accu_list,epoches=epoches,
                                              earlystop_config=args.train.earlystop.config,
                                              anormal_d_config=args.train.anormal_detect['config'],
                                              )
    start_epoch = 0
    if train_mode == "contine_train":
        weight_path,start_epoch=logsys.model_saver.get_latest_model()
        _ = model.load_from(weight_path)

        metric_dict_path = os.path.join(logsys.ckpt_root,'metric_dict')
        if os.path.exists(metric_dict_path):
            logsys.metric_dict.load(torch.load(metric_dict_path))

        routine_ckpt,best_ckpt = logsys.archive_saver(f"archive_at_{start_epoch}")
        show_start_status = True
        doearlystop = False

    inference_once_only = False
    if train_mode == "show_best_performance":
        weight_path=logsys.model_saver.get_best_model()
        print(f"the best trained model at {weight_path}")
        print("we are now going to infer the best model, so any train hyparam will be blocked")
        _ = model.load_from(weight_path)
        show_start_status = True
        do_inference = True
        inference_once_only = True

    if scheduler:scheduler.last_epoch = start_epoch - 1


    FULLNAME          = args.project_json_name #
    banner            = logsys.banner_initial(epoches,FULLNAME)
    master_bar        = logsys.create_master_bar(epoches)
    logsys.banner_show(start_epoch,FULLNAME)
    logsys.train_bar  = logsys.create_progress_bar(1,unit=' img',unit_scale=train_loader.batch_size)
    logsys.valid_bar  = logsys.create_progress_bar(1,unit=' img',unit_scale=valid_loader.batch_size)

    model.accu_list   = accu_list
    model.optimizer   = optimizer
    model.scheduler   = scheduler
    model.criterion   = criterion
    metric_dict       = logsys.metric_dict.metric_dict

    train_loss   = -1
    earlystopQ   = 0  # [two mode]: Q==1 -> enterinto SGD phase
    drop_out_limit= args.train.drop_out_limit if hasattr(args.train,'drop_out_limit') else None
    drop_out_limit= epoches if not drop_out_limit else drop_out_limit
    bad_condition_happen = False

    for epoch in master_bar:
        if epoch < start_epoch:continue
        ### training phase ########
        if epoch > start_epoch or (not show_start_status):
            if drop_rate is not None and hasattr(model,set_drop_prob):model.set_drop_prob(drop_rate* epoch / drop_out_limit if epoch<drop_out_limit else drop_rate)
            # depend on will the first epoch reveal the model performance, default is will
            if hasattr(train_loader.sampler,'set_epoch'):train_loader.sampler.set_epoch(epoch)
            train_loss,train_accu = train_epoch(model,train_loader,logsys)
            if np.isnan(train_loss) or np.isnan(train_accu):raise NanValueError
            logsys.record('the_lr_use_now', model.optimizer.param_groups[0]['lr'] , epoch)
            logsys.record('training_loss', train_loss, epoch)
            saveQ = not (hasattr(args.train,'turn_off_save_latest') and args.train.turn_off_save_latest)
            bad_condition_happen = logsys.save_latest_ckpt(model,epoch,train_loss,saveQ=saveQ,doearlystop=doanormaldt)
            if model.scheduler is not None:
                if hasattr(args.train,'use_swa') and args.train.use_swa and epoch > args.train.swa_start:
                    optimizer    = torch.optim.SGD(model.parameters(), lr=optimizer_config['lr'],momentum=0.9)
                    swa_model    = AveragedModel(model,device=next(model.parameters()).device) if swa_model is None else swa_model
                    swa_scheduler= SWALR(optimizer, swa_lr=0.05) if swa_scheduler is None else swa_scheduler
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                else:
                    try:model.scheduler.step(loss=train_loss)
                    except:model.scheduler.step()

        ### valid phase ########
        if epoch%valid_per_epoch ==0 or show_start_status:
            if hasattr(args.train,'use_swa') and args.train.use_swa and epoch > args.train.swa_start and swa_model is not None:
                swa_update_bn(train_loader, swa_model,device = next(model.parameters()).device)
                valid_acc_pool = test_epoch(swa_model,valid_loader,logsys,accu_list=accu_list)
            else:
                valid_acc_pool = test_epoch(model,valid_loader,logsys,accu_list=accu_list)
            update_accu    = logsys.metric_dict.update(valid_acc_pool,epoch)
            metric_dict    = logsys.metric_dict.metric_dict
            for accu_type in accu_list:
                logsys.record(accu_type, valid_acc_pool[accu_type], epoch)
                logsys.record('best_'+accu_type, metric_dict['best_'+accu_type][accu_type], epoch)
            logsys.banner_show(epoch,FULLNAME,train_losses=[train_loss])
            earlystopQ  = logsys.save_best_ckpt(model,metric_dict,epoch,doearlystop=doearlystop)

            # if model.scheduler is not None:
            #     accu_type = accu_list[0]
            #     model.scheduler.step(metric_dict['best_'+accu_type][accu_type])

        ### inference phase ########
        if do_inference and ((epoch%infer_epoch==0) or (epoch+1 == epoches) or (show_start_status)):
            inference_epoch(model,valid_loader,logsys,epoch,accu_list)
            show_start_status = False

        ### inference phase ########
        if trial:
            trial.report(metric_dict[accu_list[0]], epoch)
            if trial.should_prune():
                if hasattr(args.train,'not_prune') and (args.train.not_prune):
                    pass
                else:
                    raise optuna.TrialPruned()

        if inference_once_only:
            break

        ### earlystop check ########
        if (earlystopQ and epoch>warm_up_epoch):
            if not do_extra_phase:
                break
            else:
                routine_ckpt,best_ckpt = logsys.archive_saver("phase1")
                enter_into_SGD_phase(model,args,logsys,last_ckpt=best_ckpt)
                do_extra_phase = False

        if bad_condition_happen:break


    if inference_once_only:
        logsys.info("========= Stop by finish inference ==============")
    elif earlystopQ:
        logsys.info("========= Stop by earlystop! ==============")
    elif bad_condition_happen:
        logsys.info("========= Stop by bad train loss condition! ==============")
    else:
        logsys.info("========= train finish ==============")

    _ = logsys.save_latest_ckpt(model,epoch,train_loss,saveQ=True,doearlystop=False,force_do = True)
    if len(os.listdir(logsys.model_saver.best_path))<1:
        logsys.info("[!!!]Error: get train stoped, but no best weight saved")
        pass
    else:
        logsys.info(f"we now at epoch {epoch+1}/{epoches}: get best weight:")
        logsys.info(os.listdir(logsys.model_saver.best_path))
    logsys.save_scalars()
    logsys.send_info_to_hub(EXP_HUB)
    logsys.close()
    return metric_dict['best_'+accu_list[0]][accu_list[0]]
def train_for_one_task(model,project):
    # model,project= struct_config(project_config,db=db,build_model=False)
    train_loader = project.train_loader
    valid_loader = project.valid_loader
    PROJECTNAME  = project.project_name
    epoches      = project.train_epoches
    project_config = project.full_config
    print(project_config)
    print("-------------------------------------------------------------------------------------------")
    print("now trainning project: <|{}|>".format(PROJECTNAME))
    print("-------------------------------------------------------------")
    train_mode  = project_config.train_mode if hasattr(project_config,"train_mode") else "new"
    project.train_mode=train_mode
    if train_mode in ["new","replicate"]:
        trial_range = range(project.trials_num)
        for trial in trial_range:
            print(time.asctime( time.localtime(time.time())))
            tp.banner("Trainning processing:trial-{}".format(trial))
            random_seed=int(project_config.random_seed) if train_mode == 'replicate' else random.randint(1, 100000)
            TIME_NOW  = time.strftime("%m_%d_%H_%M_%S")
            TRIAL_NOW = '{}-seed-{}'.format(TIME_NOW,random_seed)
            MODEL_NAME  =project.full_config.model.str_backbone_TYPE
            DATASET_NAME=".".join(PROJECTNAME.split('.')[1:])
            save_checkpoint   = os.path.join(SAVEROOT,'checkpoints',DATASET_NAME,MODEL_NAME,TRIAL_NOW)
            logsys         = LoggingSystem(True,save_checkpoint,seed=random_seed)
            model          = struct_model(project_config,train_loader.dataset)
            #################################################################################
            do_train(model,project,train_loader,valid_loader,logsys)
            #################################################################################
            del model
            torch.cuda.empty_cache()
            project_root_dir  = os.path.join(save_checkpoint,'project_config.json')
            shutil.copy(project.project_json_config_path,project_root_dir)
    elif train_mode == "optuna":
        import optuna
        MODEL_NAME  =project.full_config.model.str_backbone_TYPE
        DATASET_NAME=".".join(PROJECTNAME.split('.')[1:])
        DB_NAME     = project.full_config.project_task_name
        TASK_NAME   = project.full_config.project_json_name.split(".")[0]
        def objective(trial):
            random_seed =random.randint(1, 100000)
            TIME_NOW    = time.strftime("%m_%d_%H_%M_%S")
            TRIAL_NOW   = '{}-seed-{}'.format(TIME_NOW,random_seed)
            MODEL_NAME  =project.full_config.model.str_backbone_TYPE
            DATASET_NAME=".".join(PROJECTNAME.split('.')[1:])
            save_checkpoint   = os.path.join(SAVEROOT,'checkpoints',DATASET_NAME,MODEL_NAME,TRIAL_NOW)
            logsys            = LoggingSystem(True,save_checkpoint,bar_log_path=f"runtime_log/bar_for_job_on_GPU{project.full_config.gpu}",seed=random_seed)
            if not os.path.exists(save_checkpoint):os.makedirs(save_checkpoint)
            project_root_dir  = os.path.join(save_checkpoint,'project_config.json')
            assert project.full_config.train.hypertuner == "optuna"
            project.full_config.optuna_hparam={}
            if hasattr(project.full_config.train,"drop_rate_range"):
                drop_rate = trial.suggest_uniform(f"drop_rate", *project.full_config.train.drop_rate_range)
                project.full_config.train.drop_rate = drop_rate
                project.full_config.optuna_hparam['drop_rate'] = drop_rate
            if hasattr(project.full_config.train,"optimizer_list"):
                optimizer_list = list(project.full_config.train.optimizer_list.keys())
                if len(optimizer_list)==1:
                    optimizer_name = optimizer_list[0]
                    trial.set_user_attr('optim', optimizer_name)
                else:
                    optimizer_name = trial.suggest_categorical("optim", optimizer_list)
                project.full_config.train.optimizer._TYPE_             = optimizer_name
                project.full_config.train.optimizer.str_optimizer_TYPE = optimizer_name
                project.full_config.optuna_hparam['optimer']           = optimizer_name
                optimizer_config = project.full_config.train.optimizer_list[optimizer_name]
                project.full_config.train.optimizer.config={}
                for name,value_list in optimizer_config.items():
                    if isinstance(value_list,list):
                        assert len(value_list)==2
                        list_case_0=isinstance(value_list[0],list)
                        list_case_1=isinstance(value_list[1],list)
                        if (not list_case_0) and (not list_case_1):
                            sampler   = trial.suggest_loguniform if max(value_list)/min(value_list)>50 else trial.suggest_uniform
                            para_name = f"{optimizer_name}_{name}"
                            the_value = sampler(para_name, *value_list)
                            project.full_config.train.optimizer.config[name] = the_value
                            project.full_config.optuna_hparam[para_name] = the_value
                        else:
                            para_name_0 = f"{optimizer_name}_{name}_0"
                            para_name_1 = f"{optimizer_name}_{name}_1"
                            the_value_0 = trial.suggest_uniform(para_name_0, *value_list[0]) if list_case_0 else value_list[0]
                            project.full_config.optuna_hparam[para_name_0] = the_value_0
                            the_value_1 = trial.suggest_uniform(para_name_1, *value_list[1]) if list_case_1 else value_list[1]
                            project.full_config.optuna_hparam[para_name_1] = the_value_1
                            project.full_config.train.optimizer.config[name] = (the_value_0,the_value_1)
                    else:
                        project.full_config.train.optimizer.config[name] = value_list
                        project.full_config.optuna_hparam[name]          = value_list
            if hasattr(project.full_config.model,"criterion_list"):
                criterion_list = project.full_config.model.criterion_list
                if len(criterion_list) == 1:
                    criterion = criterion_list[0]
                    trial.set_user_attr('crit', criterion)
                else:
                    criterion = trial.suggest_categorical("crit", project.full_config.model.criterion_list)
                assert criterion in ["BCELossLogits","CELoss","FocalLoss1","BCELoss"]
                # train_loader.dataset.use_classifier_loss(criterion)
                # valid_loader.dataset.use_classifier_loss(criterion)
                project.full_config.model.criterion_type = criterion #='FocalLoss1'
                project.full_config.optuna_hparam['criterion'] = criterion
                project.full_config.train.accu_list=accu_list=['ClassifierA', 'ClassifierP','ClassifierN']
                ##################################
                ### NITICE: for early project, we use BCELoss rather than BCELoss_withlogics.
                # # the default target is (B,1) index vector
                # # for different loss case this target should be convert to suitable shape
                # if not hasattr(train_loader.dataset,"BCEvector"):
                #     train_loader.dataset.BCEvector  =  train_loader.dataset.vector.reshape(-1,1)
                #     valid_loader.dataset.BCEvector  =  valid_loader.dataset.vector.reshape(-1,1)
                #     train_loader.dataset.BCEcurve_type_shape=(1,)
                #     valid_loader.dataset.BCEcurve_type_shape=(1,)
                # if not hasattr(train_loader.dataset,"CEvector"):
                #     train_loader.dataset.CEcurve_type_shape=(2,)
                #     valid_loader.dataset.CEcurve_type_shape=(2,)
                # # if not hasattr(train_loader.dataset,"FCEvector"):
                # #     train_loader.dataset.FCEvector=torch.zeros(train_loader.dataset.vector.shape[0], 2).scatter_(1, train_loader.dataset.vector.unsqueeze(1).long(), 1)
                # #     valid_loader.dataset.FCEvector=torch.zeros(valid_loader.dataset.vector.shape[0], 2).scatter_(1, valid_loader.dataset.vector.unsqueeze(1).long(), 1)
                # #     train_loader.dataset.FCEcurve_type_shape=(2,)
                # #     valid_loader.dataset.FCEcurve_type_shape=(2,)
                # if   criterion == "BCELoss":
                #     train_loader.dataset.vector=train_loader.dataset.BCEvector
                #     valid_loader.dataset.vector=valid_loader.dataset.BCEvector
                #     train_loader.dataset.curve_type_shape=train_loader.dataset.BCEcurve_type_shape
                #     valid_loader.dataset.curve_type_shape=valid_loader.dataset.BCEcurve_type_shape
                #     project.full_config.train.accu_list=accu_list=['BinaryAL', 'BinaryPL','BinaryNL']
                # elif criterion == "CELoss":
                #     train_loader.dataset.vector=train_loader.dataset.BCEvector
                #     valid_loader.dataset.vector=valid_loader.dataset.BCEvector
                #     train_loader.dataset.curve_type_shape=train_loader.dataset.CEcurve_type_shape
                #     valid_loader.dataset.curve_type_shape=valid_loader.dataset.CEcurve_type_shape
                #     project.full_config.train.accu_list=accu_list=['OneHotError', 'OneHotP','OneHotN']
                # elif criterion == "FocalLoss1":
                #     train_loader.dataset.vector=train_loader.dataset.FCEvector
                #     valid_loader.dataset.vector=valid_loader.dataset.FCEvector
                #     train_loader.dataset.curve_type_shape=train_loader.dataset.FCEcurve_type_shape
                #     valid_loader.dataset.curve_type_shape=valid_loader.dataset.FCEcurve_type_shape
                #     project.full_config.train.accu_list=accu_list=['BinaryA', 'BinaryP','BinaryN']
            #print(f"use criterion:{criterion}")
            #print(f"use accu_list:{accu_list}")
            if hasattr(project.full_config.train,"grad_clip_list"):
                grad_clip_list = project.full_config.train.grad_clip_list
                if len(grad_clip_list) == 1:
                    grad_clip = grad_clip_list[0]
                    trial.set_user_attr('g_clip', grad_clip_list)
                else:
                    grad_clip = trial.suggest_categorical("g_clip", project.full_config.train.grad_clip_list)
                project.full_config.train.grad_clip = grad_clip
                project.full_config.optuna_hparam['grad_clip'] = grad_clip
            trial.set_user_attr('trial_name', TRIAL_NOW)
            if os.path.exists(project.project_json_config_path):
                config_for_this_trial=project.full_config.copy({'train_mode':"new"})
                config_for_this_trial.save(project_root_dir)
                #raise
                #shutil.copy(project.project_json_config_path,project_root_dir)
            else:
                return -1

            model  = struct_model(project_config,train_loader.dataset)
            result = do_train(model,project,train_loader,valid_loader,logsys,trial=trial)
            del model
            torch.cuda.empty_cache()
            return result

        study = optuna.create_study(study_name=TASK_NAME, storage=f'sqlite:///optuna_database/{DB_NAME}.db',
                                        load_if_exists=True,
                                        sampler=optuna.samplers.CmaEsSampler(),
                                        pruner=optuna.pruners.MedianPruner(n_warmup_steps=28)
                                    )
        optuna_limit_trials = project.full_config.train.optuna_limit_trials if hasattr(project.full_config.train,"optuna_limit_trials") else 30
        if len([t.state for t in study.trials if t.state== TrialState.COMPLETE])>optuna_limit_trials:raise
        #study.optimize(objective, n_trials=50, timeout=600,pruner=optuna.pruners.MedianPruner())
        hypertuner_config = project.full_config.train.hypertuner_config if hasattr(project.full_config.train,"hypertuner_config") else {'n_trials':10}
        study.optimize(objective, **hypertuner_config)
        del model
        torch.cuda.empty_cache()
        project_root_dir = None
    else:
        save_checkpoint   = project_config.last_checkpoint
        logsys            = LoggingSystem(True,save_checkpoint)
        model             = struct_model(project_config,train_loader.dataset)
        do_train(model,project,train_loader,valid_loader,logsys)
        #################################################################################
        del model
        torch.cuda.empty_cache()
        project_root_dir  = os.path.join(save_checkpoint,'project_config.json')
    return project_root_dir
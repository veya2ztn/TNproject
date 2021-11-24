import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from utils import preprocess_sincos, preprocess_binary
from models.simple_model import MPSLinear
def preprocess_images(x):
    return (1-(aggregation_patch(x[...,3:27,3:27],divide=4)))/50

lr           = 0.1
batch_size   = 64
virtual_bond = 7
init_std     = 1

transform = transforms.Compose([transforms.ToTensor()])
DATAPATH    = '/media/tianning/DATA/DATASET/MNIST/'
mnist_train = datasets.MNIST(DATAPATH, train=True, download=False, transform=transform)
mnist_test  = datasets.MNIST(DATAPATH, train=False,download=False, transform=transform)
train_loader= torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test,  batch_size=batch_size, shuffle=False)

in_feature = 28*28
classes_num= 10
model = PEPS_einsum_uniform_shape(6,6,10,in_physics_bond=16,virtual_bond_dim=virtual_bond,init_std=init_std)
device = 'cuda'
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

import time
from mltool.dataaccelerate import DataSimfetcher
from mltool.loggingsystem import LoggingSystem

# main loop
logsys            = LoggingSystem(True,"log/test")
metric_list       = ['loss','accu']
lses=[]
lres=[]
metric_dict       = logsys.initial_metric_dict(metric_list)
master_bar        = logsys.create_master_bar(10)
#master_bar.set_multiply_graph(figsize=(12,4),engine=[['plot']*len(metric_list)],labels=[metric_list])
for epoch in master_bar:
    start_time = time.time()
    model.train()
    infiniter = DataSimfetcher(train_loader, device=device)
    inter_b   = logsys.create_progress_bar(len(train_loader))
    while inter_b.update_step():
        image,label= infiniter.next()
        optimizer.zero_grad()
        data       = preprocess_images(image)
        logits     = model(data).squeeze(-1)
        loss       = torch.nn.CrossEntropyLoss()(logits,label)
        loss.backward()
        #for p in model.parameters():nn.utils.clip_grad_norm_(p, max_norm=1)
        optimizer.step()
        lses.append(loss.item())
    model.eval()
    prefetcher = DataSimfetcher(test_loader, device=device)
    inter_b    = logsys.create_progress_bar(len(test_loader))
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

    lres.append(accu)
    #master_bar.update_graph_multiply([[lses,lres]])
    print('\nEpoch: %.3i \t Accu: %.4f \t Time: %.2f s' %(epoch, accu.item(), time.time() - start_time))

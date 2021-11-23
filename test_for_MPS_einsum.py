import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import preprocess_binary,preprocess_sincos
import time
from mltool.dataaccelerate import DataSimfetcher
from mltool.loggingsystem import LoggingSystem
from torchvision import datasets, transforms
mnist_data = np.load('archive/tn-for-unsup-ml/data/binarized_mnist.npz')
train_data = torch.from_numpy(mnist_data['train_data'])
test_data = torch.from_numpy(mnist_data['test_data'])

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=(0.0,), std=(1.0,))
])
DATAPATH    = '/media/tianning/DATA/DATASET/MNIST/'
mnist_train = datasets.MNIST(DATAPATH, train=True, download=False, transform=transform)
mnist_test  = datasets.MNIST(DATAPATH, train=False,download=False, transform=transform)

from models.mps import MPSLinear
def preprocess_images(x):return preprocess_binary(x)

train_loader= torch.utils.data.DataLoader(dataset=mnist_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test , batch_size=64, shuffle=False)
#from models import MPSLinear
model = MPSLinear(28*28,10,in_physics_bond = 2, out_physics_bond=1, virtual_bond_dim=100,
                  bias=False,label_position='center',init_std=0)

#model  = AMPSShare(n=28*28, bond_dim=10, phys_dim=2)
device = 'cuda'
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# main loop
logsys            = LoggingSystem(True,"log/test")
metric_list       = ['loss','accu']
lses=[]
lres=[]
metric_dict       = logsys.initial_metric_dict(metric_list)
master_bar        = logsys.create_master_bar(10)
for epoch in master_bar:
    start_time = time.time()
    model.train()
    infiniter = DataSimfetcher(train_loader, device=device)
    inter_b   = logsys.create_progress_bar(len(train_loader))
    while inter_b.update_step():
        image,label= infiniter.next()
        bs,c,w,h = image.shape
        optimizer.zero_grad()
        binary     = preprocess_images(image)
        logits     = model(binary).squeeze()
        loss       = torch.nn.CrossEntropyLoss()(logits,label)
        loss.backward()
        if torch.isnan(loss):raise
        optimizer.step()
        lses.append(loss.item())
        #if inter_b.now%20==0:master_bar.update_graph_multiply([[lses,lres]])
    model.eval()
    prefetcher = DataSimfetcher(test_loader, device=device)
    inter_b    = logsys.create_progress_bar(len(test_loader))
    labels     = []
    logits     = []
    with torch.no_grad():
        while inter_b.update_step():
            image,label= prefetcher.next()
            binary     = preprocess_images(image)
            logit      = model(binary).squeeze()
            loss       = torch.nn.CrossEntropyLoss()(logit ,label)
            labels.append(label)
            logits.append(logit)
    labels  = torch.cat(labels)
    logits  = torch.cat(logits)
    pred_labels  = torch.argmax(logits,-1)
    accu =  torch.sum(pred_labels == labels)/len(labels)

    lres.append(accu)
    logsys.info('\nEpoch: %.3i \t Accu: %.4f \t Time: %.2f s' %(epoch, accu.item(), time.time() - start_time))

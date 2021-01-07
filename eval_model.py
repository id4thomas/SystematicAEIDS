from __future__ import absolute_import, print_function
import torch

import argparse
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from model.ae import AE

from utils.data_utils import *
from utils.perf_utils import *
# from utils.reduc_utils import *
from utils.plot_utils import *

from sklearn.metrics import average_precision_score
from scipy.spatial import distance

import copy
import numpy as np

import wandb
wandb.init(project="svcc-cicids17")

ATK=1
SAFE=0

# Argument Setting
parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int,
                    help="random seed for reproductability")

#Model Config
parser.add_argument("--l_dim", default=10, type=int,
                    help="Latent Dim")
parser.add_argument("--num_layers", default=2, type=int,
                    help="number of layers")
parser.add_argument("--size", default=64, type=int,
                    help="Smallest Hid Size")
#Regularization
parser.add_argument("--do", default=0, type=float,
                    help="dropout rate")
parser.add_argument("--bn", default=0, type=int,
                    help="batch norm: 1 to use")

parser.add_argument("--epoch", default=10, type=int,
                    help="training epochs")
parser.add_argument("--batch_size", default=8192, type=int,
                    help="batch size for train and test")
parser.add_argument("--lr", default=1e-4, type=float,
                    help="learning rate")

args = parser.parse_args()

# Fix seed
set_seed(args.seed)
device = torch.device('cuda:0')

#GET ALL DATA!
data_path='data/cicids17/split/'
x_val,_=get_hdf5_data(data_path+'val.hdf5',labeled=False)
y_val_type=np.load(data_path+'val_label.npy',allow_pickle=True)
y_val=np.zeros(x_val.shape[0])
y_val[y_val_type!='BENIGN']=1
print("Val: Normal:{}, Atk:{}".format(x_val[y_val_type=='BENIGN'].shape[0],x_val[y_val_type!='BENIGN'].shape[0]))
#Balanced
data=make_balanced_val({'x_val': x_val, 'y_val': y_val})
x_val=data['x_val']
y_val=data['y_val']
print("Bal Val: Normal:{}, Atk:{}".format(x_val[y_val==SAFE].shape[0],x_val[y_val!=SAFE].shape[0]))


x_test,_=get_hdf5_data(data_path+'test.hdf5',labeled=False)
y_test_type=np.load(data_path+'test_label.npy',allow_pickle=True)
y_test=np.zeros(x_test.shape[0])
y_test[y_test_type!='BENIGN']=1
print("Test: Normal:{}, Atk:{}".format(x_test[y_test_type=='BENIGN'].shape[0],x_test[y_test_type!='BENIGN'].shape[0]))

#Get Model
layers=[args.l_dim]
for i in range(0,args.num_layers):
    #Multiplying
    # layers.append(args.l_dim*2**(i))
    #Fixed
    layers.append(args.size*2**(i))
layers.reverse()
model_config={
    'd_dim':80,
    'layers':layers
}
model_desc='AE-{}_{}_{}'.format(args.size,args.l_dim,args.num_layers)
print(f"Model {model_desc}")
model=AE(model_config).to(device)

with open("./weights_{}/cic_{}_{}_{}_{}_{}.pt".format(args.num_layers,args.size,args.l_dim,args.epoch,args.batch_size,args.seed), "rb") as f:
    best_model = torch.load(f)
    
#L2
model.load_state_dict(best_model["state_l2"])

x_val_cuda = torch.from_numpy(x_val).float().to(device)
eval_sampler = SequentialSampler(x_val_cuda)
eval_dataloader = DataLoader(x_val_cuda, sampler=eval_sampler, batch_size=args.batch_size)

x_test_cuda = torch.from_numpy(x_test).float().to(device)
test_sampler = SequentialSampler(x_test_cuda)
test_dataloader = DataLoader(x_test_cuda, sampler=test_sampler, batch_size=args.batch_size)

model.eval()
with torch.no_grad():
    pred_val=[]
    val_loss=[]
    for batch in test_dataloader:
        target = batch.type(torch.float32)

        outputs = model(target)
        batch_error = model.compute_loss(outputs, target)

        pred_val.append(outputs['output'].cpu().detach().numpy())
        val_loss.append(batch_error.item())

    pred_val=np.concatenate(pred_val)

test_dist_l2=np.mean(np.square(x_test-pred_val),axis=1)
# test_dist_cos=[distance.cosine(x_test[i],pred_val[i]) for i in range(x_val.shape[0])]
auc_l2,_,_=make_roc(test_dist_l2,y_test,ans_label=ATK)
print("Test AUC L2: {:.5f}".format(auc_l2))
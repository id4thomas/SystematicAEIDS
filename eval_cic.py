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

# import wandb
# wandb.init(project="svcc-cicids17")

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
parser.add_argument("--dist", default="l2", type=str,
                    help="reconstruction dist")
parser.add_argument('--bal', action='store_true')
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

####Balanced Test
# balanced_test=True
balanced_test=args.bal
if balanced_test:
    data=make_balanced_test({'x_test': x_test, 'y_test': y_test,'y_test_type':y_test_type},has_type=True)
    x_test=data['x_test']
    y_test=data['y_test']
    y_test_type=data['y_test_type']
    log_name="perf_results/cic_test_perf_bal.txt"
    print("Balanced")
else:
    print("Not Balanced")
    log_name="perf_results/cic_test_perf.txt"

print(f"Distance {args.dist}")
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

def check_th(val_dist,y_val,z,avg_type='binary'):
    mean_l2=np.mean(val_dist)
    var_l2=np.var(val_dist)
    stddev_l2=np.std(val_dist)

    #Get value with only safe
    val_dist_safe=val_dist[y_val==0]
    mean_l2=np.mean(val_dist_safe)
    var_l2=np.var(val_dist_safe)
    stddev_l2=np.std(val_dist_safe)

    # l2_dist_std=ss.zscore(data)
    th=mean_l2+stddev_l2*z
    # print("Th{:.5f} Mean{:.5f} Var{:.10f}".format(th,mean_l2,var_l2))
    # print("Val")
    y_pred=np.zeros_like(y_val)
    # test_l2_std=ss.zscore(test_l2)
    y_pred[val_dist>th]=1
    precision, recall, f_score, support = precision_recall_fscore_support(y_val, y_pred, pos_label=1, average=avg_type)
    return f_score,th

#Val Eval
model.eval()
with torch.no_grad():
    pred_val=[]
    for batch in eval_dataloader:
        target = batch.type(torch.float32)
        # print(target.shape)
        outputs = model(target)
        batch_error = model.compute_loss(outputs, target)
        pred_val.append(outputs['output'].cpu().detach().numpy())
    pred_val=np.concatenate(pred_val)

if args.dist=="l2":
    val_dist=np.mean(np.square(x_val-pred_val),axis=1)
else:
    #cos
    val_dist=np.array([distance.cosine(x_val[i],pred_val[i]) for i in range(x_val.shape[0])])
    print(val_dist)
val_dist_safe=val_dist[y_val==0]
comb_dist_mean=np.mean(val_dist_safe)
comb_dist_std=np.std(val_dist_safe)

val_dist_norm=(val_dist-comb_dist_mean)/comb_dist_std

val_dist_safe=val_dist[y_val==0]

comb_dist_mean=np.mean(val_dist_safe)
comb_dist_std=np.std(val_dist_safe)

val_dist_norm=(val_dist-comb_dist_mean)/comb_dist_std

avg_type='binary'

best_f1=0
best_th=0
best_z=0
for z in range(1,1001):
    cand=-1+0.01*z
    # print("\nCand",cand)

    f1,th=check_th(val_dist,y_val,cand,avg_type=avg_type)
    if best_f1<f1:
        best_f1=f1
        best_th=th
        best_z=cand
comb_best_z=best_z
print("Best z {:.3f} th {:.10f}".format(best_z,best_th))

#Test Eval
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

if args.dist=="l2":
    test_dist=np.mean(np.square(x_test-pred_val),axis=1)
else:
    test_dist=np.array([distance.cosine(x_test[i],pred_val[i]) for i in range(x_test.shape[0])])

test_auc,_,_=make_roc(test_dist,y_test,ans_label=ATK)
print("Test AUC L2: {:.5f}".format(test_auc))

#Standardize Test Dist
test_dist_norm=(test_dist-comb_dist_mean)/comb_dist_std

y_pred=np.zeros_like(y_test)
y_pred[test_dist_norm>comb_best_z]=1
prf(y_test,y_pred,ans_label=1,avg_type=avg_type)
accuracy = accuracy_score(y_test,y_pred)
precision, recall, f_score, support = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average=avg_type)
f_0_5=fbeta_score(y_test, y_pred, pos_label=1, average=avg_type, beta=0.5)
f_2=fbeta_score(y_test, y_pred,pos_label=1, average=avg_type, beta=2)
print("F1 {:.5f} F0.5 {:.5f} F2 {:.5f}\n".format(f_score,f_0_5,f_2))



if not os.path.isfile(log_name):
     with open(log_name, "a") as myfile:
         myfile.write("size,num_layers,l_dim,epoch,batch,dropout,bn,dist,"+"auc,z,acc,p,r,f,label"+",seed\n")

with open(log_name, "a") as myfile:
    #L2
    myfile.write(f"{args.size},{args.num_layers},{args.l_dim},")
    myfile.write(f"{args.epoch},{args.batch_size},")
    myfile.write(f"{args.do},{args.bn},")
    #PERF
    # myfile.write("l2,{:.5f},{},".format(model_best['auc_l2'],model_best['epoch_l2']))
    myfile.write("{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},total,".format(args.dist,test_auc,comb_best_z,accuracy,precision,recall,f_score))
    ##
    myfile.write(f"{args.seed}\n")
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
wandb.init(project="svcc-iotbotnet")

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

parser.add_argument('--dec', action='store_true',
                    help="Decrease From Input")
parser.add_argument('--dec_rate',type=float,
                    help="Decrease From Input Rate")
# parser.add_argument("--val_batch", default=8192, type=int,
#                     help="batch size for train and test")

parser.add_argument("--lr", default=1e-4, type=float,
                    help="learning rate")

# parser.add_argument("--data", default="iotbotnet", type=str,
#                         help="Dataset")
args = parser.parse_args()

# Fix seed
set_seed(args.seed)
device = torch.device('cuda:0')

##### Load Data #####
data_path='data/iotbotnet/processed/'
x_train,y_train=get_hdf5_data(data_path+'train.hdf5',labeled=True)
x_val,y_val=get_hdf5_data(data_path+'val.hdf5',labeled=True)

print("Val: Normal:{}, Atk:{}".format(x_val[y_val==SAFE].shape[0],x_val[y_val!=SAFE].shape[0]))

#Filter atk from train
x_train,y_train=filter_label(x_train,y_train,select_label=SAFE)
print("Filter Train: Normal:{}, Atk:{}".format(x_train[y_train==0].shape[0],x_train[y_train==1].shape[0]))

#Balanced
# data=make_balanced({'x': x_val, 'y': y_val})
# x_val=data['x']
# y_val=data['y']
# print("Bal Val: Normal:{}, Atk:{}".format(x_val[y_val==SAFE].shape[0],x_val[y_val!=SAFE].shape[0]))

print("Train",x_train.shape)
print("Val",x_val.shape)
# exit()

#Get Model

layer_dec_rates=[0.75,0.5,0.33,0.25]
# layer_dec_rates=[args.dec_rate,]
if args.dec:
    layers=[]
    for i in range(0,args.num_layers):
        #Multiplying
        # layers.append(args.l_dim*2**(i))
        #Fixed
        # layers.append(args.size*2**(i))
        #Decreasing Rate Fixed
        # layers.append(int(x_train.shape[1]*layer_dec_rates[i]))
        layers.append(int(x_train.shape[1]*args.dec_rate**(i+1)))
    layers.append(args.l_dim)
else:
    #Increase from Smallest Hid
    layers=[args.l_dim]
    for i in range(0,args.num_layers):
        #Multiplying
        # layers.append(args.l_dim*2**(i))
        #Fixed
        layers.append(args.size*2**(i))
        #Decreasing Rate
        # layers.append(int(x_train.shape[0]*layer_dec_rates[i]))
        layers.reverse()
model_config={
    'd_dim':x_train.shape[1],
    'layers':layers
}
model_desc='AE-{}_{}_{}'.format(args.dec_rate,args.l_dim,args.num_layers)
model=AE(model_config).to(device)

#Check Model
from torchsummary import summary
print(summary(model,(args.batch_size,x_train.shape[1])))
# exit()

#Init Wandb
wandb.watch(model)

#Xavier Init Weights
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
model.apply(init_weights)


#Train
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

model.zero_grad()
model.train(True)

#Load to Cuda
x_train = torch.from_numpy(x_train).float().to(device)
data_sampler = RandomSampler(x_train)
data_loader = DataLoader(x_train, sampler=data_sampler, batch_size=args.batch_size)

x_val_cuda = torch.from_numpy(x_val).float().to(device)
eval_sampler = SequentialSampler(x_val_cuda)
eval_dataloader = DataLoader(x_val_cuda, sampler=eval_sampler, batch_size=x_val.shape[0])#args.batch_size)

#History
train_hist={
    'loss':[]
}
val_hist={
    'loss':[],
    'auc_l2':[],
    'auc_cos':[],
    'f1':[]
}
model_best={
    'auc_l2':0,
    'auc_cos':0,

    'state_l2':None,
    'state_cos':None,

    'epoch_l2':0,
    'epoch_cos':0,
}


#Evaluate before start
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
    y_pred=np.zeros_like(y_val)
    # test_l2_std=ss.zscore(test_l2)
    y_pred[val_dist>th]=1
    precision, recall, f_score, support = precision_recall_fscore_support(y_val, y_pred, pos_label=1, average=avg_type)
    return f_score,th


model.eval()
with torch.no_grad():
    pred_val=[]
    val_loss=[]

    for batch in eval_dataloader:
        target = batch.type(torch.float32)

        outputs = model(target)
        batch_error = model.compute_loss(outputs, target)

        pred_val.append(outputs['output'].cpu().detach().numpy())
        val_loss.append(batch_error.item())

    pred_val=np.concatenate(pred_val)

val_dist_l2=np.mean(np.square(x_val-pred_val),axis=1)
val_dist_cos=[distance.cosine(x_val[i],pred_val[i]) for i in range(x_val.shape[0])]
# val_dist_safe=val_dist_l2[y_val==0]

# comb_dist_mean=np.mean(val_dist_safe)
# comb_dist_std=np.std(val_dist_safe)
# val_dist_norm=(val_dist_l2-comb_dist_mean)/comb_dist_std

# best_f1=0
# best_th=0
# best_z=0
# for z in range(1,1001):
#     cand=-1+0.01*z
#     # print("\nCand",cand)

#     f1,th=check_th(val_dist_l2,y_val,cand,avg_type='binary')
#     if best_f1<f1:
#         best_f1=f1
#         best_th=th
#         best_z=cand

auc_l2,_,_=make_roc(val_dist_l2,y_val,ans_label=ATK)
auc_cos,_,_=make_roc(val_dist_cos,y_val,ans_label=ATK)
val_hist['auc_l2'].append(auc_l2)
val_hist['auc_cos'].append(auc_cos)
# val_hist['f1'].append(best_f1)
# wandb.log({"{}_{} F1".format(model_desc,args.batch_size):best_f1,"{}_{} AUC".format(model_desc,args.batch_size):auc_l2})


batch_iter=0
stop_train=False
for epoch in range(args.epoch):
    epoch_loss = []
    print("\nTraining Epoch",epoch+1)
    model.train()
    for step, batch in enumerate(data_loader):
        batch_iter+=1
        target = batch.type(torch.float32)

        outputs = model(target)
        loss = model.compute_loss(outputs, target)

        loss.backward()
        optimizer.step()
        model.zero_grad()

        epoch_loss.append(loss.item())

        # if batch_iter%100==0:
    # print("Batch {}: Train: {:.5f}".format(batch_iter, sum(epoch_loss)/len(epoch_loss)))
    train_loss=loss.item()
    # print("Epoch {}: Train: {:.5f}".format(epoch+1,train_loss))
    #Validation loss
    # if (epoch+1) %5==0:
    print("Val")
    model.eval()
    with torch.no_grad():
        pred_val=[]
        val_loss=[]
        for batch in eval_dataloader:
            target = batch.type(torch.float32)

            outputs = model(target)
            batch_error = model.compute_loss(outputs, target)

            pred_val.append(outputs['output'].cpu().detach().numpy())
            val_loss.append(batch_error.item())

        pred_val=np.concatenate(pred_val)

    val_dist_l2=np.mean(np.square(x_val-pred_val),axis=1)
    val_dist_cos=[distance.cosine(x_val[i],pred_val[i]) for i in range(x_val.shape[0])]
    # val_dist_safe=val_dist_l2[y_val==0]

    # comb_dist_mean=np.mean(val_dist_safe)
    # comb_dist_std=np.std(val_dist_safe)
    # val_dist_norm=(val_dist_l2-comb_dist_mean)/comb_dist_std

    # best_f1=0
    # best_th=0
    # best_z=0
    # for z in range(1,1001):
    #     cand=-1+0.01*z
    #     # print("\nCand",cand)

    #     f1,th=check_th(val_dist_l2,y_val,cand,avg_type='binary')
    #     if best_f1<f1:
    #         best_f1=f1
    #         best_th=th
    #         best_z=cand
            
    print("iter {}: Train: {:.5f}, Val: {:.5f}".format(batch_iter, train_loss,sum(val_loss)/len(val_loss)))

    print("\nL2")
    # print('Average Precision',average_precision_score(y_val, val_dist_l2, pos_label=ATK))
    auc_l2,_,_=make_roc(val_dist_l2,y_val,ans_label=ATK)
    auc_cos,_,_=make_roc(val_dist_cos,y_val,ans_label=ATK)
    print("AUC L2 {:.5f} Cos {:.5f}".format(auc_l2,auc_cos))
    #Save Hist
    train_hist['loss'].append(sum(epoch_loss)/len(epoch_loss))

    val_hist['loss'].append(sum(val_loss)/len(val_loss))
    val_hist['auc_l2'].append(auc_l2)
    val_hist['auc_cos'].append(auc_cos)
    # val_hist['f1'].append(best_f1)
    # wandb.log({"{}_{} F1".format(model_desc,args.batch_size):best_f1,"{}_{} AUC".format(model_desc,args.batch_size):auc_l2,"{}_{} Val Loss".format(model_desc,args.batch_size):sum(val_loss)/len(val_loss),"{}_{} Train Loss".format(model_desc,args.batch_size):train_loss})
    wandb.log({"{}_{} Val Loss".format(model_desc,args.batch_size):sum(val_loss)/len(val_loss),"{}_{} Train Loss".format(model_desc,args.batch_size):train_loss})
    wandb.log({"{}_{} AUC L2".format(model_desc,args.batch_size):auc_l2})
    wandb.log({"{}_{} AUC Cos".format(model_desc,args.batch_size):auc_cos})
    
    # val_loss=sum(val_loss)/len(val_loss)
    val_loss=auc_l2

    prev_loss=val_loss
    #Save Model
    if auc_l2>model_best['auc_l2']:
        model_best['state_l2']=copy.deepcopy(model.state_dict())
        # model_best['epoch']=batch_iter
        model_best['epoch_l2']=epoch+1
        model_best['auc_l2']=auc_l2
    
    if auc_cos>model_best['auc_cos']:
        model_best['state_cos']=copy.deepcopy(model.state_dict())
        model_best['auc_cos']=auc_cos
        model_best['epoch_cos']=epoch+1
                


print("\nTrain Complete")
if not os.path.isfile("perf_results/iotbotnet_train_log.txt"):
     with open("perf_results/iotbotnet_train_log.txt", "a") as myfile:
         myfile.write("size,num_layers,l_dim,epoch,batch,dropout,bn,dist,best_auc,best_ep,seed,dec_rate\n")
with open("perf_results/iotbotnet_train_log.txt", "a") as myfile:
    #L2
    myfile.write(f"{args.size},{args.num_layers},{args.l_dim},")
    myfile.write(f"{args.epoch},{args.batch_size},")
    myfile.write(f"{args.do},{args.bn},")
    myfile.write("l2,{:.5f},{},".format(model_best['auc_l2'],model_best['epoch_l2']))
    myfile.write(f"{args.seed},{args.dec_rate}\n")
    
    myfile.write(f"{args.size},{args.num_layers},{args.l_dim},")
    myfile.write(f"{args.epoch},{args.batch_size},")
    myfile.write(f"{args.do},{args.bn},")
    myfile.write("cos,{:.5f},{},".format(model_best['auc_cos'],model_best['epoch_cos']))
    myfile.write(f"{args.seed},{args.dec_rate}\n")
    
    # myfile.write(f"{args.do},{args.bn},")
    # myfile.write("Val L2 Best {:.5f} at {}\n".format(model_best['auc_l2'],model_best['epoch_l2']))
    # myfile.write("Val Cos Best {:.5f} at {}\n".format(model_best['auc_cos'],model_best['epoch_cos']))
# exit()
#Make Folder
if not os.path.exists('weights_{}'.format(args.num_layers)):
    os.makedirs('weights_{}'.format(args.num_layers))
if not os.path.exists('loss_plot'):
    os.makedirs('loss_plot')
if not os.path.exists('auc_plot'):
    os.makedirs('auc_plot') 
# if not os.path.exists('f1_plot'):
#     os.makedirs('f1_plot') 

with open("./weights_{}/iotbotnet_{}_{}_{}_{}_{}.pt".format(args.num_layers,args.dec_rate,args.l_dim,args.epoch,args.batch_size,args.seed), "wb") as f:
    torch.save(
        {
            "state_l2": model_best['state_l2'],
            "state_cos": model_best['state_cos'],
        },
        f,
    )
# exit()
#Loss Plot
loss_fig=plot_losses(train_hist['loss'],val_hist['loss'],"Loss History")
loss_fig.savefig('./loss_plot/iotbotnet_{}_{}_{}_{}_{}_{}.png'.format(args.num_layers,args.dec_rate,args.l_dim,args.epoch,args.batch_size,args.seed))

# #AUC Plot
auc_fig=plot_auc(val_hist['auc_l2'])
auc_fig.savefig('./auc_plot/iotbotnet_l2_{}_{}_{}_{}_{}.png'.format(args.num_layers,args.dec_rate,args.l_dim,args.epoch,args.batch_size,args.seed))

auc_fig=plot_auc(val_hist['auc_cos'])
auc_fig.savefig('./auc_plot/iotbotnet_cos_{}_{}_{}_{}_{}.png'.format(args.num_layers,args.dec_rate,args.l_dim,args.epoch,args.batch_size,args.seed))
exit()

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

from sklearn.model_selection import KFold
from data.preprocess_iotid20 import *

#Saving
import joblib
import pickle
from torchsummary import summary
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

parser.add_argument('--dist_cos', action='store_true',
                    help="Use Cosine as Dist")
# parser.add_argument("--data", default="iotid20", type=str,
#                         help="Dataset")
args = parser.parse_args()

wandb.init(project="svcc-iotid20")
# Fix seed
set_seed(args.seed)
device = torch.device('cuda:0')

data_dir='data/iotid20/split'
train=pd.read_csv(data_dir+'/train.csv')
val=pd.read_csv(data_dir+'/val.csv')

kf = KFold(n_splits=5)

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
     
#Get Model Predictions
def get_model_preds(model,loader):
    model.eval()
    with torch.no_grad():
        pred=[]
        loss=[]
        for batch in loader:
            target = batch.type(torch.float32)

            outputs = model(target)
            batch_error = model.compute_loss(outputs, target)

            pred.append(outputs['output'].cpu().detach().numpy())
            loss.append(batch_error.item())

        pred=np.concatenate(pred)
    return pred,loss

def get_threshold(dist,y,avg_type='binary'):
    dist_safe=dist[y==0]
    mean=np.mean(dist_safe)
    std=np.std(dist_safe)
    
    best_score=0
    best_th=0
    best_z=0
    
    #Z Score Search Range
    for z in range(1,801):
        cand=-4+0.01*z
        # print("\nCand",cand)
        score,th=check_th(dist,y,mean,std,cand,avg_type='binary')
        if best_score<score:
            best_score=score
            best_th=th
            best_z=cand
            
    print("Best z {:.3f} th {:.10f} with score {:.5f}".format(best_z,best_th,best_score))
    return mean,std,best_z

def check_th(dist,y,mean,stddev,z,avg_type='binary'):
    th=mean+stddev*z
    
    y_pred=np.zeros_like(y)
    y_pred[dist>th]=1
    # precision, recall, f_score, support = precision_recall_fscore_support(y, y_pred, pos_label=1, average=avg_type)
    # score=f_score
    score=mcc(y,y_pred)
    return score,th

best_val_mcc=0
best_weights=None
best_scaler=None
best_num_desc=None
best_fold=0
best_std=0
best_mean=0
best_z=0
k=0

log_name="perf_results/iotid20_train_perf_fixed.txt"
if not os.path.isfile(log_name):
     with open(log_name, "a") as myfile:
         myfile.write("size,num_layers,l_dim,epoch,batch,dropout,bn,dist,"+"auc,z,acc,p,r,f,"+"fpr,mcc,"+"label,seed,dec_rate,fold,best_epoch\n")
         
for train_index, val_index in kf.split(train):
    k=k+1
    model_desc='AE-{}_{}_{}_{}-fold{}'.format(args.size,args.dec_rate,args.l_dim,args.num_layers,k)
    #Split
    #Training Model
    train_t=train.iloc[train_index]
    train_v=train.iloc[val_index]
    
    train_t_df,_,y_train_t,scaler=preprocess(train_t)  
    train_v_df,_,y_train_v,_=preprocess(train_v,scaler=scaler)
    
    x_train_t=train_t_df.values
    x_train_v=train_v_df.values
    
    x_train_t,y_train_t=filter_label(x_train_t,y_train_t,select_label=SAFE)
    
    print("Filter Train-t: Normal:{}, Atk:{}".format(x_train_t[y_train_t==0].shape[0],x_train_t[y_train_t==1].shape[0]))
    print("Filter Train-v: Normal:{}, Atk:{}".format(x_train_v[y_train_v==0].shape[0],x_train_v[y_train_v==1].shape[0]))
    #Train
    x_train_t = torch.from_numpy(x_train_t).float().to(device)
    data_sampler = RandomSampler(x_train_t)
    data_loader = DataLoader(x_train_t, sampler=data_sampler, batch_size=args.batch_size)

    x_train_v_cuda = torch.from_numpy(x_train_v).float().to(device)
    eval_sampler = SequentialSampler(x_train_v_cuda)
    eval_dataloader = DataLoader(x_train_v_cuda, sampler=eval_sampler, batch_size=args.batch_size)
    
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
    
    batch_iter=0
    if args.dec:
        layers=[]
        for i in range(0,args.num_layers):
            layers.append(int(x_train.shape[1]*args.dec_rate**(i+1)))
        layers.append(args.l_dim) 
        
        #Set Val for logs
        # size_val=args.dec_rate
        size_val="In"
        model_dec_type="dec"
        
    else:
        #Decrease from Biggest - default 0.5
        layers=[]
        for i in range(0,args.num_layers):
            layers.append(int(args.size*args.dec_rate**(i)))
        layers.append(args.l_dim)
        size_val=args.size
        model_dec_type="fixed"
        
    model_config={
        'd_dim':x_train_t.shape[1],
        'layers':layers
    } 
    model=AE(model_config).to(device)
    
    print(summary(model,(args.batch_size,x_train_t.shape[1])))
    wandb.watch(model)
    
    #Xavier Init Weights
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr ,weight_decay=1e-5)
    
    model.zero_grad()
    model.train(True)
    
    for epoch in range(args.epoch):
        epoch_loss = []
        print(f"\nTraining Fold {k},Epoch {epoch+1} Ldim {args.l_dim}")
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

        train_loss=loss.item()
        pred_val,val_loss=get_model_preds(model,eval_dataloader)
        
        print("iter {}: Train: {:.5f}, Val: {:.5f}".format(batch_iter, train_loss,sum(val_loss)/len(val_loss)))

        #Save Hist
        train_hist['loss'].append(sum(epoch_loss)/len(epoch_loss))
        val_hist['loss'].append(sum(val_loss)/len(val_loss))
        
        # val_hist['f1'].append(best_f1)
        # wandb.log({"{}_{} F1".format(model_desc,args.batch_size):best_f1,"{}_{} AUC".format(model_desc,args.batch_size):auc_l2,"{}_{} Val Loss".format(model_desc,args.batch_size):sum(val_loss)/len(val_loss),"{}_{} Train Loss".format(model_desc,args.batch_size):train_loss})
        wandb.log({"{}_{} Val Loss".format(model_desc,args.batch_size):sum(val_loss)/len(val_loss),"{}_{} Train Loss".format(model_desc,args.batch_size):train_loss})
            
            
        #Cos Simil
        if args.dist_cos:
            print("\nCos")
            val_dist_cos=np.array([distance.cosine(x_train_v[i],pred_val[i]) for i in range(x_train_v.shape[0])])
            auc_cos,_,_=make_roc(val_dist_cos,y_train_v,ans_label=ATK)
            print("AUC Cos {:.5f}".format(auc_cos))
            val_hist['auc_cos'].append(auc_cos)
            wandb.log({"{}_{} AUC Cos".format(model_desc,args.batch_size):auc_cos})
            
            if auc_cos>model_best['auc_cos']:
                model_best['state_cos']=copy.deepcopy(model.state_dict())
                # model_best['epoch']=batch_iter
                model_best['epoch_cos']=epoch+1
                model_best['auc_cos']=auc_cos
                
        #L2
        else:
            print("\nL2")
            # print('Average Precision',average_precision_score(y_val, val_dist_l2, pos_label=ATK))
            val_dist_l2=np.mean(np.square(x_train_v-pred_val),axis=1)
            auc_l2,_,_=make_roc(val_dist_l2,y_train_v,ans_label=ATK)
            print("AUC L2 {:.5f}".format(auc_l2))
            val_hist['auc_l2'].append(auc_l2)
            wandb.log({"{}_{} AUC L2".format(model_desc,args.batch_size):auc_l2})
            
            #Save Model
            if auc_l2>model_best['auc_l2']:
                model_best['state_l2']=copy.deepcopy(model.state_dict())
                # model_best['epoch']=batch_iter
                model_best['epoch_l2']=epoch+1
                model_best['auc_l2']=auc_l2

    if args.dist_cos:
        model.load_state_dict(model_best['state_cos'])
        pred_val,val_loss=get_model_preds(model,eval_dataloader)
        val_dist=np.array([distance.cosine(x_train_v[i],pred_val[i]) for i in range(x_train_v.shape[0])])
        
    else:
        model.load_state_dict(model_best['state_l2'])
        pred_val,val_loss=get_model_preds(model,eval_dataloader)
        val_dist=np.mean(np.square(x_train_v-pred_val),axis=1)

    #Get Threshold
    comb_dist_mean,comb_dist_std,comb_best_z=get_threshold(val_dist,y_train_v,avg_type='binary')

    
    #Test
    val_df,_,y_val,_=preprocess(val,scaler=scaler)
    x_val=val_df.values
    x_val_cuda = torch.from_numpy(x_val).float().to(device)
    test_sampler = SequentialSampler(x_val_cuda)
    test_dataloader = DataLoader(x_val_cuda, sampler=test_sampler, batch_size=args.batch_size)


    if args.dist_cos:
        pred_val,_=get_model_preds(model,test_dataloader)
        test_dist=np.array([distance.cosine(x_val[i],pred_val[i]) for i in range(x_val.shape[0])])
    else:
        pred_val,_=get_model_preds(model,test_dataloader)
        test_dist=np.mean(np.square(x_val-pred_val),axis=1)
        
    test_dist_norm=(test_dist-comb_dist_mean)/comb_dist_std
    
    y_pred=np.zeros_like(y_val)
    y_pred[test_dist_norm>comb_best_z]=1
    
    test_auc,_,_=make_roc(test_dist,y_val,ans_label=ATK)
    prf(y_val,y_pred,ans_label=1,avg_type='binary')
    accuracy = accuracy_score(y_val,y_pred)
    precision, recall, f_score, support = precision_recall_fscore_support(y_val, y_pred, pos_label=1, average='binary')
    f_0_5=fbeta_score(y_val, y_pred, pos_label=1, average='binary', beta=0.5)
    f_2=fbeta_score(y_val, y_pred,pos_label=1, average='binary', beta=2)
    print("F1 {:.5f} F0.5 {:.5f} F2 {:.5f}\n".format(f_score,f_0_5,f_2))

    #Added (210201) - FPR, MCC
    test_fpr=fpr(y_val,y_pred)
    test_mcc=mcc(y_val,y_pred)
    print("FPR {:.5f}, MCC {:.5f}".format(test_fpr,test_mcc))
    
    #Best MCC Model
    if test_mcc>best_val_mcc:
        best_fold=k
        best_val_mcc=copy.deepcopy(test_mcc)
        if args.dist_cos:
            best_weights=copy.deepcopy(model_best['state_cos'])
        else:
            best_weights=copy.deepcopy(model_best['state_l2'])
        best_scaler=copy.deepcopy(scaler)
        best_mean=copy.deepcopy(comb_dist_mean)
        best_std=copy.deepcopy(comb_dist_std)
        best_z=copy.deepcopy(comb_best_z)
    with open(log_name, "a") as myfile:
        #L2
        myfile.write(f"{size_val},{args.num_layers},{args.l_dim},")
        myfile.write(f"{args.epoch},{args.batch_size},")
        myfile.write(f"{args.do},{args.bn},")
        #PERF
        if args.dist_cos:
            myfile.write("cos,{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},".format(test_auc,comb_best_z,accuracy,precision,recall,f_score))
        else:
            myfile.write("l2,{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},".format(test_auc,comb_best_z,accuracy,precision,recall,f_score))
        
        #Added (210201) - FPR, MCC
        myfile.write("{:.5f},{:.5f},".format(test_fpr,test_mcc))
        ##
        if args.dist_cos:
            myfile.write(f"total,{args.seed},{args.dec_rate},{k},{model_best['epoch_cos']}\n")
        else:
            myfile.write(f"total,{args.seed},{args.dec_rate},{k},{model_best['epoch_l2']}\n")
    
    del model
    
print("Best Fold {} mcc {:.4f}".format(best_fold,best_val_mcc))
if args.dist_cos:
    joblib.dump(best_scaler, "./weights_{}_{}_{}/iotid20_{}_{}_{}_{}_{}_cos.scaler".format(args.num_layers,model_dec_type,args.dec_rate,size_val,args.l_dim,args.epoch,args.batch_size,args.seed)) 
    pickle.dump(best_num_desc,open("./weights_{}_{}_{}/iotid20_{}_{}_{}_{}_{}_cos.numdesc".format(args.num_layers,model_dec_type,args.dec_rate,size_val,args.l_dim,args.epoch,args.batch_size,args.seed), 'wb'))
    # best_num_desc.to_csv(,header=None,index=None)
    with open("./weights_{}_{}_{}/iotid20_{}_{}_{}_{}_{}_cos.pt".format(args.num_layers,model_dec_type,args.dec_rate,size_val,args.l_dim,args.epoch,args.batch_size,args.seed), "wb") as f:
        torch.save(
            {
                "state_cos": best_weights,
                # "state_cos": model_best['state_cos'],
            },
            f,
        )
else:
    joblib.dump(best_scaler, "./weights_{}_{}_{}/iotid20_{}_{}_{}_{}_{}.scaler".format(args.num_layers,model_dec_type,args.dec_rate,size_val,args.l_dim,args.epoch,args.batch_size,args.seed)) 
    pickle.dump(best_num_desc,open("./weights_{}_{}_{}/iotid20_{}_{}_{}_{}_{}.numdesc".format(args.num_layers,model_dec_type,args.dec_rate,size_val,args.l_dim,args.epoch,args.batch_size,args.seed), 'wb'))
    # best_num_desc.to_csv(,header=None,index=None)
    with open("./weights_{}_{}_{}/iotid20_{}_{}_{}_{}_{}.pt".format(args.num_layers,model_dec_type,args.dec_rate,size_val,args.l_dim,args.epoch,args.batch_size,args.seed), "wb") as f:
        torch.save(
            {
                "state_l2": best_weights,
                # "state_cos": model_best['state_cos'],
            },
            f,
        )
    
print("\nTrain Complete")
if args.dist_cos:
    best_num_desc=pickle.load(open("./weights_{}_{}_{}/iotid20_{}_{}_{}_{}_{}_cos.numdesc".format(args.num_layers,model_dec_type,args.dec_rate,size_val,args.l_dim,args.epoch,args.batch_size,args.seed), 'rb'))
    best_scaler=joblib.load("./weights_{}_{}_{}/iotid20_{}_{}_{}_{}_{}_cos.scaler".format(args.num_layers,model_dec_type,args.dec_rate,size_val,args.l_dim,args.epoch,args.batch_size,args.seed))
    with open("./weights_{}_{}_{}/iotid20_{}_{}_{}_{}_{}_cos.th".format(args.num_layers,model_dec_type,args.dec_rate,size_val,args.l_dim,args.epoch,args.batch_size,args.seed),'w') as f:
        f.write(f'{str(best_mean)}\n{str(best_std)}\n{str(best_z)}')
else:
    best_num_desc=pickle.load(open("./weights_{}_{}_{}/iotid20_{}_{}_{}_{}_{}.numdesc".format(args.num_layers,model_dec_type,args.dec_rate,size_val,args.l_dim,args.epoch,args.batch_size,args.seed), 'rb'))
    best_scaler=joblib.load("./weights_{}_{}_{}/iotid20_{}_{}_{}_{}_{}.scaler".format(args.num_layers,model_dec_type,args.dec_rate,size_val,args.l_dim,args.epoch,args.batch_size,args.seed))
    with open("./weights_{}_{}_{}/iotid20_{}_{}_{}_{}_{}.th".format(args.num_layers,model_dec_type,args.dec_rate,size_val,args.l_dim,args.epoch,args.batch_size,args.seed),'w') as f:
        f.write(f'{str(best_mean)}\n{str(best_std)}\n{str(best_z)}')
  
if args.dec:
    layers=[]
    for i in range(0,args.num_layers):
        layers.append(int(x_train.shape[1]*args.dec_rate**(i+1)))
    layers.append(args.l_dim) 
    
    #Set Val for logs
    # size_val=args.dec_rate
    size_val="In"
    model_dec_type="dec"
    
else:
    #Decrease from Biggest - default 0.5
    layers=[]
    for i in range(0,args.num_layers):
        layers.append(int(args.size*args.dec_rate**(i)))
    layers.append(args.l_dim)
    size_val=args.size
    model_dec_type="fixed"
    
model_config={
    'd_dim':x_train_t.shape[1],
    'layers':layers
} 
model=AE(model_config).to(device)
model.load_state_dict(best_weights)

test=pd.read_csv(data_dir+'/test.csv')
test_df,_,y_test,_=preprocess(test,scaler=best_scaler)

x_test=test_df.values
x_test_cuda = torch.from_numpy(x_test).float().to(device)
eval_sampler = SequentialSampler(x_test_cuda)
eval_dataloader = DataLoader(x_test_cuda, sampler=eval_sampler, batch_size=args.batch_size)

pred_test,_=get_model_preds(model,eval_dataloader)
if args.dist_cos:
    test_dist=np.array([distance.cosine(x_test[i],pred_test[i]) for i in range(x_test.shape[0])])
else:
    test_dist=np.mean(np.square(x_test-pred_test),axis=1)
test_dist_norm=(test_dist-best_mean)/best_std

y_pred=np.zeros_like(y_test)
y_pred[test_dist_norm>best_z]=1
# y_pred[test_dist_norm>comb_best_z]=1

test_auc,_,_=make_roc(test_dist,y_test,ans_label=ATK)
prf(y_test,y_pred,ans_label=1,avg_type='binary')
accuracy = accuracy_score(y_test,y_pred)
precision, recall, f_score, support = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
f_0_5=fbeta_score(y_test, y_pred, pos_label=1, average='binary', beta=0.5)
f_2=fbeta_score(y_test, y_pred,pos_label=1, average='binary', beta=2)
print("F1 {:.5f} F0.5 {:.5f} F2 {:.5f}\n".format(f_score,f_0_5,f_2))

#Added (210201) - FPR, MCC
test_fpr=fpr(y_test,y_pred)
test_mcc=mcc(y_test,y_pred)
print("FPR {:.5f}, MCC {:.5f}".format(test_fpr,test_mcc))

if args.dist_cos:
    log_name=f"perf_results/iotid20_{args.size}_{args.num_layers}_cos.csv"
else:
    log_name=f"perf_results/iotid20_{args.size}_{args.num_layers}.csv"

if not os.path.isfile(log_name):
     with open(log_name, "a") as myfile:
         myfile.write("size,num_layers,l_dim,epoch,batch,dropout,bn,dist,"+"auc,z,acc,p,r,f,"+"fpr,mcc,"+"label,seed,dec_rate\n")
         
with open(log_name, "a") as myfile:
    #L2
    myfile.write(f"{size_val},{args.num_layers},{args.l_dim},")
    myfile.write(f"{args.epoch},{args.batch_size},")
    myfile.write(f"{args.do},{args.bn},")
    #PERF
    # myfile.write("l2,{:.5f},{},".format(model_best['auc_l2'],model_best['epoch_l2']))
    if args.dist_cos:
        myfile.write("cos,{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},".format(test_auc,comb_best_z,accuracy,precision,recall,f_score))
    else:
        myfile.write("l2,{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},".format(test_auc,comb_best_z,accuracy,precision,recall,f_score))
    #Added (210201) - FPR, MCC
    myfile.write("{:.5f},{:.5f},".format(test_fpr,test_mcc))
    ##
    myfile.write(f"total,{args.seed},{args.dec_rate}\n")
exit()


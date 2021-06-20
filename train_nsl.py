from __future__ import absolute_import, print_function
import argparse
import os

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from model.ae import AE

#Load Utils
from utils.data_utils import *
from utils.perf_utils import *
from utils.plot_utils import *
from utils.train_utils import *

#Preprocess Script for data
from data.preprocess_nsl import *

import copy
import numpy as np

#Wandb Logging
import wandb
from tqdm import tqdm

#Saving
import pickle

#Label Info
ATK=1
SAFE=0

def load_data():
    data_dir='data/nsl_kdd/split'
    train=pd.read_csv(data_dir+'/train.csv',header=None)
    val=pd.read_csv(data_dir+'/val.csv',header=None)

    service = open(data_dir+'/service.txt', 'r')
    serviceData = service.read().split('\n')
    service.close()

    flag = open(data_dir+'/flag.txt', 'r')
    flagData = flag.read().split('\n')
    flag.close()

    #Preprocess
    train_df,y_train,y_train_types,scaler,num_desc=preprocess(train,serviceData,flagData)  
    x_train=train_df.values
    x_train,y_train=select_label_data(x_train,y_train,select_label=SAFE)
    print("Filter Train: Normal:{}, Atk:{}".format(x_train[y_train==0].shape[0],x_train[y_train==1].shape[0]))

    val_df,y_val,y_val_types,_,_=preprocess(val,serviceData,flagData,is_train=False,scaler=scaler, num_desc=num_desc)
    x_val=val_df.values
    return x_train,y_train,x_val,y_val

#Initialize Model
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
     
def get_model(args,data_dim):
    layers=[]
    for i in range(0,args.num_layers):
        #Halved for each succeeding layer size
        layers.append(int(args.max_hid_size*0.5**(i)))
    layers.append(args.l_dim)
    
    model_config={
        'd_dim':data_dim,
        'layers':layers
    } 
    model=AE(model_config)
    return model
    
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

def get_threshold(dist,y):
    #Calculate Best Threshold
    
    #Standardize Distance
    dist_safe=dist[y==0]
    mean=np.mean(dist_safe)
    std=np.std(dist_safe)
    
    dist_standardized=(dist-mean)/std
    
    best_score=0
    best_th=0
    
    #Z Score Search Range
    for z in range(1,801):
        th_cand=-4+0.01*z

        #Prediction
        y_pred=np.zeros_like(y)
        y_pred[dist_standardized>th_cand]=1
        
        #Score: MCC
        score=mcc(y,y_pred)
        
        if best_score<score:
            best_score=score
            best_th=th_cand
            
    print("Best Z-threshold {:.3f} with score {:.5f}".format(best_th,best_score))
    return mean,std,best_th

def train(args,device,weight_dir):
    #Load Data
    x_train,y_train,x_val,y_val=load_data()
    
    #Get Model
    model=get_model(args,x_train.shape[1])
    
    #Log Validation Performance
    train_log_name="train_log/nsl.txt"

    if not os.path.isfile(train_log_name):
        with open(train_log_name, "a") as myfile:
            #To Record
            myfile.write("max_hid_size,num_layers,l_dim,epoch,batch_size,z_th,auc,acc,p,r,f,fpr,mcc,seed,best_epoch\n")
         
    model_desc='NSL {}-{}_{}'.format(args.num_layers,args.max_hid_size,args.l_dim)

    #Dataloader
    x_train = torch.from_numpy(x_train).float().to(device)
    data_sampler = RandomSampler(x_train)
    data_loader = DataLoader(x_train, sampler=data_sampler, batch_size=args.batch_size)

    x_val_cuda = torch.from_numpy(x_val).float().to(device)
    val_sampler = SequentialSampler(x_val_cuda)
    val_dataloader = DataLoader(x_val_cuda, sampler=val_sampler)

    loss_hist={
        'train':[],
        'val':[],
        'val_auc':[]
    }
    best_perf={
        'best_auc':0,
        'model_weights':None,
        'best_epoch':0
    }

    #Load Model
    model=get_model(args,x_train.shape[1])
    model=model.to(device)
    wandb.watch(model)

    #Xavier Init Weights
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr ,weight_decay=1e-5)

    model.zero_grad()
    model.train(True)

    #Train Model
    for epoch in range(args.epoch):
        epoch_loss = []
        print(f"\nTraining Epoch {epoch+1} Ldim {args.l_dim} Seed {args.seed}")
        model.train()
        # for step, batch in enumerate(data_loader):
        for step, batch in enumerate(tqdm(data_loader)):
            target = batch.type(torch.float32)

            outputs = model(target)
            loss = model.compute_loss(outputs, target)

            loss.backward()
            optimizer.step()
            model.zero_grad()

            epoch_loss.append(loss.item())

        pred_val,val_loss=get_model_preds(model,val_dataloader)

        #Save Hist
        train_loss_val=sum(epoch_loss)/len(epoch_loss)
        val_loss_val=sum(val_loss)/len(val_loss)
        loss_hist['train'].append(train_loss_val)
        loss_hist['val'].append(val_loss_val)
        
        print("Epoch {}: Train: {:.5f}, Val: {:.5f}".format(epoch+1, train_loss_val,val_loss_val))
        wandb.log({f"{model_desc} Train Loss":train_loss_val,f"{model_desc} Val Loss":val_loss_val})

        #Calculate AUC
        val_dist_l2=np.mean(np.square(x_val-pred_val),axis=1)
        auc_l2,_,_=make_roc(val_dist_l2,y_val,ans_label=ATK)
        print("AUC {:.5f}".format(auc_l2))
        loss_hist['val_auc'].append(auc_l2)
        wandb.log({"{} AUC L2".format(model_desc):auc_l2})
        
        #Save Model if Better
        if auc_l2>best_perf['best_auc']:
            best_perf['model_weights']=copy.deepcopy(model.state_dict())
            best_perf['best_epoch']=epoch+1
            best_perf['best_auc']=auc_l2
    
    print("Model Train Complete")
    
    #Get Threshold
    model.load_state_dict(best_perf['model_weights'])
    pred_val,val_loss=get_model_preds(model,val_dataloader)
    val_dist=np.mean(np.square(x_val-pred_val),axis=1)

    val_safe_mean,val_safe_stddev,val_best_z_th=get_threshold(val_dist,y_val)
    
    #Standardized Distance
    val_dist_standardized=(val_dist-val_safe_mean)/val_safe_stddev
    
    y_pred=np.zeros_like(y_val)
    y_pred[val_dist_standardized>val_best_z_th]=1
    
    #Performance
    val_auc,_,_=make_roc(val_dist,y_val,ans_label=ATK)
    accuracy,precision,recall,f_score = aprf(y_val,y_pred,pos_label=ATK, average='binary')
    val_fpr=fpr(y_val,y_pred)
    val_mcc=mcc(y_val,y_pred)

    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision, recall, f_score))
    print("FPR {:.5f}, MCC {:.5f}".format(val_fpr,val_mcc))
    
    #Log Results
    with open(train_log_name, "a") as myfile:
        #L2
        myfile.write(f"{args.max_hid_size},{args.num_layers},{args.l_dim},")
        myfile.write(f"{args.epoch},{args.batch_size},")
        #PERF
        myfile.write("{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},".format(val_best_z_th,val_auc,accuracy,precision,recall,f_score))
        myfile.write("{:.5f},{:.5f},".format(val_fpr,val_mcc))
        myfile.write(f"{args.seed},{best_perf['best_epoch']}\n")
    
    pickle.dump(loss_hist, open(f"train_log/nsl_{args.num_layers}_{args.max_hid_size}_{args.l_dim}_{args.epoch}_{args.batch_size}_{args.seed}.pkl", "wb"))
    
    #Save Files
    with open(weight_dir+"/nsl_{}_{}_{}_{}.pt".format(args.l_dim,args.epoch,args.batch_size,args.seed), "wb") as f:
        torch.save(
            {
                "best_weights": best_perf['model_weights'],
            },
            f,
        )
    with open(weight_dir+"/nsl_{}_{}_{}_{}.th".format(args.l_dim,args.epoch,args.batch_size,args.seed),'w') as f:
        f.write(f'{str(val_safe_mean)}\n{str(val_safe_stddev)}\n{str(val_best_z_th)}')


if __name__ == "__main__":
    # Argument Setting
    parser = argparse.ArgumentParser()

    #Random Seed (For Reproducibility)
    parser.add_argument("--seed", default=42, type=int,
                        help="random seed for reproductability")

    #Model Config
    parser.add_argument("--l_dim", default=10, type=int,
                        help="Latent dimension size")
    parser.add_argument("--num_layers", default=2, type=int,
                        help="number of hidden layers for each encoder,decoder")
    parser.add_argument("--max_hid_size", default=64, type=int,
                        help="Biggest Hid Size in Encoder,decoder")


    #Training Params
    parser.add_argument("--epoch", default=10, type=int,
                        help="number of epochs")
    parser.add_argument("--batch_size", default=8192, type=int,
                        help="batch size")
    parser.add_argument("--lr", default=1e-4, type=float,
                        help="learning rate")


    args = parser.parse_args()

    wandb.init(project='svcc-nslkdd')

    # Fix seed
    set_seed(args.seed)
    device = torch.device('cuda:0')
    
    #Train Log Folder
    if not os.path.exists("train_log"):
        os.makedirs("train_log")
        
    #Weight Folder
    weight_dir=f"weights/nsl_{args.num_layers}_{args.max_hid_size}"
    if not os.path.exists("weights"):
        os.makedirs("weights")
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
        
    #Train Model
    train(args,device,weight_dir)
    

from __future__ import absolute_import, print_function
import argparse
import os

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from model.ae import AE
from utils.data_utils import *
from utils.perf_utils import *
from utils.plot_utils import *
from utils.train_utils import *

#Preprocess Script
from data.preprocess_nsl import *

import numpy as np

import joblib
import pickle

import seaborn as sns
from scipy.interpolate import interp1d

ATK=1
SAFE=0

def load_data():
    data_dir='data/nsl_kdd/split'
    train=pd.read_csv(data_dir+'/train.csv',header=None)
    test=pd.read_csv(data_dir+'/test.csv',header=None)

    service = open(data_dir+'/service.txt', 'r')
    serviceData = service.read().split('\n')
    service.close()

    flag = open(data_dir+'/flag.txt', 'r')
    flagData = flag.read().split('\n')
    flag.close()

    #Preprocess
    _,_,_,scaler,num_desc=preprocess(train,serviceData,flagData)  

    test=pd.read_csv(data_dir+'/test.csv',header=None)
    test_df,y_test,y_test_types,_,_=preprocess(test,serviceData,flagData,is_train=False,scaler=scaler, num_desc=num_desc)
    x_test=test_df.values
    
    return x_test,y_test,y_test_types

def get_model(args,data_dim,l_dim):
    layers=[]
    for i in range(0,args.num_layers):
        #Halved for each succeeding layer size
        layers.append(int(args.max_hid_size*0.5**(i)))
    layers.append(l_dim)
    
    model_config={
        'd_dim':data_dim,
        'layers':layers
    } 
    model=AE(model_config)
    return model

def get_model_preds(model,loader):
    model.eval()
    with torch.no_grad():
        pred=[]
        # loss=[]
        for batch in loader:
            target = batch.type(torch.float32)

            outputs = model(target)
            # batch_error = model.compute_loss(outputs, target)

            pred.append(outputs['output'].cpu().detach().numpy())
            # loss.append(batch_error.item())

        pred=np.concatenate(pred)
    return pred

def eval_seed(args,device,weight_dir,seed,l_dim):
    set_seed(seed)
    x_test,y_test,y_test_types=load_data()
    
    #Get Model
    model=get_model(args,x_test.shape[1],l_dim)
    model.to(device)
    
    with open(weight_dir+"/nsl_{}_{}_{}_{}.pt".format(l_dim,args.epoch,args.batch_size,seed), "rb") as f:
        best_model = torch.load(f)
        
    model.load_state_dict(best_model["best_weights"])
    
    #Load mean,stddev,th
    threshold_file = open(weight_dir+"/nsl_{}_{}_{}_{}.th".format(l_dim,args.epoch,args.batch_size,seed), 'r')
    threshold = threshold_file.read().split('\n')
    mean=float(threshold[0])
    stddev=float(threshold[1])
    z_th=float(threshold[2])
    
    x_test_cuda = torch.from_numpy(x_test).float().to(device)
    test_sampler = SequentialSampler(x_test_cuda)
    test_dataloader = DataLoader(x_test_cuda, sampler=test_sampler)
    
    pred_test=get_model_preds(model,test_dataloader)

    test_dist=np.mean(np.square(x_test-pred_test),axis=1)
    test_dist_standardized=(test_dist-mean)/stddev

    y_pred=np.zeros_like(y_test)
    y_pred[test_dist_standardized>z_th]=1
    
    cats=["DoS","U2R","R2L","Probe"]
    # sub_cats={
    #     'DoS':["neptune","smurf","pod","teardrop","land","back","apache2","udpstorm","processtable","mailbomb"],
    #     "U2R":["buffer_overflow","loadmodule","perl","rootkit","spy","xterm","ps","httptunnel","sqlattack","worm","snmpguess"],
    #     "R2L":["guess_passwd","ftp_write","imap","phf","multihop","warezmaster","warezclient","snmpgetattack","named","xlock","xsnoop","sendmail"],
    #     "Probe":["portsweep","ipsweep","nmap","satan","saint","mscan"]
    # }
    sub_cats={
        'DoS':["neptune","smurf","pod","teardrop","land","back","apache2","udpstorm","processtable","mailbomb"],
        "U2R":["buffer_overflow","loadmodule","perl","rootkit","xterm","ps","httptunnel","sqlattack"],
        "R2L":["guess_passwd","ftp_write","imap","phf","multihop","warezmaster","warezclient","snmpgetattack","named","xlock","xsnoop","sendmail","spy","worm","snmpguess"],
        "Probe":["portsweep","ipsweep","nmap","satan","saint","mscan"]
    }

    # cat_accs=[recall]
    dist_safe=test_dist_standardized[y_test==0]
    dists=[dist_safe]
    for cat in cats:
        print(cat)
        dist_cat=[]
        y_cat=[]

        for sub_cat in sub_cats[cat]:
            # print(sub_cat)
            dist_subcat=test_dist_standardized[y_test_types==sub_cat]
            y_subcat=y_test[y_test_types==sub_cat]
            
            dist_cat.append(dist_subcat)
            y_cat.append(y_subcat)

        dist_cat=np.concatenate(dist_cat,axis=0)
        y_cat=np.concatenate(y_cat,axis=0)

        print(dist_cat.shape)
        # print(accuracy_score(y_cat,pred_cat))
        # cat_accs.append(accuracy_score(y_cat,pred_cat))
        
        dists.append(dist_cat)
        
    return dists,z_th

def eval_ldim(args,device,weight_dir,l_dim):
    seeds=range(10,10*(args.num_runs+1),10)
    seed_dists=[]
    seed_ths=[]
    for seed in seeds:
        seed_dist,seed_th=eval_seed(args,device,weight_dir,seed,l_dim)
        if seed_dist is not None:
            seed_dists.append(seed_dist)
            seed_ths.append(seed_th)

    seed_dists=np.array(seed_dists)
    avg_dist=np.mean(seed_dists,axis=0)
    
    avg_th=np.mean(np.array(seed_ths))
    return avg_dist,avg_th

def plot_dist(dist,th,l_dim):
    config=f"{args.num_layers}_{args.max_hid_size}"

    fig=plt.figure(figsize=(10, 6))
    axes=plt.axes()

    cats=["R2L"]
    axes.set_xlim([-2,6.5])
    axes.set_ylim([0,2])

    #Plot Recon Distance of Safe Samples
    dist_safe=dist[0]
    num_bins=int((np.max(dist_safe)-np.min(dist_safe))/0.2)
    sns.distplot(dist_safe,hist=True,bins=num_bins,color="navy",label="Normal")

    ##Plot Recon Distance of Attack Samples (R2L)
    dist_cat=dist[3]
    num_bins=int((np.max(dist_cat)-np.min(dist_cat))/0.2)
    ax=sns.distplot(dist_cat,hist=True,bins=num_bins,color="orangered",label="R2L")

    axes.axvline(x=th,ls='--',c='black',label="Threshold")
    axes.set_xlabel('Z-Score',fontsize=16)
    axes.set_ylabel('Density',fontsize=14)

    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        # label.set_fontname('Arial')
        label.set_fontsize(14)

    axes.legend(prop={'size': 16})
    fig.tight_layout()
    fig.savefig(f'./dist_plot/nsl_r2l_{config}_{l_dim}.png')

if __name__ == "__main__":
    # Argument Setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", default=20, type=int,
                        help="number of runs with different seeds")

    parser.add_argument("--l_dim", default=10, type=int,
                        help="Latent dimension size")
    parser.add_argument("--num_layers", default=2, type=int,
                        help="number of hidden layers for each encoder,decoder")
    parser.add_argument("--max_hid_size", default=64, type=int,
                        help="Biggest Hid Size in Encoder,decoder")

    #Train Params
    parser.add_argument("--epoch", default=10, type=int,
                        help="number of epochs")
    parser.add_argument("--batch_size", default=8192, type=int,
                        help="batch size")

    args = parser.parse_args()

    device = torch.device('cuda:0')
    weight_dir=f"weights/nsl_{args.num_layers}_{args.max_hid_size}"
    
    
    ldim_avg_dist,ldim_avg_th=eval_ldim(args,device,weight_dir,args.l_dim)
    
    plot_dist(ldim_avg_dist,ldim_avg_th,args.l_dim)
    
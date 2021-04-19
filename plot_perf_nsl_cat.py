from __future__ import absolute_import, print_function
import torch

import argparse
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from model.ae import AE

from utils.data_utils import *
from utils.perf_utils import *
from utils.plot_utils import *

#Preprocess Script
from data.preprocess_nsl import *

from sklearn.metrics import average_precision_score
from scipy.spatial import distance

from sklearn.model_selection import KFold

import copy
import numpy as np

import wandb
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

def get_model(args,l_dim,data_dim):
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
        for batch in loader:
            target = batch.type(torch.float32)
            outputs = model(target)
            pred.append(outputs['output'].cpu().detach().numpy())
        pred=np.concatenate(pred)
    return pred


def eval_perf_seed(args,l_dim,seed,x_test,y_test,y_test_types):
    # Evaluate Model with this Latent Size & Seed
    # Fix seed
    set_seed(seed)
    device = torch.device('cuda:0')
    
    model=get_model(args,l_dim,x_test.shape[1])
    model.to(device)
    
    weight_dir=f"weights/nsl_{args.num_layers}_{args.max_hid_size}"
    
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
    sub_cats={
        'DoS':["neptune","smurf","pod","teardrop","land","back","apache2","udpstorm","processtable","mailbomb"],
        "U2R":["buffer_overflow","loadmodule","perl","rootkit","spy","xterm","ps","httptunnel","sqlattack","worm","snmpguess"],
        "R2L":["guess_passwd","ftp_write","imap","phf","multihop","warezmaster","warezclient","snmpgetattack","named","xlock","xsnoop","sendmail"],
        "Probe":["portsweep","ipsweep","nmap","satan","saint","mscan"]
    }
    
    precision, recall, f_score, support = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
    # TPR
    cat_tprs=[recall]
    for cat in cats:
        print(cat)
        pred_cat=[]
        y_cat=[]

        for sub_cat in sub_cats[cat]:
            # print(sub_cat)
            pred_subcat=y_pred[y_test_types==sub_cat]
            y_subcat=y_test[y_test_types==sub_cat]
            
            pred_cat.append(pred_subcat)
            y_cat.append(y_subcat)

        pred_cat=np.concatenate(pred_cat,axis=0)
        y_cat=np.concatenate(y_cat,axis=0)

        print(pred_cat.shape)
        print(accuracy_score(y_cat,pred_cat))
        cat_tprs.append(accuracy_score(y_cat,pred_cat))
        
    return cat_tprs

def eval_perf(args):
    x_test,y_test,y_test_types=load_data()
    
    l_dims=range(1,int(args.max_hid_size/(2**(args.num_layers-1))))
    seeds=range(10,210,10)
    
    dim_tprs=[]
    for l_dim in l_dims:
        # dim_acc=eval_seed(args,l_dim)
        
        seed_tprs=[]
        for seed in seeds:
            print(f"Evaluating L_Dim:{l_dim}, Seed:{seed}")
            seed_tpr=eval_perf_seed(args,l_dim,seed,x_test,y_test,y_test_types)
            if seed_tpr is not None:
                seed_tprs.append(seed_tpr)
        seed_tprs=np.array(seed_tprs)
        avg_perf=np.mean(seed_tprs,axis=0)
        
        dim_tprs.append(avg_perf)
    
    return np.array(dim_tprs)
    # seeds=range(10,210,10)
    # # seeds=range(10,20,10)
    # seed_tprs=[]
    # for seed in seeds:
    #     seed_tpr=eval_dim(args,l_dim,seed,x_test,y_test,y_test_types)
    #     if seed_tpr is not None:
    #         seed_tprs.append(seed_tpr)
    # seed_tprs=np.array(seed_tprs)
    # avg_perf=np.mean(seed_tprs,axis=0)
    # return avg_perf

def plot_tpr_cat(args):
    # dim_tprs=eval_perf(args)
    #Save Results
    # np.save(f'./perf_results/nsl_cat_tpr_{args.num_layers}_{args.max_hid_size}.npy',dim_tprs)
    
    #Load from File
    dim_tprs=np.load(f'./perf_results/nsl_cat_tpr_{args.num_layers}_{args.max_hid_size}.npy')
    
    cats=["DoS","U2R","R2L","Probe"]
    cat_idxs=range(1,5) #idx 0 is all attack tpr (recall)

    l_dims=range(1,int(args.max_hid_size/(2**(args.num_layers-1))))
    
    fig=plt.figure(figsize=(10,6))
    # fig=plt.figure(figsize=(12, 7))
    mcc_plot=fig.add_subplot(1,1,1)
    colors=['b','g','r','darkorange','m']
    colors=['b','g','orangered','red','m']
    markers=['o','^','s','P','D']

    for i in range(len(cats)):
        # print(cats[i]," Acc:{:.5f}".format(avg_perf[i]))
        cat_acc=[dim_tprs[dim-1][cat_idxs[i]] for dim in l_dims]
        print(cats[i])
        print(cat_acc)

        col=colors[i]
        marker=markers[i]
        
        xnew = np.linspace(1, len(l_dims), num=101, endpoint=True)
        f2 = interp1d(l_dims, cat_acc, kind='cubic')
        mcc_plot.plot(xnew,f2(xnew),c=col)
        mcc_plot.scatter(l_dims,cat_acc,c=col,marker=marker,s=80,label=cats[i])
        
    mcc_plot.set_xlabel('Latent Size',fontsize=20)
    mcc_plot.set_ylabel('TPR',fontsize=20)

    axes=plt.axes()
    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        # label.set_fontname('Arial')
        label.set_fontsize(20)

    # mcc_plot.legend(loc="lower right",prop={'size': 14})
    # mcc_plot.legend(loc="center left",prop={'size': 16})
    mcc_plot.legend(loc='best',prop={'size': 15})
    fig.tight_layout()
    # fig.savefig(f'./perf_plot/nsl_cat_{args.size}_{args.num_layers}_avg_l2.png')

    

# l_dims=range(1,int(args.size/(2**(args.num_layers-1))))

# print(args.num_layers)
# # exit()

# dim_accs=[]
# for l_dim in l_dims:
#     # dim_acc=eval_dim(args,l_dim)
#     dim_acc=eval_seed(args,l_dim)
#     dim_accs.append(dim_acc)


# dim_accs=np.array(dim_accs)
# np.save(f'nsl_cat_{args.size}_{args.num_layers}.npy',dim_accs)
# # print(np.mean(seed_accs,axis=0).shape)
# # avg_perf=np.mean(seed_accs,axis=0)
# # cats=["All","DoS","U2R","R2L","Probe"]
# # cats=["DoS","Probe"]
# # cat_idxs=[1,4]

# # cats=["U2R","R2L"]
# # cat_idxs=[2,3]

# cats=["DoS","U2R","R2L","Probe"]
# cat_idxs=range(1,5) #0 is all attacks

# fig=plt.figure(figsize=(10,6))
# # fig=plt.figure(figsize=(12, 7))
# mcc_plot=fig.add_subplot(1,1,1)
# colors=['b','g','r','darkorange','m']
# colors=['b','g','orangered','red','m']
# markers=['o','^','s','P','D']

# for i in range(len(cats)):
#     # print(cats[i]," Acc:{:.5f}".format(avg_perf[i]))
#     cat_acc=[dim_accs[dim-1][cat_idxs[i]] for dim in l_dims]
#     print(cats[i])
#     print(cat_acc)

#     col=colors[i]
#     marker=markers[i]
    
#     xnew = np.linspace(1, len(l_dims), num=101, endpoint=True)
#     f2 = interp1d(l_dims, cat_acc, kind='cubic')
#     mcc_plot.plot(xnew,f2(xnew),c=col)
#     mcc_plot.scatter(l_dims,cat_acc,c=col,marker=marker,s=80,label=cats[i])
# mcc_plot.set_xlabel('Latent Size',fontsize=20)
# mcc_plot.set_ylabel('TPR',fontsize=20)

# axes=plt.axes()
# for label in (axes.get_xticklabels() + axes.get_yticklabels()):
#     # label.set_fontname('Arial')
#     label.set_fontsize(20)

# # mcc_plot.legend(loc="lower right",prop={'size': 14})
# # mcc_plot.legend(loc="center left",prop={'size': 16})
# mcc_plot.legend(loc='best',prop={'size': 15})
# fig.tight_layout()
# fig.savefig(f'./perf_plot/nsl_cat_{args.size}_{args.num_layers}_avg_l2.png')

# exit()

if __name__=="__main__":
    # Argument Setting
    parser = argparse.ArgumentParser()

    #Model Config
    # parser.add_argument("--l_dim", default=10, type=int,
    #                     help="Latent dimension size")
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
    
    #Plot MCC Latent Plot
    plot_tpr_cat(args)
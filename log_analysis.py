import numpy as np
import pandas as pd

import argparse

from utils.plot_utils import *

def train_log_analysis(args):
    # train_log=pd.read_csv('train_auc.txt')
    if args.dec:
        train_log=pd.read_csv(f'perf_results/{args.data}_train_log_dec.txt')
    else:
        train_log=pd.read_csv(f'perf_results/{args.data}_train_log_fixed.txt')

    print(train_log.shape)
    # size,num_layers,l_dim,epoch,batch,dropout,bn,dist,best_auc,best_ep,seed,dec_rate
    log=train_log.loc[
        (train_log["size"]==args.size)&
        (train_log["dec_rate"]==args.dec_rate)&
        (train_log["num_layers"]==args.num_layers)&
        # (train_log["l_dim"]==args.l_dim)&
        (train_log["epoch"]==args.epoch)&
        (train_log["batch"]==args.batch_size)&
        (train_log["dropout"]==args.do)&
        (train_log["bn"]==args.bn)
    ]
    # print(log["seed"].mean(axis=0))
    dist="l2"
    print(log[log["dist"]==dist])
    print(log[log['dist']==dist].groupby(['l_dim']).mean())
    log=log[log["dist"]==dist]
    for l_dim in log.l_dim.unique():
        select=log[log["l_dim"]==l_dim].nlargest(3, 'best_auc')
        print(select)
        print(select.seed.values)
        print(select.groupby(['l_dim']).mean())


def test_log_analysis(args):
    # if args.bal:
    #     log=pd.read_csv(f'perf_results/{args.data}_test_perf_bal.txt')
    # else:
    #     log=pd.read_csv(f'perf_results/{args.data}_test_perf_dec.txt')
    
    if args.dec:
        log=pd.read_csv(f'perf_results/{args.data}_test_perf_dec.txt')
    else:
        log=pd.read_csv(f'perf_results/{args.data}_test_perf_fixed.txt')
        
    print(log.shape)
    #size,num_layers,l_dim,epoch,batch,dropout,bn,dist,auc,z,acc,p,r,f,/fpr,mcc/label,seed
    log_select=log.loc[
        (log["size"]==args.size)&
        (log["dec_rate"]==args.dec_rate)&
        (log["num_layers"]==args.num_layers)&
        # (log["l_dim"]==args.l_dim)&
        # (log["l_dim"]==10)&
        (log["epoch"]==args.epoch)&
        (log["batch"]==args.batch_size)&
        (log["dropout"]==args.do)&
        (log["label"]=="total")&
        (log["bn"]==args.bn)
    ]
    #seed condition
    # seed_cond=(log["seed"]<=50)
    # log_select=log_select[seed_cond]
    # print(log["seed"].mean(axis=0))
    # dist="l2"
    dist=args.dist
    
    print(log_select[log_select["dist"]==dist])
    print(log_select[log_select["dist"]==dist].shape)
    if args.bal:
        print(f"Balanced: dist {dist}")
    else:
        print(f"Full: dist {dist}")
    log_select=log_select[log_select['dist']==dist]
    # print(log_select[log_select["dist"]==dist][log_select["l_dim"]==1])
    print("\nAvg\n",log_select.groupby(['l_dim']).mean())
    print("\nVar\n",log_select.groupby(['l_dim']).var())
    
    print("\nMax\n",log_select.groupby(['l_dim']).max())
    #n largest
    for l_dim in log_select.l_dim.unique():
        select=log_select[log_select["l_dim"]==l_dim].nlargest(3, 'acc')
        print(select.groupby(['l_dim']).mean())
    exit()
    # Group By Latent dim 
    grouped=log_select.groupby(['l_dim'],as_index=False)
    
    # Remove outlier using Interquartile Range (IQR)
    metric="auc"
    high_q=grouped[metric].quantile(0.75)
    low_q=grouped[metric].quantile(0.25)
    print(high_q)
    for l_dim in high_q["l_dim"]:
        # print(high_q[high_q["l_dim"]==l_dim]["mcc"])
        high_bound=high_q[high_q["l_dim"]==l_dim][metric].values[0]
        low_bound=low_q[low_q["l_dim"]==l_dim][metric].values[0]
        iqr=high_bound-low_bound
        print(high_bound,low_bound)
        # print([log_select[log_select["l_dim"]==l_dim]["mcc"]<=(high_bound+1.5*iqr)])
        # q1 -1.5*iqr <= x <= q3 + 1.5*iqr
        high_cond=(log_select[log_select["l_dim"]==l_dim][metric]<=(high_bound+1.5*iqr))
        low_cond=(log_select[log_select["l_dim"]==l_dim][metric]>=(low_bound-1.5*iqr))
        print(log_select[log_select["l_dim"]==l_dim][high_cond & low_cond])
        print(log_select[log_select["l_dim"]==l_dim][high_cond & low_cond].groupby(['l_dim']).mean())
        # print(l_dim,grouped[grouped["l_dim"]==l_dim]["mcc"]<high_q[high_q['l_dim']==l_dim])

    # print("\nMax\n",log_select[log_select['dist']==dist].groupby(['l_dim']).max())
    # print("\nMin\n",log_select[log_select['dist']==dist].groupby(['l_dim']).min())

if __name__=="__main__":
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
    parser.add_argument('--bal', action='store_true')
    parser.add_argument("--dist", default="l2", type=str,
                    help="reconstruction dist")
    parser.add_argument("--data", default="cic", type=str,
                    help="data set")
    parser.add_argument("--dec_rate", default="0.5", type=float,
                    help="Decreae_rate")
    parser.add_argument('--dec',action='store_true', help='decreasing model')
    args = parser.parse_args()
    # train_log_analysis(args)
    test_log_analysis(args)

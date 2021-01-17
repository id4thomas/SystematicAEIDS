import numpy as np
import pandas as pd

import argparse

from utils.plot_utils import *

def train_log_analysis(args):
    train_log=pd.read_csv('train_auc.txt')
    print(train_log.shape)
    #size,num_layers,l_dim,epoch,batch,dropout,bn,dist,best_auc,best_ep,seed
    log=train_log.loc[
        (train_log["size"]==args.size)&
        (train_log["num_layers"]==args.num_layers)&
        # (train_log["l_dim"]==args.l_dim)&
        (train_log["epoch"]==args.epoch)&
        (train_log["batch"]==args.batch_size)&
        (train_log["dropout"]==args.do)&
        (train_log["bn"]==args.bn)
    ]
    # print(log["seed"].mean(axis=0))
    dist="cos"
    print(log[log["dist"]==dist])
    print(log[log['dist']==dist].groupby(['l_dim']).mean())

def test_log_analysis(args):
    if args.bal:
        log=pd.read_csv(f'perf_results/{args.data}_test_perf_bal.txt')
    else:
        log=pd.read_csv(f'perf_results/{args.data}_test_perf.txt')
    print(log.shape)
    #size,num_layers,l_dim,epoch,batch,dropout,bn,dist,auc,z,acc,p,r,f,label,seed
    log_select=log.loc[
        (log["size"]==args.size)&
        (log["num_layers"]==args.num_layers)&
        # (log["l_dim"]==args.l_dim)&
        (log["epoch"]==args.epoch)&
        (log["batch"]==args.batch_size)&
        (log["dropout"]==args.do)&
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
    # print(log_select[log_select["dist"]==dist][log_select["l_dim"]==1])
    print(log_select[log_select['dist']==dist].groupby(['l_dim']).mean())


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
    args = parser.parse_args()
    # train_log_analysis(args)
    test_log_analysis(args)

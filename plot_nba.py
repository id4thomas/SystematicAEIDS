import numpy as np
import pandas as pd

import argparse

from utils.plot_utils import *
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

def test_log_analysis(args):
    
    devices=["nba_Danmini_Doorbell","nba_Provision_PT_737E_Security_Camera","nba_Ecobee_Thermostat",\
        "nba_SimpleHome_XCS7_1002_WHT_Security_Camera","nba_Philips_B120N10_Baby_Monitor"]
    
    # devices=["nba_Danmini_Doorbell","nba_Provision_PT_737E_Security_Camera","nba_Ecobee_Thermostat",\
    #     "nba_Philips_B120N10_Baby_Monitor"]
    # devices=["nba_Danmini_Doorbell"]
    #64,3
    # ths=[1,13,11.05,11,8]
    # fig=plt.figure(figsize=(16, 8))
    # fig=plt.figure(figsize=(16, 6))
    fig=plt.figure(figsize=(18, 6))
    
    mcc_plot=fig.add_subplot(1,1,1)
    # mcc_plot.plot(l_dims,mean_mcc,c='r',label='AVG')
    
    colors=['b','darkgreen','darkorange','r','purple']
    colors=['b','darkgreen','r','darkorange','purple']
    markers=['o','^','s','P','D']
    for i in range(len(devices)):  
        device=devices[i]
        print(device)
        col=colors[i]
        marker=markers[i]
        dist=args.dist

        log=pd.read_csv(f'./perf_results_bak/{device}_{args.size}_{args.num_layers}.csv')
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
        seed_cond=(log["seed"]<=50)
        log_select=log_select[seed_cond]
        
        
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
        # exit()
        l_dims=np.sort(log_select.l_dim.unique())#[:-1]
        print(l_dims)
        mean_mcc=log_select.groupby(['l_dim']).mean()["mcc"].values#[:-1]
        print(mean_mcc)
        # mcc_plot.plot(l_dims,mean_mcc,c=col,label=device[4:])
        if l_dims.shape[0]==16:
            mean_mcc=mean_mcc[:-1]
            l_dims=l_dims[:-1]
        xnew = np.linspace(1, len(l_dims), num=101, endpoint=True)
        f2 = interp1d(l_dims, mean_mcc, kind='quadratic')
        mcc_plot.plot(xnew,f2(xnew),c=col)
        mcc_plot.scatter(l_dims,mean_mcc,c=col,marker=marker,s=80,label=device[4:])

        # mcc_plot.axvline(x=ths[i],ls='--',c=col,linewidth=3)
        
    # mcc_plot.set_title(f"NBaIoT Avg MCC {args.size} {args.num_layers}")
    mcc_plot.set_ylim((0.65,1.02))
    mcc_plot.set_xlabel('Latent Size',fontsize=18)
    mcc_plot.set_ylabel('MCC',fontsize=18)

    # axes=mcc_plot.axes()
    for label in (mcc_plot.get_xticklabels() + mcc_plot.get_yticklabels()):
        # label.set_fontname('Arial')
        label.set_fontsize(16)

    
    # mcc_plot.legend(loc="lower right",prop={'size': 16})
    fig.tight_layout()
    fig.savefig(f'./perf_plot/nba_mcc_{args.size}_{args.num_layers}.png')
    # print(mean_mcc)
    

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

    parser.add_argument("--dec_rate", default="0.5", type=float,
                    help="Decreae_rate")

    args = parser.parse_args()

    test_log_analysis(args)
    # draw_seed_plot(args)
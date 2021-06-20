import numpy as np
import pandas as pd

import argparse

from utils.plot_utils import *
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

def plot_mcc(args):
    config=f"{args.num_layers}_{args.max_hid_size}"
    
    #Decide Fig Size
    if config=="32_2" or config=="64_3":
        fig=plt.figure(figsize=(10,6))
    else:
        fig=plt.figure(figsize=(12, 7))
        
    mcc_plot=fig.add_subplot(1,1,1)
    #Plot Parameters
    colors=['b','g','r','c','m']
    colors=['black']
    markers=['o','^','s','P','D']
    
    col='black'
    marker='o'
    log=pd.read_csv(f"perf_results/nsl_{args.num_layers}_{args.max_hid_size}.csv")
    
    print(log.shape)
    #Select Model Structure
    log_select=log.loc[
        (log["max_hid_size"]==args.max_hid_size)&
        (log["num_layers"]==args.num_layers)&
        (log["epoch"]==args.epoch)&
        (log["batch"]==args.batch_size)
    ]

    # print("\nAvg\n",log_select.groupby(['l_dim']).mean())
    # print("\nVar\n",log_select.groupby(['l_dim']).var())

    # Get Latent Sizes
    l_dims=np.sort(log_select.l_dim.unique())[:-1]
    print(l_dims)
    
    #Get Average MCC per Latent Size
    mean_mcc=log_select.groupby(['l_dim']).mean()["mcc"].values[:-1]
    print(mean_mcc)
    #Interpolate Plot
    xnew = np.linspace(1, l_dims.shape[0], num=101, endpoint=True)
    f2 = interp1d(l_dims, mean_mcc, kind='cubic')
    mcc_plot.plot(xnew,f2(xnew),c=col,label=config)
    mcc_plot.scatter(l_dims,mean_mcc,c=col,marker=marker,s=80)
        
    # mcc_plot.set_title(f"Avg MCC {args.size} {args.num_layers}")
    mcc_plot.set_ylim((0.5,0.75))
    mcc_plot.set_xlabel('Latent Size',fontsize=20)
    mcc_plot.set_ylabel('MCC',fontsize=20)

    axes=fig.axes[0]
    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        # label.set_fontname('Arial')
        label.set_fontsize(20)

    # mcc_plot.legend(loc="lower right",prop={'size': 16})
    fig.tight_layout()
    fig.savefig(f'./perf_plot/nsl_mcc_{config}.png')

def plot_tpr(args):
    dim_tprs=np.load(f'./perf_results/nsl_cat_tpr_{args.num_layers}_{args.max_hid_size}.npy')
    cats=["DoS","U2R","R2L","Probe"]
    cat_idxs=range(1,5) #idx 0 is all attack tpr (recall)

    l_dims=range(1,int(args.max_hid_size/(2**(args.num_layers-1))))
    
    config=f"{args.num_layers}_{args.max_hid_size}"
    #Decide Fig Size
    if config=="32_2" or config=="64_3":
        fig=plt.figure(figsize=(10,6))
    else:
        fig=plt.figure(figsize=(12, 7))
    tpr_plot=fig.add_subplot(1,1,1)

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
        tpr_plot.plot(xnew,f2(xnew),c=col)
        tpr_plot.scatter(l_dims,cat_acc,c=col,marker=marker,s=80,label=cats[i])
    tpr_plot.set_xlabel('Latent Size',fontsize=20)
    tpr_plot.set_ylabel('TPR',fontsize=20)

    axes=fig.axes[0]
    for label in (axes.get_xticklabels() + axes.get_yticklabels()):
        # label.set_fontname('Arial')
        label.set_fontsize(20)
    
    # mcc_plot.legend(loc="lower right",prop={'size': 15},ncol=2)
    # mcc_plot.legend(loc='center right',prop={'size': 15},ncol=4)
    tpr_plot.legend(loc='lower right',prop={'size': 15},ncol=4)
    fig.tight_layout()
    fig.savefig(f'./perf_plot/nsl_cat_{args.max_hid_size}_{args.num_layers}.png')


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

    #Plot MCC Latent Plot
    plot_mcc(args)

    #Plot TPR Latent Plot
    plot_tpr(args)
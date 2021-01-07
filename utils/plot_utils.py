import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

# from perf_utils import *
# from .reduc_utils import *

# from sklearn.manifold import TSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.ticker as mticker

ATK=1
SAFE=0

#For Latents
def plot2d(data,label,idx=[0,1],atk_front=False):
    fig=plt.figure()
    plt2d=fig.add_subplot(1,1,1)
    if atk_front:
        #safe
        s = plt2d.scatter(data[label==SAFE,idx[0]], data[label==SAFE,idx[1]], marker='x', color='y')
        #atk
        a = plt2d.scatter(data[label==ATK,idx[0]], data[label==ATK,idx[1]], marker='o', color='b')
    else:
        #atk
        a = plt2d.scatter(data[label==ATK,idx[0]], data[label==ATK,idx[1]], marker='o', color='b')
        #safe
        s = plt2d.scatter(data[label==SAFE,idx[0]], data[label==SAFE,idx[1]], marker='x', color='y')
    
    plt2d.legend((s,a),('normal','attack'))
    return fig

def plot3d(data,label,idx=[0,1,2],atk_front=False):
    fig=plt.figure()
    plt3d=fig.add_subplot(1,1,1,projection='3d')

    if atk_front:
        #safe
        s=plt3d.scatter(data[label==SAFE,idx[0]], data[label==SAFE,idx[1]],data[label==SAFE,idx[2]], marker='x', color='y')
        #atk
        a=plt3d.scatter(data[label==ATK,idx[0]], data[label==ATK,idx[1]],data[label==ATK,idx[2]], marker='o', color='b')
    else:
        #safe
        s=plt3d.scatter(data[label==SAFE,idx[0]], data[label==SAFE,idx[1]],data[label==SAFE,idx[2]], marker='x', color='y')
        #atk
        a=plt3d.scatter(data[label==ATK,idx[0]], data[label==ATK,idx[1]],data[label==ATK,idx[2]], marker='o', color='b')
    plt3d.legend((s,a),('normal','attack'))
    return fig

def plot_losses(train_l,val_l,title):
    fig=plt.figure()
    l_plot=fig.add_subplot(1,1,1)
    l_plot.plot(range(len(train_l)),train_l,c='r',label='train')
    l_plot.plot(range(len(val_l)),val_l,c='b',label='val')
    l_plot.set_title(title)
    l_plot.legend()
    return fig

def plot_aucs(auc_l1,auc_l2,auc_cos,title="AUC"):
    fig=plt.figure()
    auc_plot=fig.add_subplot(1,1,1)
    auc_plot.plot(range(len(auc_l1)),auc_l1,c='r',label='AUC L1')
    auc_plot.plot(range(len(auc_l2)),auc_l2,c='g',label='AUC L2')
    auc_plot.plot(range(len(auc_cos)),auc_cos,c='b',label='AUC Cos')
    auc_plot.set_title(title)
    auc_plot.legend()
    return fig

def plot_auc(auc,title="AUC"):
    fig=plt.figure()
    auc_plot=fig.add_subplot(1,1,1)
    auc_plot.plot(range(len(auc)),auc,c='g',label='AUC L2')
    auc_plot.set_title(title)
    auc_plot.legend()
    return fig

def plot_auc_f1(auc,f1,title="AUC & F1"):
    fig=plt.figure()
    auc_plot=fig.add_subplot(1,1,1)
    auc_plot.plot(range(len(auc)),auc,c='g',label='AUC')
    auc_plot.plot(range(len(f1)),f1,c='b',label='F1')
    auc_plot.set_title(title)
    auc_plot.legend()
    return fig
    
def plot_ld(data,reduc,label,atk_front=False):
    #Low Dim Data
    reduced,_=train_reduc(reduc,reduc_type='pca',n_c=2)

    fig=plt.figure()
    plt2d=fig.add_subplot(1,1,1)
    idx=[0,1]
    if atk_front:
        #safe
        s = plt2d.scatter(reduced[label==SAFE,idx[0]], reduced[label==SAFE,idx[1]], marker='x', color='y')
        #atk
        a = plt2d.scatter(reduced[label==ATK,idx[0]], reduced[label==ATK,idx[1]], marker='o', color='b')
    else:
        #atk
        a = plt2d.scatter(reduced[label==ATK,idx[0]], reduced[label==ATK,idx[1]], marker='o', color='b')
        #safe
        s = plt2d.scatter(reduced[label==SAFE,idx[0]], reduced[label==SAFE,idx[1]], marker='x', color='y')
    
    plt2d.legend((s,a),('normal','attack'))
    plt2d.set_xlabel("Reduced Dim 1")
    plt2d.set_ylabel("Reduced Dim 2")
    fig.savefig('./ae/pca_32_ld.png')

def plot_dist_2d(data,recon,reduc,label,atk_front=False):
    dist=np.mean(np.square(data-recon),axis=1)
    #Low Dim Data
    reduced,_=train_reduc(reduc,reduc_type='pca',n_c=2)

    fig=plt.figure()
    plt2d=fig.add_subplot(1,1,1)
    idx=[0,1]
    if atk_front:
        #safe
        s = plt2d.scatter(reduced[label==SAFE,idx[0]], dist[label==SAFE], marker='x', color='y')
        #atk
        a = plt2d.scatter(reduced[label==ATK,idx[0]], dist[label==ATK], marker='o', color='b')
    else:
        #atk
        a = plt2d.scatter(reduced[label==ATK,idx[0]], dist[label==ATK], marker='o', color='b')
        #safe
        s = plt2d.scatter(reduced[label==SAFE,idx[0]], dist[label==SAFE], marker='x', color='y')
    
    plt2d.legend((s,a),('normal','attack'))
    plt2d.set_xlabel("Latent Dim 1")
    plt2d.set_ylabel("L2 Distance")
    fig.savefig('./ae/ae_dist_2d_mse.png')

def plot_dist(data,dist,label,atk_front=False):

    fig=plt.figure()
    plt3d=fig.add_subplot(1,1,1,projection='3d')
    idx=[0,1]
    if atk_front:
        #safe
        s=plt3d.scatter(dist[label==SAFE],data[label==SAFE,idx[0]], data[label==SAFE,idx[1]], marker='x', color='y')
        #atk
        a=plt3d.scatter(dist[label==ATK],data[label==ATK,idx[0]], data[label==ATK,idx[1]], marker='o', color='b')
    else:
        #atk
        a=plt3d.scatter(dist[label==ATK],data[label==ATK,idx[0]], data[label==ATK,idx[1]], marker='o', color='b')
        #safe
        s=plt3d.scatter(dist[label==SAFE],data[label==SAFE,idx[0]], data[label==SAFE,idx[1]], marker='x', color='y')    
    plt3d.legend((s,a),('normal','attack'))
    plt3d.set_ylabel("Latent Dim 1")
    plt3d.set_zlabel("Latent Dim 2")
    plt3d.set_xlabel("L2 Distance")
    plt3d.invert_xaxis()


    return fig


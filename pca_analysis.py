import numpy as np

from utils.data_utils import *
from utils.perf_utils import *
from utils.reduc_utils import *
from utils.plot_utils import *

from scipy.stats import norm
import scipy.stats as ss

# Analyze Cumulative Explained Variance ratio of training data with PCA

#CIC-IDS 17
print("CICIDS 17")
data_path='data/cicids17/processed/'
train_name='mon'
x_train,y_train=get_hdf5_data(data_path+'{}.hdf5'.format(train_name),labeled=False)
print("Train: {}".format(x_train.shape[0]))
x_train_r,reduc=train_reduc(x_train,reduc_type='pca',n_c=x_train.shape[1])
vr=np.array(reduc.explained_variance_ratio_)
print("Explained Variance Ratio")
print(reduc.explained_variance_ratio_)
print("Cumulative Explained Variance Ratio")
print(np.cumsum(vr))

#NSL KDD
print("NSL KDD")
data_path='data/nsl_kdd/split/'
x_train,y_train=get_hdf5_data(data_path+'train.hdf5',labeled=True)

x_train,y_train=filter_label(x_train,y_train,select_label=SAFE)
print("Filter Train: Normal:{}, Atk:{}".format(x_train[y_train==0].shape[0],x_train[y_train==1].shape[0]))

x_train_r,reduc=train_reduc(x_train,reduc_type='pca',n_c=x_train.shape[1])
vr=np.array(reduc.explained_variance_ratio_)
print("Explained Variance Ratio")
print(reduc.explained_variance_ratio_)
print("Cumulative Explained Variance Ratio")
print(np.cumsum(vr))
# np.save('nsl_pca_vr.npy',vr)
# np.save('nsl_pca_cumvr.npy',np.cumsum(vr))


#Kyoto Honeypot 15
exit()
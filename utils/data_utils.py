import os
import h5py
import torch
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def split_data(data,label,split_ratio=0.1,seed_num=42):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=split_ratio, random_state=seed_num)
    return X_train, X_test, y_train, y_test

def get_hdf5_data(file_path,labeled=False,only_safe=False):
    with h5py.File(file_path,'r') as f:
        data=f['x'].value
        if labeled:
            label=f['y'].value
        else:
            label=[]
    if only_safe:
        filter_atk(data,label)
    return data,label

def make_balanced_val(data,has_type=False):
    x_val=data['x_val']
    y_val=data['y_val']

    #Val
    x_val_safe=x_val[y_val==0]
    y_val_safe=np.zeros(x_val_safe.shape[0])

    x_val_atk=x_val[y_val==1]
    y_val_atk=np.ones(x_val_atk.shape[0])

    if has_type:
        y_val_type=data['y_val_type']
        y_val_safe_type=np.zeros(x_val_safe.shape[0])
        # y_test_atk_type=np.ones(x_test_atk.shape[0])
        y_val_atk_type=y_val_type[y_val==1]

    sample_idx=np.random.choice(x_val_safe.shape[0], x_val_atk.shape[0])

    x_val_safe_sampled=x_val_safe[sample_idx]
    y_val_safe_sampled=y_val_safe[sample_idx]

    x_val=np.concatenate((x_val_safe_sampled,x_val_atk),axis=0)
    y_val=np.concatenate((y_val_safe_sampled,y_val_atk),axis=0)
    if has_type:
        y_val_safe_type_sampled=y_val_safe_type[sample_idx]
        y_val_type=np.concatenate((y_val_safe_type_sampled,y_val_atk_type),axis=0)

        return {'x_val': x_val, 'y_val': y_val,'y_val_type':y_val_type}
    else:
        return {'x_val': x_val, 'y_val': y_val}

def make_balanced_test(data,has_type=False):
    x_test=data['x_test']
    y_test=data['y_test']
    #Test
    x_test_safe=x_test[y_test==0]
    y_test_safe=np.zeros(x_test_safe.shape[0])

    x_test_atk=x_test[y_test==1]
    y_test_atk=np.ones(x_test_atk.shape[0])

    if has_type:
        y_test_type=data['y_test_type']
        y_test_safe_type=np.zeros(x_test_safe.shape[0])
        # y_test_atk_type=np.ones(x_test_atk.shape[0])
        y_test_atk_type=y_test_type[y_test==1]

    sample_idx=np.random.choice(x_test_safe.shape[0], x_test_atk.shape[0])
    x_test_safe_sampled=x_test_safe[sample_idx]
    y_test_safe_sampled=y_test_safe[sample_idx]

    x_test=np.concatenate((x_test_safe_sampled,x_test_atk),axis=0)
    y_test=np.concatenate((y_test_safe_sampled,y_test_atk),axis=0)
    if has_type:
        y_test_safe_type_sampled=y_test_safe_type[sample_idx]
        y_test_type=np.concatenate((y_test_safe_type_sampled,y_test_atk_type),axis=0)
        return {'x_test': x_test, 'y_test': y_test,'y_test_type':y_test_type}
    else:
        return {'x_test': x_test, 'y_test': y_test}
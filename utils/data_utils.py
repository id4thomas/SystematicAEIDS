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

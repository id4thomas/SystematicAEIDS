import h5py
from sklearn.model_selection import train_test_split

#Split Data with given ratio
def split_data(data,label,split_ratio=0.1,seed_num=42):
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=split_ratio, random_state=seed_num)
    return X_train, X_test, y_train, y_test

#Load data in hdf5 format
def get_hdf5_data(file_path,labeled=False):
    with h5py.File(file_path,'r') as f:
        data=f['x'].value
        if labeled:
            label=f['y'].value
        else:
            label=[]
    return data,label

#Select Certain Labels
def select_label_data(data,label,select_label=0):
    select_idx=[label==select_label]
    data=data[tuple(select_idx)]
    label=label[tuple(select_idx)]
    return data,label

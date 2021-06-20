import numpy as np
import pandas as pd

from utils.data_utils import * 

# Preprocessing Scripts
from data.preprocess_nsl import preprocess as nsl_preprocess


SAFE=0
ATK=1

def preprocess(dataset="nsl",is_train=False,scaler=None,num_desc=None,serviceData=None,flagData=None):
    if dataset=="nsl":
        if is_train:
            processed_df,y,y_type,scaler,num_desc=preprocess(train,serviceData,flagData)  
        else:
            processed_df,y,y_type,_,_=preprocess(train,serviceData,flagData,is_train=is_train,scaler=scaler, num_desc=num_desc))  

    return processed_df,y,y_type,scaler,num_desc
def load_data(dataset="nsl",device_name=None):
    data={}
    if dataset=="nsl":
        data_dir="data/nsl_kdd/split"
        train=pd.read_csv(data_dir+'/train.csv',header=None)
        val=pd.read_csv(data_dir+'/val.csv',header=None)
        
        service = open(data_dir+'/service.txt', 'r')
        serviceData = service.read().split('\n')
        service.close()

        flag = open(data_dir+'/flag.txt', 'r')
        flagData = flag.read().split('\n')
        flag.close()
        
        train_df,y_train,y_train_types,scaler,num_desc=preprocess(train,serviceData,flagData)  
        

    if dataset=="cic":
        #Train
        data_path='data/cicids17/processed/'
        train_name='mon'
        x_train,y_train=get_hdf5_data(data_path+'{}.hdf5'.format(train_name),labeled=False)
        print("Train: {}".format(x_train.shape[0]))
        # print("Train: Normal:{}, Atk:{}".format(x_train[y_train=='BENIGN'].shape[0],x_train[y_train!='BENIGN'].shape[0]))

        #Val
        data_path='data/cicids17/split/'
        x_val,_=get_hdf5_data(data_path+'val.hdf5',labeled=False)
        y_val_type=np.load(data_path+'val_label.npy',allow_pickle=True)
        y_val=np.zeros(x_val.shape[0])
        y_val[y_val_type!='BENIGN']=1
        print("Val: Normal:{}, Atk:{}".format(x_val[y_val_type=='BENIGN'].shape[0],x_val[y_val_type!='BENIGN'].shape[0]))

        #Balanced
        # data=make_balanced_val({'x_val': x_val, 'y_val': y_val})
        # x_val=data['x_val']
        # y_val=data['y_val']
        # print("Bal Val: Normal:{}, Atk:{}".format(x_val[y_val==SAFE].shape[0],x_val[y_val!=SAFE].shape[0]))
        
        x_test,_=get_hdf5_data(data_path+'test.hdf5',labeled=False)
        y_test_type=np.load(data_path+'test_label.npy',allow_pickle=True)
        y_test=np.zeros(x_test.shape[0])
        y_test[y_test_type!='BENIGN']=1
        print("Test: Normal:{}, Atk:{}".format(x_test[y_test_type=='BENIGN'].shape[0],x_test[y_test_type!='BENIGN'].shape[0]))
        
        data['x_train']=x_train
        data['y_train']=y_train
        data['x_val']=x_val
        data['y_val']=y_val
        data['y_val_type']=y_val_type
        data['x_test']=x_test
        data['y_test']=y_test
        data['y_test_type']=y_test_type


    elif dataset=="iotbotnet":
        data_path='data/iotbotnet/processed/'
        x_train,y_train=get_hdf5_data(data_path+'train.hdf5',labeled=True)
        x_train,y_train=filter_label(x_train,y_train,select_label=SAFE)
        print("Filter Train: Normal:{}, Atk:{}".format(x_train[y_train==0].shape[0],x_train[y_train==1].shape[0]))

        x_val,y_val=get_hdf5_data(data_path+'val.hdf5',labeled=True)
        print("Val: Normal:{}, Atk:{}".format(x_val[y_val==SAFE].shape[0],x_val[y_val!=SAFE].shape[0]))
        
        x_test,y_test=get_hdf5_data(data_path+'test.hdf5',labeled=True)
        # y_test_type=np.load(data_path+'test_label.npy',allow_pickle=True)
        print("Test Val: Normal:{}, Atk:{}".format(x_test[y_test==SAFE].shape[0],x_test[y_test!=SAFE].shape[0]))
        
        #Category Label in Separate CSV - to be implemented
        
        data['x_train']=x_train
        data['y_train']=y_train
        data['x_val']=x_val
        data['y_val']=y_val
        # data['y_val_type']=y_val_type
        data['x_test']=x_test
        data['y_test']=y_test
        # data['y_test_type']=y_test_type
        
    elif dataset=="nbaiot":
        data_path='data/nbaiot/split/'
        # x_train,y_train
        
    return data
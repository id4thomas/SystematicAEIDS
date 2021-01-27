import numpy as np
import pandas as pd
import os
import h5py
import argparse
import random
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def preprocess(df,scaler=None):
    #Check Labels
    print(df["Label"].value_counts())

    #Remove NAN,INF samples
    is_nan=df.isin([np.nan]).any(1)
    is_inf=df.isin([np.inf, -np.inf]).any(1)
    print("NAN: {}, INF: {}".format(df[is_nan].shape[0],df[is_inf].shape[0]))

    prev_count=df.shape[0]
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    aft_count=df.shape[0]
    print("Removed {} Now {}".format(prev_count-aft_count,df.shape))

    #Process Categorical
    #Port -> 3 Categories
    df["Src_Port"] = df["Src_Port"].map(lambda x : 2 if x > 49152 else 1 if x > 1024 else 0)
    df["Dst_Port"] = df["Dst_Port"].map(lambda x : 2 if x > 49152 else 1 if x > 1024 else 0)

    enc = OneHotEncoder(categories=[[0,1,2],[0,1,2],[0,6,17]])
    enc.fit(df[["Src_Port","Dst_Port","Protocol"]].values)
    oneHotEncoding = enc.transform(df[["Src_Port","Dst_Port","Protocol"]].values).toarray()
    print(oneHotEncoding.shape)

    #Label
    # Label/Cat/Sub_Cat
    # df[" Label"] = df[" Label"].map(lambda x : 0 if x=='BENIGN' else 1)
    label_df=df[["Label","Cat","Sub_Cat"]]
    label_oh=df["Label"].values
    print("Label",label_df.shape)
    print(label_df.head)
    print(label_oh)
    # print(type(label))

    #Check Flags
    print(df['Fwd_PSH_Flags'].value_counts())
    print(df['Bwd_PSH_Flags'].value_counts())
    print(df['Fwd_URG_Flags'].value_counts())
    print(df['Bwd_URG_Flags'].value_counts())

    #Drop Unnecessry,Categorical Features
    df.drop(["Flow_ID","Src_IP","Dst_IP","Src_Port","Dst_Port","Timestamp","Label","Cat","Sub_Cat","Protocol"],axis=1,inplace=True)
    print(df.shape)
    # exit()
    #Fit scaler if not given
    if scaler is None:
        scaler=MinMaxScaler()
        scaler.fit(df.values)
    scaled=scaler.transform(df.values)
    print(scaled.shape)

    #Final DF
    df=pd.DataFrame(np.concatenate((scaled,oneHotEncoding),axis=1))
    # print("Wo Label",df.shape)
    # df=pd.concat([df,pd.DataFrame(label)],axis=1)
    print("Processed",df.shape)
    print(df.head)
    return df,label_df,label_oh,scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=10, type=int,
                        help="random seed")
    parser.add_argument("--data_dir", default="iotbotnet/split", type=str,
                        help="Data Directory")
    parser.add_argument("--save_dir", default="iotbotnet/processed", type=str,
                        help="Save Directory")
    args = parser.parse_args()
    
    #Set Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    #Check Directory existence
    save_dir=args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data_dir=args.data_dir
    #Fit Minmax with train data
    df=pd.read_csv(data_dir+'/train.csv')
    # df=pd.read_csv(data_dir+log_files['mon'])
    # exit()
    df,label_df,label_oh,scaler=preprocess(df)
    
    
    print("Saving Train")
    with h5py.File(save_dir+'/train.hdf5', 'w') as hdf:
        hdf['x'] = df.values[:]
        hdf['y'] = label_oh
    label_df.to_csv(save_dir+'/train_label.csv')
    

    #Val
    df=pd.read_csv(data_dir+'/val.csv')
    # df=pd.read_csv(data_dir+log_files['mon'])
    # exit()
    df,label_df,label_oh,_=preprocess(df,scaler=scaler)
    
    
    print("Saving Val")
    with h5py.File(save_dir+'/val.hdf5', 'w') as hdf:
        hdf['x'] = df.values[:]
        hdf['y'] = label_oh
    label_df.to_csv(save_dir+'/val_label.csv')

    #Test
    df=pd.read_csv(data_dir+'/test.csv')
    # df=pd.read_csv(data_dir+log_files['mon'])
    # exit()
    df,label_df,label_oh,_=preprocess(df,scaler=scaler)

    
    print("Saving Test")
    with h5py.File(save_dir+'/test.hdf5', 'w') as hdf:
        hdf['x'] = df.values[:]
        hdf['y'] = label_oh
    label_df.to_csv(save_dir+'/test_label.csv')
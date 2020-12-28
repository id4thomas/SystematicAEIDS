import numpy as np
import pandas as pd
import os
import h5py
import argparse
import random
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

#MachineLearningCVE provided files
#monday used as training data
log_files={
    'mon': 'Monday-WorkingHours.pcap_ISCX.csv',
    'tue': 'Tuesday-WorkingHours.pcap_ISCX.csv',
    'wed': 'Wednesday-workingHours.pcap_ISCX.csv',
    'thur_morning': 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'thur_aft': 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'fri_morning': 'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'fri_aft_portscan': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'fri_aft_ddos': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
}
log_keys=['mon','tue','wed','thur_morning','thur_aft','fri_morning','fri_aft_portscan','fri_aft_ddos']

def preprocess(df,scaler=None):
    #Check Labels
    print(df[" Label"].value_counts())

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
    df[" Destination Port"] = df[" Destination Port"].map(lambda x : 2 if x > 49152 else 1 if x > 1024 else 0)
    enc = OneHotEncoder(categories=[[0,1,2]])
    enc.fit(df[[" Destination Port"]].values)
    oneHotEncoding = enc.transform(df[[" Destination Port"]].values).toarray()

    #Label
    # df[" Label"] = df[" Label"].map(lambda x : 0 if x=='BENIGN' else 1)
    label=df[" Label"].values
    print("Label",label.shape)
    print(type(label))
    #Check Flags
    print(df["Fwd PSH Flags"].value_counts()) #0,1
    print(df[" Fwd URG Flags"].value_counts()) #0

    #Normalize Numeric Values
    df.drop([" Destination Port"," Label"],axis=1,inplace=True)

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
    return df,label,scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int,
                        help="random seed")
    parser.add_argument("--data_dir", default="cicids17/MachineLearningCVE", type=str,
                        help="Data Directory")
    parser.add_argument("--save_dir", default="cicids17/processed", type=str,
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
    #Fit Minmax with monday data
    df=pd.read_csv(data_dir+'/{}'.format(log_files['mon']))
    # df=pd.read_csv(data_dir+log_files['mon'])
    df,label,scaler=preprocess(df)
    
    print("Saving Processed: mon")
    with h5py.File(save_dir+'/mon.hdf5', 'w') as hdf:
        hdf['x'] = df.values[:]
    np.save(save_dir+'/mon_label.npy',label)
    
    for key in log_keys[1:]:
        df=pd.read_csv(data_dir+'/{}'.format(log_files[key]))
        df,label,_=preprocess(df,scaler)
        print("Saving Processed: {}".format(key))
        with h5py.File(save_dir+'/{}.hdf5'.format(key), 'w') as hdf:
            hdf['x'] = df.values[:]
        np.save(save_dir+'/{}_label.npy'.format(key),label)
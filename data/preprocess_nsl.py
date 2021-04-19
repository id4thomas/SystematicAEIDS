import numpy as np
import pandas as pd
import os
import h5py
import argparse
import random
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def to_machine_readable(df, service_list, flag_list, scaler=None, num_desc=None, test=False):
    if test:
        sc=scaler
        num_desc = num_desc
    else:
        sc = MinMaxScaler()
        num_desc=df.loc[:, [0, 4, 5]].describe()
        
    enc = OneHotEncoder(categories=[range(3), range(len(service_list)), range(len(flag_list))])
    

    # extract and drop label
    label, df_label = [], []
    # if test:
    #     label = df[41].copy().values.reshape((df.shape[0], 1))
    #     df_label = pd.DataFrame(label)
    #     df.drop([41], axis=1, inplace=True)

    # index 0, 4, 5: duration, src_bytes, dst_bytes (in kyoto: index 0, 2, 3)
    attr_name = ['duration', '', '', '', 'src_bytes', 'dst_bytes']
    for i in [0, 4, 5]:
        print('Converting {0} data (index {1}) to machine readable...'.format(attr_name[i], i))
        iqr = (num_desc[i].values[6] - num_desc[i].values[4])
        std = num_desc[i].values[6] + iqr * 1.5  # IQR upper fence = Q3 + 1.5 * IQR
        if std == 0:
            df[i] = df[i].map(lambda x: 1 if x > 0 else 0)
        else:
            df[i] = df[i].map(lambda x: std if x > std else x)
    sc.fit(df[[0, 4, 5]].values)
    df[[0, 4, 5]] = sc.transform(df[[0, 4, 5]].values)

    # index 22, 31, 32: count, dst_host_count, dst_host_srv_count (in kyoto: index 4, 8, 9)
    print('Converting count data (index 22, 31, 32) to machine readable...')
    sc.fit(df[[22, 31, 32]].values.astype(np.float32))
    df[[22, 31, 32]] = sc.transform(df[[22, 31, 32]].values.astype(np.float32))

    # index 1, 2, 3: protocol_type, service, flag (in kyoto: index 23, 1, 13)
    print('Converting type data (index 1, 2, 3) to machine readable...')
    enc.fit(df[[1, 2, 3]].values)
    one_hot_arr = enc.transform(df[[1, 2, 3]].values).toarray()

    # drop one-hot data and attach it again
    print('Dropping and attaching one-hot encoding data...')
    df.drop([1, 2, 3], axis=1, inplace=True)
    df_final = np.concatenate((df.values, one_hot_arr), axis=1)
    df_final = pd.DataFrame(df_final)
    return df_final,sc,num_desc

def to_numeric(df, service_list, flag_list, test=False, scaler=None, num_desc=None):
    # Categorical Data to Numeric Values
    # index 1: protocol_type
    print('Replacing protocol_type values to numeric...')
    df[1].replace(['tcp', 'udp', 'icmp'], range(3), inplace=True)

    # index 2: service
    print('Replacing service values to numeric...')
    df[2].replace(service_list, range(len(service_list)), inplace=True)

    # index 3: flag
    print('Replacing flag values to numeric...')
    df[3].replace(flag_list, range(len(flag_list)), inplace=True)


    #Label
    print(df[41].value_counts())
    label_type=df[41].values
    df[41] = df[41].map(lambda x: 0 if x == 'normal' else 1)
    label=df[41].values
    print(label.shape)
    df = df.loc[:, [0, 1, 2, 3, 4, 5, 22, 24, 25, 28, 31, 32, 35, 37, 38]]
    
    # Preprocess Numerical Values
    df,scaler,num_desc=to_machine_readable(df,service_list,flag_list, scaler=scaler, num_desc=num_desc, test=test)
    print(df.shape) #114 + 1
    # if not test:
    #     # extract only the same features from Kyoto 2006+ dataset
    #     df = df.loc[:, [0, 1, 2, 3, 4, 5, 22, 24, 25, 28, 31, 32, 35, 37, 38]]
    # else:
    #     # include label
    #     df = df.loc[:, [0, 1, 2, 3, 4, 5, 22, 24, 25, 28, 31, 32, 35, 37, 38, 41]]
    #     df[41] = df[41].map(lambda x: 0 if x == 'normal' else 1)  # normal 0, attack 1

    # save as csv file
    # if save:
    #     if not os.path.exists('csv'):
    #         os.makedirs('csv')
    #     if not test:
    #         if not attack:
    #             print('Saving file:', os.path.join('csv', 'train_normal_numeric.csv'))
    #             df.to_csv(os.path.join('csv', 'train_normal_numeric.csv'))
    #         else:
    #             print('Saving file:', os.path.join('csv', 'train_mixed_numeric.csv'))
    #             df.to_csv(os.path.join('csv', 'train_mixed_numeric.csv'))
    #     else:
    #         print('Saving file:', os.path.join('csv', 'test_numeric.csv'))
    #         df.to_csv(os.path.join('csv', 'test_numeric.csv'))

    return df,label,label_type,scaler,num_desc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=10, type=int,
                        help="random seed")
    parser.add_argument("--data_dir", default="nsl_kdd/split", type=str,
                        help="Data Directory")
    
    parser.add_argument("--save_dir", default="nsl_kdd/processed", type=str,
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
    #Read Service, Flag
    service = open(data_dir+'/service.txt', 'r')
    serviceData = service.read().split('\n')
    service.close()

    flag = open(data_dir+'/flag.txt', 'r')
    flagData = flag.read().split('\n')
    flag.close()

    #Train Data
    # train_df=pd.read_csv(data_dir+'/KDDTrain+.txt',header=None)
    train_df=pd.read_csv(data_dir+'/train.csv',header=None)
    train_df,train_label,train_label_types,scaler,num_desc=to_numeric(train_df,serviceData,flagData)

    val_df=pd.read_csv(data_dir+'/val.csv',header=None)
    val_df,val_label,val_label_types,_,_=to_numeric(val_df,serviceData,flagData,test=True,scaler=scaler, num_desc=num_desc)

    test_df=pd.read_csv(data_dir+'/test.csv',header=None)
    test_df,test_label,test_label_types,_,_=to_numeric(test_df,serviceData,flagData,test=True,scaler=scaler, num_desc=num_desc)

    # #Save Data
    

    # with h5py.File(save_path+'processed/kddcup_{}.hdf5'.format(data_name), 'w') as hdf:
    #     print('Saving file : {}'.format(save_path+'processed/kddcup_{}.hdf5'.format(data_name)))
    #     hdf['x'] = df_final.values[:]

    # #Save Label
    # np.save(save_path+'processed/kddcup_{}_label.npy'.format(data_name),label)

    # #Split Train/Test
    # df_final = np.concatenate((df_final.values, np.expand_dims(label, axis=1)), axis=1)
    # df_final = pd.DataFrame(df_final)
    # train, test = train_test_split(df_final, test_size=0.5)

    # #114 + 1 (Label)
    # print("Train {} Test {}".format(train.shape,test.shape))

    with h5py.File(save_dir+'/train.hdf5', 'w') as hdf:
        print('Saving file : {}'.format(save_dir+'/train.hdf5'))
        # hdf['x'] = train.values[:,:train.shape[1]-1]
        # hdf['y'] = train.values[:,train.shape[1]-1]
        hdf['x'] = train_df.values
        hdf['y'] = train_label
    np.save(save_dir+'/train_label.npy',train_label_types)

    with h5py.File(save_dir+'/val.hdf5', 'w') as hdf:
        print('Saving file : {}'.format(save_dir+'/val.hdf5'))
        # hdf['x'] = train.values[:,:test.shape[1]-1]
        # hdf['y'] = train.values[:,test.shape[1]-1]
        hdf['x'] = val_df.values
        hdf['y'] = val_label
    np.save(save_dir+'/test_label.npy',test_label_types)

    with h5py.File(save_dir+'/test.hdf5', 'w') as hdf:
        print('Saving file : {}'.format(save_dir+'/test.hdf5'))
        # hdf['x'] = train.values[:,:test.shape[1]-1]
        # hdf['y'] = train.values[:,test.shape[1]-1]
        hdf['x'] = test_df.values
        hdf['y'] = test_label
    np.save(save_dir+'/test_label.npy',test_label_types)
    
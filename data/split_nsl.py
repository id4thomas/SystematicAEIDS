import argparse

import sys
sys.path.append('..')
from utils.data_utils import *
# from utils.perf_utils import *
# from utils.reduc_utils import *
# from utils.plot_utils import *

ATK=1
SAFE=0

# Argument Setting
parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int,
                    help="random seed for reproductability")
parser.add_argument("--val", default=0.1, type=float,
                    help="Validation Ratio")
parser.add_argument("--data_dir", default="nsl_kdd/processed", type=str,
                        help="Data Directory")
parser.add_argument("--save_dir", default="nsl_kdd/split", type=str,
                    help="Save Directory")
args = parser.parse_args()
set_seed(args.seed)


#Check Directory existence
save_dir=args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
data_dir=args.data_dir

x,y=get_hdf5_data(data_dir+'/train.hdf5',labeled=True)
y_type=np.load(data_dir+'/train_label.npy',allow_pickle=True)
print("All Data",x.shape)

x_train,x_val,y_train,y_val,y_train_type,y_val_type=train_test_split(x,y,y_type ,test_size=args.val, random_state=42)
print("Train: Normal:{}, Atk:{}".format(x_train[y_train==0].shape[0],x_train[y_train!=0].shape[0]))
print("Val: Normal:{}, Atk:{}".format(x_val[y_val==0].shape[0],x_val[y_val!=0].shape[0]))

# print("Saving Validation")
with h5py.File(save_dir+'/val.hdf5', 'w') as hdf:
        print('Saving file : {}'.format(save_dir+'/val.hdf5'))
        hdf['x'] = x_val[:]
        hdf['y'] = y_val[:]

# print("Saving Train")
with h5py.File(save_dir+'/train.hdf5', 'w') as hdf:
        print('Saving file : {}'.format(save_dir+'/train.hdf5'))
        hdf['x'] = x_train[:]
        hdf['y'] = y_train[:]
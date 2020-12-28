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
parser.add_argument("--data_dir", default="cicids17/processed", type=str,
                        help="Data Directory")
parser.add_argument("--save_dir", default="cicids17/split", type=str,
                    help="Save Directory")
args = parser.parse_args()
set_seed(args.seed)

val_keys=['tue','wed','thur_morning','thur_aft','fri_morning','fri_aft_portscan','fri_aft_ddos']

#Check Directory existence
save_dir=args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
data_dir=args.data_dir

x=[]
y=[]
for file_key in val_keys:
    data,_=get_hdf5_data(data_dir+'/{}.hdf5'.format(file_key),labeled=False)
    label=np.load(data_dir+'/{}_label.npy'.format(file_key),allow_pickle=True)
    x.append(data)
    y.append(label)
x=np.concatenate(x)
y=np.concatenate(y)
print("All Data",x.shape)

x_test,x_val,y_test,y_val=split_data(x,y,split_ratio=args.val,seed_num=args.seed)
print("Val: Normal:{}, Atk:{}".format(x_val[y_val=='BENIGN'].shape[0],x_val[y_val!='BENIGN'].shape[0]))
print("Test: Normal:{}, Atk:{}".format(x_test[y_test=='BENIGN'].shape[0],x_test[y_test!='BENIGN'].shape[0]))

print("Saving Validation")
with h5py.File(save_dir+'/val.hdf5', 'w') as hdf:
        hdf['x'] = x_val[:]
np.save(save_dir+'/val_label.npy',y_val)

print("Saving Test")
with h5py.File(save_dir+'/test.hdf5', 'w') as hdf:
        hdf['x'] = x_test[:]
np.save(save_dir+'/test_label.npy',y_test)
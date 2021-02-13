import argparse

import sys
import os
sys.path.append('..')
from utils.data_utils import *

ATK=1
SAFE=0

# Argument Setting
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=10, type=int,
                    help="random seed")
parser.add_argument("--data_dir", default="nbaiot/data", type=str,
                    help="Data Directory")
parser.add_argument("--save_dir", default="nbaiot/split", type=str,
                    help="Save Directory")
parser.add_argument("--device", default="Danmini_Doorbell", type=str,
                    help="Device Name")
args = parser.parse_args()
set_seed(args.seed)

#Check Directory existence
save_dir=args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
data_dir=args.data_dir

df=pd.read_csv(data_dir+f'/{args.device}.csv')
print(df.head)
print(df["Label"])
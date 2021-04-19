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
parser.add_argument("--data_dir", default="nbaiot", type=str,
                    help="Data Directory")
parser.add_argument("--save_dir", default="nbaiot/data", type=str,
                    help="Save Directory")
parser.add_argument("--device", default="Danmini_Doorbell", type=str,
                    help="Device Name")

args = parser.parse_args()
set_seed(args.seed)

#Make Split per Device

#Check Directory existence
save_dir=args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
data_dir=args.data_dir+f'/{args.device}'


benign_df=pd.read_csv(data_dir+'/benign_traffic.csv')
# print(benign_df.head)
# print(benign_df.columns)
benign_df["Label"]=0
benign_df["Cat"]="Benign"
benign_df["Sub_cat"]="Benign"
print(benign_df.head)
print(benign_df.columns)
print(benign_df["Label"])
print(benign_df["Cat"])

# for cat in os.walk(data_dir):
#     if os.path.isdir(cat[0]):
#         print(cat[0])
print(os.listdir(data_dir))
atk_dfs=[]
for cat_f in os.scandir(data_dir):
    #Category
    if os.path.isdir(cat_f.path):
        #Sub Category
        cat_dir=cat_f.path
        cat=cat_dir[cat_dir.rfind('/')+1:]
        # print(sub_cat)
        print("Cat Path",cat_dir)
        for sub_cat_f in os.scandir(cat_dir):
            if not os.path.isdir(sub_cat_f.path):
                print(sub_cat_f.path)
                sub_cat_dir=sub_cat_f.path
                sub_cat=sub_cat_dir[sub_cat_dir.rfind('/')+1:sub_cat_dir.rfind('.')]
                print(sub_cat)
                atk_df=pd.read_csv(sub_cat_f.path)
                atk_df["Label"]=1
                atk_df["Cat"]=cat
                atk_df["Sub_cat"]=sub_cat
                print(atk_df.shape)
                atk_dfs.append(atk_df)
atk_df=pd.concat(atk_dfs,axis=0)
print(atk_df["Cat"])
print(atk_df["Sub_cat"])

df=pd.concat([benign_df,atk_df],axis=0)
df.to_csv(save_dir+'/'+args.device+'.csv',index=False)


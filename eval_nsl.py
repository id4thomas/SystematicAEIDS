from __future__ import absolute_import, print_function
import argparse
import os

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from model.ae import AE

from utils.data_utils import *
from utils.perf_utils import *
from utils.plot_utils import *
from utils.train_utils import *


#Preprocess Script
from data.preprocess_nsl import *

import numpy as np

ATK=1
SAFE=0

def load_data():
    data_dir='data/nsl_kdd/split'
    train=pd.read_csv(data_dir+'/train.csv',header=None)
    test=pd.read_csv(data_dir+'/test.csv',header=None)

    service = open(data_dir+'/service.txt', 'r')
    serviceData = service.read().split('\n')
    service.close()

    flag = open(data_dir+'/flag.txt', 'r')
    flagData = flag.read().split('\n')
    flag.close()

    #Preprocess
    _,_,_,scaler,num_desc=preprocess(train,serviceData,flagData)  

    test=pd.read_csv(data_dir+'/test.csv',header=None)
    test_df,y_test,y_test_types,_,_=preprocess(test,serviceData,flagData,is_train=False,scaler=scaler, num_desc=num_desc)
    x_test=test_df.values
    
    return x_test,y_test

def get_model(args,data_dim):
    layers=[]
    for i in range(0,args.num_layers):
        #Halved for each succeeding layer size
        layers.append(int(args.max_hid_size*0.5**(i)))
    layers.append(args.l_dim)
    
    model_config={
        'd_dim':data_dim,
        'layers':layers
    } 
    model=AE(model_config)
    return model

def get_model_preds(model,loader):
    model.eval()
    with torch.no_grad():
        pred=[]
        # loss=[]
        for batch in loader:
            target = batch.type(torch.float32)

            outputs = model(target)
            # batch_error = model.compute_loss(outputs, target)

            pred.append(outputs['output'].cpu().detach().numpy())
            # loss.append(batch_error.item())

        pred=np.concatenate(pred)
    return pred

def eval_perf(args,device,weight_dir):
    #Load Data
    x_test,y_test=load_data()
    
    #Get Model
    model=get_model(args,x_test.shape[1])
    model.to(device)
    
    with open(weight_dir+"/nsl_{}_{}_{}_{}.pt".format(args.l_dim,args.epoch,args.batch_size,args.seed), "rb") as f:
        best_model = torch.load(f)
        
    model.load_state_dict(best_model["best_weights"])
    
    #Load mean,stddev,th
    threshold_file = open(weight_dir+"/nsl_{}_{}_{}_{}.th".format(args.l_dim,args.epoch,args.batch_size,args.seed), 'r')
    threshold = threshold_file.read().split('\n')
    mean=float(threshold[0])
    stddev=float(threshold[1])
    z_th=float(threshold[2])
    
    x_test_cuda = torch.from_numpy(x_test).float().to(device)
    test_sampler = SequentialSampler(x_test_cuda)
    test_dataloader = DataLoader(x_test_cuda, sampler=test_sampler)
    
    #Run test data through model
    pred_test=get_model_preds(model,test_dataloader)

    #L2 Distance & Standardization
    test_dist=np.mean(np.square(x_test-pred_test),axis=1)
    test_dist_standardized=(test_dist-mean)/stddev

    #Prediction with Threshold
    y_pred=np.zeros_like(y_test)
    y_pred[test_dist_standardized>z_th]=1

    #Performance Evaluation
    test_auc,_,_=make_roc(test_dist,y_test,ans_label=ATK)
    accuracy,precision,recall,f_score = aprf(y_test,y_pred,pos_label=ATK, average='binary')
    test_fpr=fpr(y_test,y_pred)
    test_mcc=mcc(y_test,y_pred)
    print("FPR {:.5f}, MCC {:.5f}".format(test_fpr,test_mcc))
    
    #Confusion Matrix
    # print(confusion_matrix(y_test,y_pred))

    #Write to Log
    log_name=f"perf_results/nsl_{args.max_hid_size}_{args.num_layers}.csv"
    if not os.path.isfile(log_name):
         with open(log_name, "a") as myfile:
             myfile.write("max_hid_size,num_layers,l_dim,epoch,batch,z_th,auc,acc,p,r,f,fpr,mcc,seed\n")
            
    with open(log_name, "a") as myfile:
        #L2
        myfile.write(f"{args.max_hid_size},{args.num_layers},{args.l_dim},")
        myfile.write(f"{args.epoch},{args.batch_size},")
        myfile.write("{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},".format(z_th,test_auc,accuracy,precision,recall,f_score))
        #Added (210201) - FPR, MCC
        myfile.write("{:.5f},{:.5f},".format(test_fpr,test_mcc))
        ##
        myfile.write(f"{args.seed}\n")

if __name__ == "__main__":
    # Argument Setting
    parser = argparse.ArgumentParser()

    #Random Seed (For Reproducibility)
    parser.add_argument("--seed", default=42, type=int,
                        help="random seed for reproductability")

    #Model Config
    parser.add_argument("--l_dim", default=10, type=int,
                        help="Latent dimension size")
    parser.add_argument("--num_layers", default=2, type=int,
                        help="number of hidden layers for each encoder,decoder")
    parser.add_argument("--max_hid_size", default=64, type=int,
                        help="Biggest Hid Size in Encoder,decoder")


    #Train Params
    parser.add_argument("--epoch", default=10, type=int,
                        help="number of epochs")
    parser.add_argument("--batch_size", default=8192, type=int,
                        help="batch size")

    args = parser.parse_args()
    
    # Fix seed
    set_seed(args.seed)
    device = torch.device('cuda:0')
    
    weight_dir=f"weights/nsl_{args.num_layers}_{args.max_hid_size}"
    
    if not os.path.exists("perf_results"):
        os.makedirs("perf_results")
        
    eval_perf(args,device,weight_dir)
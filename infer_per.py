# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE
import torch
from datasets.read_data import  read_partition_data
import os 
import numpy as np 
import json
import argparse
import time
from utils.infer_utils import init_seed,prepare_infer_model, test

parser = argparse.ArgumentParser()

parser.add_argument('--p',
                    help='path;',
                    type=str,default='outputs/cifar10/resnet18_adapter')


parser.add_argument('--dirichlet_alpha',
                        type=float,
                        default=1)
parser.add_argument('--model',
                    type=str,default='resnet18')
parser.add_argument('--f',
                    help='path;',
                    type=str,default='')
parser.add_argument('--dataset',
                    help='path;',
                    type=str,default='cifar10')
parser.add_argument('--seed',
                    help='random seed for reproducibility;',
                    type=int,
                        default=1)
parser.add_argument('--num_clients',
                        type=int,
                        default=20)
args = parser.parse_args()
init_seed(args.seed)



clients, groups, _, test_data , _, _ = read_partition_data(args.dataset, args.num_clients, args.dirichlet_alpha, batch_size= 16384, test_batch_size=16384 , server_batch_size=50000, data_dir='data')
train_users = clients['train_users']
test_users = clients['test_users']

output_fname= 'per_inference_{}.json'.format(args.f)
adapter_model, vanilla_model = prepare_infer_model(args)
folder = args.p
if len(args.f)==0:
    subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
else:
    subfolders =  [os.path.join(folder, args.f)]
print("len",  len(subfolders), subfolders)

output_folder= os.path.join(folder, 'infer')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

results= dict()
print("will save to file",os.path.join(output_folder, output_fname) )

for PATH in subfolders:  
    start= time.time()
    print("start", os.path.basename(PATH))
    one_run_results= dict()
    if 'adapter' in os.path.basename(PATH):
        model = adapter_model
    else:
        model = vanilla_model

    # test the personalized models 
    test_per_acces = []
    num_test_samples_all_clients= []
    for u in range(args.num_clients):
        fname= os.path.join(PATH,'permodel_{}.ckpt'.format(u))
        try:
            stat_dict = torch.load(fname)
        except:
            
            print(fname, "model not exist") 
            continue
        
        model.load_state_dict(stat_dict['state_dict'])
        _, u_test_per_acc   = test(model, test_data[u]['dataloader'], args.dataset)
        test_per_acces.append(u_test_per_acc)
        num_test_samples_all_clients.append(len( test_data[u]['indices']))
        
    if len(test_per_acces)>0:
        one_run_results['per']= test_per_acces
        one_run_results['per_mean']  = round(np.average(test_per_acces),2)
        one_run_results['per_weighted_mean']  = round(np.average(test_per_acces,weights=num_test_samples_all_clients),2)
        one_run_results['per_std']  = round(np.array(test_per_acces).std(),2) 
    

    results[PATH]= one_run_results

    with open(os.path.join(output_folder, output_fname), 'w') as f:
        json.dump(results, f)
    print("time spent on one run", PATH , time.time()-start)
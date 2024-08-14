# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE
import torch 
import os 
import numpy as np 
import json
import argparse
import time
from utils.infer_utils import init_seed, get_cifar100_dataloader,  test_all_loaders,prepare_infer_model, corruptions,get_cifar_dataloader


parser = argparse.ArgumentParser()

parser.add_argument('--p',
                    help='path;',
                    type=str,default='outputs/cifar10/resnet18_adapter')

parser.add_argument('--f',
                    help='path;',
                    type=str,default='')
parser.add_argument('--dataset',
                    help='path;',
                    type=str,default='cifar10')
parser.add_argument('--model',
                    type=str,default='resnet18')
parser.add_argument('--seed',
                    help='random seed for reproducibility;',
                    type=int,
                        default=1)
parser.add_argument('--num_clients',
                        type=int,
                        default=20)
                            
parser.add_argument('--corrupt',
                        action='store_true')
args = parser.parse_args()
init_seed(args.seed)
output_fname= 'inference_{}.json'.format(args.f)
adapter_model, vanilla_model = prepare_infer_model(args)

folder = args.p
if len(args.f)==0:
    subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
else:
    subfolders =  [os.path.join(folder, args.f)]
print("len",  len(subfolders), subfolders)
# all_name = ['cifar10.1','natural' ]+ corruptions
if args.corrupt: 
    all_name = corruptions
else:
    if args.dataset=='cifar10':
        all_name = ['cifar10.1','natural']
    else:
        all_name = ['natural'] 

loader_dict =  dict()
for cname in all_name:
    if args.dataset=='cifar10':
        loader_dict[cname] = get_cifar_dataloader(cname = cname)
    elif args.dataset=='cifar100':
        loader_dict[cname] = get_cifar100_dataloader(cname = cname)
    elif args.dataset=='oh':
        from datasets.read_data import read_office_home_data
        args.num_clients=4
        _, _, _, cli_test_data, _,test_dataloader = read_office_home_data(args.num_clients, batch_size= 2048,test_batch_size=2048, server_batch_size=1024)
        loader_dict[cname] = test_dataloader      
    elif args.dataset=='chexpert':
        from datasets.read_data import read_chexpert_data
        _, _, _, _, _,test_dataloader = read_chexpert_data(args.num_clients, batch_size= 1024,test_batch_size=50000, server_batch_size=1024)
        loader_dict[cname] = test_dataloader
if args.dataset=='oh':
    all_name = ['natural',"art", "clipart", "product", "real_world"] 
    domains_name = ["art", "clipart", "product", "real_world"]
    for domain_id in range(args.num_clients):
        loader_dict[domains_name[domain_id]]= cli_test_data[domain_id]['dataloader']

       
infer_folder='infer_corrupt' if args.corrupt else 'infer'
output_folder= os.path.join(folder, infer_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

results= dict()
print("will save to file",os.path.join(output_folder, output_fname) )


for PATH in subfolders:
    print("start", os.path.basename(PATH))
    start= time.time()
    one_run_results= dict()

    if 'adapter' in os.path.basename(PATH):
        model = adapter_model
    else:
        model = vanilla_model

    fname= os.path.join(PATH,'gmodel.ckpt')
    # test the global model 
    try:
        stat_dict = torch.load(fname)
        load_epoch = stat_dict['epoch']
        one_run_results['epoch'] = load_epoch

        model.load_state_dict(stat_dict['state_dict'])
        print("model load", fname)
        global_acc_all = test_all_loaders(model,all_name, loader_dict,dataset= args.dataset ,cor=args.corrupt)
        for key,value in global_acc_all.items():
            one_run_results['global_'+key]  = global_acc_all[key]
    except:
        print(fname, "model not exist")
        
    # test the personalized models 
    per_acc_all=dict()
    for u in range(args.num_clients):
        fname= os.path.join(PATH,'permodel_{}.ckpt'.format(u))
        try:
            stat_dict = torch.load(fname)
        except:
            print(fname, "model not exist")
            continue
        model.load_state_dict(stat_dict['state_dict'])
        per_acc_all = test_all_loaders(model,all_name, loader_dict,dataset= args.dataset , cor=args.corrupt)
        for key,value in per_acc_all.items():
            if 'per_'+key in one_run_results:
                one_run_results['per_'+key].append(value)
            else:
                one_run_results['per_'+key]=[value]
    if len(per_acc_all)>0:
        for key,value in per_acc_all.items():
            one_run_results['per_'+key+'_mean']  = round(np.average(one_run_results['per_'+key]),2)
            one_run_results['per_'+key+'_std']  = round(np.array(one_run_results['per_'+key]).std(),2) 

    results[PATH]= one_run_results
    with open(os.path.join(output_folder, output_fname), 'w') as f:
        json.dump(results, f)
    print("time spent on one run", PATH , time.time()-start)
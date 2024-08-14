# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

import torch 
import json
import os
import numpy


def read_partition_data(data_name,num_clients,alpha,  batch_size, test_batch_size , server_batch_size, 
            shard_per_user= None, data_dir ='data',drop_last=False,img_resolution=None,kd_data_fraction=1):
    
    if data_name == 'oh': 
        return read_office_home_data(num_clients, batch_size, test_batch_size , server_batch_size,kd_data_fraction) # feature noniid; don't need alpha
    elif data_name == 'chexpert':
        return read_chexpert_data(num_clients, batch_size, test_batch_size , server_batch_size,img_resolution,kd_data_fraction,alpha)
    else:
        from datasets.prepare_data import get_dataset, cifar_noniid,svhn_noniid

        train_dataset=get_dataset(
                    data_name, data_dir, split="train",img_resolution=img_resolution
                )
        test_dataset= get_dataset(
            data_name, data_dir, split="test",img_resolution=img_resolution
        )

        if data_name == 'svhn':
            dict_users_train, server_idx, cnts_dict_train, train_proportions_dict = svhn_noniid(train_dataset, num_clients, user_split=0.9, alpha=alpha , shard_per_user=shard_per_user, proportions_dict =None) # 45000 for client, 5000 for server
            dict_users_test, _, cnts_dict_test, _ = svhn_noniid(test_dataset, num_clients, alpha=alpha ,  shard_per_user=shard_per_user, proportions_dict =train_proportions_dict) # 10000 for server 

        else:
            if  'cifar10.1' in data_name or 'CIFAR-10-C' in data_name:
                outfile = "data/cifar10_train_proportions_dict.npy"
                train_proportions_dict = numpy.load(outfile, allow_pickle=True).item()
                print(train_proportions_dict)
                print("load", 'data/cifar10_train_proportions_dict.npy')
            else:
                train_proportions_dict=None
            dict_users_train, server_idx, cnts_dict_train, train_proportions_dict = cifar_noniid(train_dataset, num_clients, user_split=0.9, alpha=alpha , shard_per_user=shard_per_user, proportions_dict =train_proportions_dict) # 45000 for client, 5000 for server
            dict_users_test, _, cnts_dict_test, _ = cifar_noniid(test_dataset, num_clients, alpha=alpha ,  shard_per_user=shard_per_user, proportions_dict =train_proportions_dict) # 10000 for server 
            

        train_data = {}
        test_data = {}
        for user_id in range(num_clients):  
     
            train_data.update({user_id: {'dataloader':  torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, num_workers=2, pin_memory=False,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler( dict_users_train[user_id]),drop_last=drop_last), 
                            'indices': dict_users_train[user_id]}})
            
            test_data.update({user_id: {'dataloader':  torch.utils.data.DataLoader(test_dataset, batch_size= test_batch_size, num_workers=2, pin_memory=False,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler( dict_users_test[user_id])), 
                            'indices': dict_users_test[user_id]}})
        

        clients = {
            'train_users': list(train_data.keys()),
            'test_users': list(test_data.keys())
        }
        val_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= server_batch_size, num_workers=2, pin_memory=False,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(server_idx))
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size= server_batch_size, num_workers=2, pin_memory=False)

        num_kd_samples= int(len(server_idx)*kd_data_fraction)
        kd_idx= [server_idx[i] for i in range(num_kd_samples)]
        print("kd_idx len", len(kd_idx), "out of", len(server_idx))
        kd_dataloader= torch.utils.data.DataLoader(train_dataset, batch_size= server_batch_size, num_workers=2, pin_memory=False,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(kd_idx))
        return clients, kd_dataloader, train_data, test_data, val_dataloader,test_dataloader


def read_chexpert_data(num_clients,batch_size, test_batch_size , server_batch_size,img_resolution=None,kd_data_fraction=1,alpha=1):
    from datasets.chexpert import CheXpert
    chexpert= CheXpert(root ='data', num_clients= num_clients,alpha=alpha) 
    clients, kd_dataloader, train_data, test_data, val_dataloader,test_dataloader = chexpert.get_data_loaders(batch_size, test_batch_size , server_batch_size,img_resolution,kd_data_fraction)
    return clients, kd_dataloader, train_data, test_data, val_dataloader,test_dataloader

def read_office_home_data(num_clients,batch_size, test_batch_size , server_batch_size,kd_data_fraction=1):
    from datasets.office_home import OfficeHome
    oh= OfficeHome()
    assert len(oh.domains) == num_clients
    clients, kd_dataloader, train_data, test_data, val_dataloader,test_dataloader = oh.get_data_loaders(batch_size, test_batch_size , server_batch_size,kd_data_fraction)
    return clients, kd_dataloader, train_data, test_data, val_dataloader,test_dataloader


def read_natural_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)

        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = {
        'train_users': list(train_data.keys()),
        'test_users': list(test_data.keys())
    }

    return clients, groups, train_data, test_data


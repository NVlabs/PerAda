# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

import numpy as np
import csv
from PIL import Image
import os 
import copy
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import json

class DatasetWrapper(Dataset):

    def __init__(self,  items_list, transform=None):
        self.items_list = items_list
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        
        image_name = self.items_list[index][0]
        image = Image.open(image_name).convert('RGB')
        label =  self.items_list[index][1]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.items_list)



class CheXpert(object):
    dataset_dir = "CheXpert-v1.0-small"
    class_names = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
    cname2class={"Cardiomegaly":0, "Edema":1, "Consolidation":2, "Atelectasis":3,"Pleural Effusion":4}

    nnClassCount = 5
    url = 'http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip'
    
    def __init__(self, root ='data', num_clients=20 ,alpha=1):

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.root= root 
        self.num_clients= num_clients
      
     
      
        self.local_train_list, self.local_test_list ,self.server_val_all  = self.get_non_iid_data_train_test_val(
            dataset_dir= self.dataset_dir,  num_clients= num_clients, test_ratio=0.1,alpha=alpha)
        self.server_test_all = sum(self.local_test_list, [])



    def get_non_iid_data_train_test_val(self, dataset_dir, num_clients, test_ratio=0.1,alpha=1):
        train_file = os.path.join(dataset_dir, "train.csv")
        img_label_items = self.read_data_from_csv( train_file, policy='ones')
        total_number_samples= len(img_label_items)
        total_index_list= list(range(total_number_samples))
        with open(os.path.join('datasets/chexpert_partition_{}.json'.format(alpha))) as json_file:
            data = json.load(json_file)
     
        train_data_items_clientlist =[]
        test_data_items_clientlist =[]

        for u in range(num_clients):
            local_index_list = data["{}".format(u)]

            local_num_samples= len(local_index_list)
            test_idx= np.random.choice(list(set(range(local_num_samples))), int(local_num_samples*test_ratio), replace=False)
            train_idx = [i for i in range(local_num_samples) if i not in test_idx]
            
            train_items= [img_label_items[i] for  i in train_idx]
            test_items=[img_label_items[i] for  i in test_idx]
            

            print("client", u, "train", len(train_items), "test", len(test_items))
            train_data_items_clientlist.append(train_items)
            test_data_items_clientlist.append(test_items)
        
        val_data_items_list= [img_label_items[i] for  i in data["{}".format(-1)]] 
        print("server val", len (val_data_items_list))
        
        return train_data_items_clientlist, test_data_items_clientlist,val_data_items_list


    def get_data_loaders (self, batch_size = 64, test_batch_size =64, server_batch_size=128,img_resolution=None,kd_data_fraction=1):
        num_clients = self.num_clients
        train_data = {}
        test_data = {}

        MEAN_IMAGENET = (0.485, 0.456, 0.406)
        STD_IMAGENET = (0.229, 0.224, 0.225)
        image_size = 64 if img_resolution is None else img_resolution
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.05, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
        ])

        all_train_samples= 0
        all_test_samples=0
        for user_id in range(num_clients):  
            train_data.update({user_id: {'dataloader':  torch.utils.data.DataLoader(DatasetWrapper(self.local_train_list[user_id],train_transform), batch_size= batch_size, num_workers=2, pin_memory=True,
                                shuffle =True,persistent_workers=True), 
                            'indices': self.local_train_list[user_id]  }})

            test_data.update({user_id: {'dataloader':  torch.utils.data.DataLoader(DatasetWrapper(self.local_test_list[user_id],test_transform), batch_size= test_batch_size, num_workers=2, pin_memory=True,
                            shuffle =True,persistent_workers=True), 
                            'indices': self.local_test_list[user_id]}})

            all_train_samples+=len(self.local_train_list[user_id])
            all_test_samples+=len(self.local_test_list[user_id])
        
        print("all samples train and test", all_train_samples, all_test_samples)


        clients = {
            'train_users': list(train_data.keys()),
            'test_users': list(test_data.keys())
        }
        val_dataloader = torch.utils.data.DataLoader(DatasetWrapper(self.server_val_all,test_transform), batch_size= server_batch_size, num_workers=2, pin_memory=True,
                                                    persistent_workers=True)


        kd_idx = np.random.choice(list(set(range(len(self.server_val_all)))) , int(len(self.server_val_all)*kd_data_fraction) , replace=False) 
        print("kd_idx len", len(kd_idx), "out of", len(self.server_val_all))

        kd_dataloader = torch.utils.data.DataLoader(DatasetWrapper(self.server_val_all,test_transform), batch_size= server_batch_size, num_workers=2, pin_memory=True,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(kd_idx),persistent_workers=True)

        test_dataloader = torch.utils.data.DataLoader(DatasetWrapper(self.server_test_all,test_transform), batch_size= server_batch_size, num_workers=2, 
                                                pin_memory=True,persistent_workers=True)
        return clients, kd_dataloader, train_data, test_data, val_dataloader,test_dataloader

    def read_data_from_csv(self ,path_list ,policy='ones' ):
        
        items=[]
        with open(path_list, "r") as f:
            print(path_list)
            csvReader = csv.reader(f)
            next(csvReader, None)
            
            for line in csvReader:
                image_name = line[0]
                npline = np.array(line)
                idx = [7, 10, 11, 13, 15]
                label = list(npline[idx])
                for i in range(self.nnClassCount):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == 'diff':
                                if i == 1 or i == 3 or i == 4:  # Atelectasis, Edema, Pleural Effusion
                                    label[i] = 1                    # U-Ones
                                elif i == 0 or i == 2:          # Cardiomegaly, Consolidation
                                    label[i] = 0                    # U-Zeroes
                            elif policy == 'ones':              # All U-Ones
                                label[i] = 1
                            else:
                                label[i] = 0                    # All U-Zeroes
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                
                items.append( (os.path.join(self.root,image_name), label ) )
                # lable:[0, 0, 0, 1, 1]
        return items


    def read_personalized_iid_data_train_test_val(self,dataset_dir, num_clients, split, test_ratio =0.2):
    
        train_file = os.path.join(dataset_dir, "train.csv")
        img_label_items = self.read_data_from_csv( train_file, policy='ones')
        total_number_samples= len(img_label_items)
        total_index_list= list(range(total_number_samples))
        n = int(total_number_samples/num_clients)            
        local_index_list = [total_index_list[i:i + n] for i in range(0, len(total_index_list), n)] 

        if len(local_index_list) >num_clients:
            local_index_list[-2]= copy.deepcopy(local_index_list[-2]+local_index_list[-1])
            local_index_list.pop(-1)

        train_data_items_clientlist =[]
        test_data_items_clientlist =[]
        val_data_items_clientlist =[]

        for u in range(num_clients):
            local_num_samples= len(local_index_list[u])
            test_idx= np.random.choice(list(set(range(local_num_samples))), int(local_num_samples*test_ratio), replace=False)
            train_idx = [i for i in range(local_num_samples) if i not in test_idx]
            
            train_items= [img_label_items[i] for  i in train_idx]

            real_test_idx=  test_idx[:int(len(test_idx)/2)]
            val_idx= test_idx[int(len(test_idx)/2):]

            test_items=[img_label_items[i] for  i in real_test_idx]
            val_items=[img_label_items[i] for  i in val_idx]
            
            print("client", u, "train", len(train_items), "test", len(test_items),"val", len(val_items))
            train_data_items_clientlist.append(train_items)
            test_data_items_clientlist.append(test_items)
            val_data_items_clientlist.append(val_items)
        
        return train_data_items_clientlist, test_data_items_clientlist,val_data_items_clientlist
    
    def label_stats(self, items_list):
        label_dis= {0:0, 1:0, 2:0, 3:0, 4:0}
        for item in items_list:
            img, label =item 
            for i in range(5):
                if label[i]>0:
                    label_dis[i]+=1

        return label_dis


 

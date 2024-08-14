# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

import os 
import copy 
import numpy as np
import csv 
import collections
import random
import torch 



def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


init_seed(1)

def read_data_from_csv(root, path_list ,policy='ones' ):
        image_names = []
        labels = []
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
                for i in range(5):
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
                
                items.append( (os.path.join(root,image_name), label ) )
                # lable:[0, 0, 0, 1, 1]
        return items

 
root='data'
train_file=os.path.join(root, 'CheXpert-v1.0-small/train.csv')
alpha=0.3 # non-iid degree
num_users=20


img_label_items = read_data_from_csv(root, train_file, policy='ones')
total_number_samples= len(img_label_items)
total_index_list= list(range(total_number_samples))

print(len(total_index_list))


multilabel_dist = {}
count_list=[]

for idx in range(len(img_label_items)):
    item=img_label_items[idx]
    img, label =item 
    count=0
    candidate=[]
    for i in range(5):
        if label[i]>0:
            count+=1
            candidate.append(i)
    key = str(candidate)
    if key in multilabel_dist:
        multilabel_dist[key].append(idx)
    else:
        multilabel_dist[key]=[idx]
    count_list.append(count)



new_od = collections.OrderedDict(sorted(multilabel_dist.items(), key=lambda x: len(x[1]),reverse=True))
print(new_od.keys())

import copy
combined_classes= dict()
additional_class_key='else'
combined_classes[additional_class_key]=[]
for key, value in new_od.items():
    if len(value)< 2000:
        combined_classes[additional_class_key].extend(value)
    else:
        combined_classes[key]= copy.deepcopy(value)
    

print(combined_classes.keys())
value_lens= [len(value) for value in combined_classes.values()]
print( value_lens, sum(value_lens))
print(len(value_lens))



num_classes=len(combined_classes.keys())
_lst_sample=2
user_split=0.9

total_number_samples=len(img_label_items)
sample_per_client= int(total_number_samples/num_users)
num_user_data=int(user_split*total_number_samples)


least_idx = np.zeros((num_users, num_classes, _lst_sample), dtype=np.int)
classes_keys_list= list(combined_classes.keys())
for i in range(num_classes):
    key = classes_keys_list[i]
    idx_i = np.random.choice( combined_classes[key], num_users*_lst_sample, replace=False)
    least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
least_idx = np.reshape(least_idx, (num_users, -1))
least_idx_set = set(np.reshape(least_idx, (-1)))

print(least_idx.shape)

server_idx = np.random.choice(list(set(range(total_number_samples))-least_idx_set), total_number_samples-num_user_data, replace=False)
local_idx = np.array([i for i in range(total_number_samples) if i not in server_idx and i not in least_idx_set])


print(len(server_idx))
print(len(local_idx))

dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
idx_batch = [[] for _ in range(num_users)]
N = total_number_samples
proportions_dict= {k: None for k in range(num_classes)}
# for each class in the dataset
for k in range(num_classes):
    key = classes_keys_list[k]
    idx_k_select = combined_classes[key]

    idx_k =[]
    for id in idx_k_select:
        if id in local_idx:
            idx_k.append(id)

    np.random.shuffle(idx_k)
   
    proportions = np.random.dirichlet(np.repeat(alpha, num_users))
    proportions_dict[k] = proportions
    ## Balance
    proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])
    proportions = proportions/proportions.sum()
    proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
    idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
    print("class" , k, [len(idx.tolist()) for idx in np.split(idx_k,proportions)] )

for j in range(num_users):
    np.random.shuffle(idx_batch[j])
    dict_users[j] = idx_batch[j]  
    dict_users[j] = np.concatenate((dict_users[j], least_idx[j]), axis=0)          


dict_users[-1]= list(server_idx)


import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


with open(os.path.join('chexpert_partition_{}.json'.format(alpha)), 'w') as f:
    json.dump(dict_users, f, cls=NpEncoder)

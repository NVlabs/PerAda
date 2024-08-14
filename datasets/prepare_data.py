# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

from torchvision import datasets, transforms
import os 
import numpy as np
import torch

def _get_transform(data_name, is_training,img_resolution=None):
    if data_name == "cifar10" or data_name == "cifar10.1" or "CIFAR-10-C" in data_name:
        img_resolution=  img_resolution if img_resolution is not None else 32

        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        )
        if is_training:
            transform = transforms.Compose(
                [   
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((32, 32), 4),
                    transforms.Resize(img_resolution),
                    transforms.ToTensor(), 
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose([transforms.Resize(img_resolution),transforms.ToTensor(),normalize])

    elif data_name == "cifar100":
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )
        img_resolution=  img_resolution if img_resolution is not None else 32
        if is_training:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((32, 32), 4),
                    transforms.Resize(img_resolution),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose([transforms.Resize(img_resolution),transforms.ToTensor(),normalize])

    elif data_name == "stl10":
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )
        img_resolution=  img_resolution if img_resolution is not None else 96
        if is_training:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((96, 96), 4),
                    transforms.Resize((img_resolution, img_resolution)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose([
                transforms.Resize((img_resolution, img_resolution)),
                transforms.ToTensor(),
                normalize
                ])
    else:
        raise NotImplementedError


    print("img_resolution", img_resolution)

    return transform


def _get_cifar(data_name, root, split, transform, target_transform, download):
    is_train = split == "train"
    # decide normalize parameter.
    
    if data_name == "cifar10":
        dataset_loader = datasets.CIFAR10(root=root,
                train=is_train,
                transform=transform,
                target_transform=target_transform,
                download=download,
            )
    elif data_name == "cifar100":
        dataset_loader = datasets.CIFAR100(root=root,
                train=is_train,
                transform=transform,
                target_transform=target_transform,
                download=download,
            )
    elif data_name == "cifar10.1":
        from datasets.cifar import CIFAR101
        dataset_loader = CIFAR101(
                    root,
                     transform=transform
                )
    elif "CIFAR-10-C" in data_name:
        cname = data_name.split("@")[1]
        from datasets.cifar import CIFAR10C
        dataset_loader = CIFAR10C(
                   os.path.join('data', 'CIFAR-10-C'),  cname,  # hardcode the path..
                     transform=transform
                )
    else:
        raise NotImplementedError(f"invalid data_name={data_name}.")


    return dataset_loader



def _get_stl10(data_name, root, split, transform, target_transform, download):
    # right now this function is only used for unlabeled dataset.
    is_train = split == "train"

    if is_train:
        split = "train+unlabeled" # 105000 data

    return datasets.STL10(
        root=root,
        split=split,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def get_dataset(
    data_name,
    datasets_path,
    split="train",
    transform=None,
    target_transform=None,
    download=True,
    img_resolution=None,
):
    # create data folder if it does not exist.
    root = os.path.join(datasets_path, data_name)
    is_train = split == "train"
    transform = _get_transform(data_name=data_name, is_training= is_train, img_resolution= img_resolution)
    if data_name == "cifar10" or data_name == "cifar100" or data_name == "cifar10.1" or "CIFAR-10-C" in data_name:
        return _get_cifar(data_name, root, split, transform, target_transform, download)
    elif data_name == "stl10":
        return _get_stl10(data_name, root, split, transform, target_transform, download)
    else: 
        raise NotImplementedError


def cifar_noniid(dataset, num_users, user_split=1, alpha =None, shard_per_user=None, proportions_dict=None):
    """
    Sample non-I.I.D client data from CIFAR dataset
    :param dataset:
    :param num_users:
    :return:
    """
    
    total_number_samples= len(dataset)
    num_classes = len(np.unique(dataset.targets))
   

    num_user_data = int( user_split * total_number_samples)
    labels = np.array(dataset.targets)
    _lst_sample = 2
    if alpha is not None:
        method= 'dir'
    elif shard_per_user is not None:
        method= 'shard'
    
    print("cifar_noniid", total_number_samples,num_classes,method) 
    
    
    if method=="shard":
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        idxs_dict = {}
        server_idx = np.random.choice(list(set(range(total_number_samples))) , total_number_samples- num_user_data , replace=False)
        local_idx = [i for i in range(total_number_samples) if i not in server_idx]
        
        for i in local_idx:
            label = torch.tensor(dataset.targets[i]).item()
            if label not in idxs_dict.keys():
                idxs_dict[label] = []
            idxs_dict[label].append(i)

       
        shard_per_class = int(shard_per_user * num_users / num_classes)
        for label in idxs_dict.keys():
            x = idxs_dict[label]
            num_leftover = len(x) % shard_per_class
            leftover = x[-num_leftover:] if num_leftover > 0 else []
            x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
            x = x.reshape((shard_per_class, -1))
            x = list(x)

            for i, idx in enumerate(leftover):
                x[i] = np.concatenate([x[i], [idx]])
            idxs_dict[label] = x


        if proportions_dict is None:
            proportions_dict = list(range(num_classes)) * shard_per_class
            np.random.shuffle(proportions_dict)
            proportions_dict = np.array(proportions_dict).reshape((num_users, -1))

        # Divide and assign
        for i in range(num_users):
            rand_set_label = proportions_dict[i]
            rand_set = []
            for label in rand_set_label:
                idx = np.random.choice(len(idxs_dict[label]), replace=False)
                rand_set.append(idxs_dict[label].pop(idx))
            dict_users[i] = np.concatenate(rand_set)
    
    elif method == "dir":
        if proportions_dict is None:
            proportions_dict= {k: None for k in range(num_classes)}

        y_train = labels
        
        least_idx = np.zeros((num_users, num_classes, _lst_sample), dtype=int)
        for i in range(num_classes):
            idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
            least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
        least_idx = np.reshape(least_idx, (num_users, -1))
        
        least_idx_set = set(np.reshape(least_idx, (-1)))
        #least_idx_set = set([])
        
        server_idx = np.random.choice(list(set(range(total_number_samples))-least_idx_set), total_number_samples-num_user_data, replace=False)
        local_idx = np.array([i for i in range(total_number_samples) if i not in server_idx and i not in least_idx_set])
    
        print(len(server_idx), len(local_idx), len(least_idx_set) )
        N = y_train.shape[0]
        net_dataidx_map = {}
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        
        idx_batch = [[] for _ in range(num_users)]
        # for each class in the dataset
        for k in range(num_classes):
            idx_k_select = np.where(y_train == k)[0]
        
            idx_k =[]
            for id in idx_k_select:
                if id in local_idx:
                    idx_k.append(id)

            np.random.shuffle(idx_k)
            if proportions_dict[k] is not None:
                proportions = proportions_dict[k]
            else:
                proportions = np.random.dirichlet(np.repeat(alpha, num_users))
                proportions_dict[k] = proportions
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
           

        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            dict_users[j] = idx_batch[j]  
            dict_users[j] = np.concatenate((dict_users[j], least_idx[j]), axis=0)          

    # print(proportions_dict)
    cnts_dict = {}
    # with open("data_%d_u%d_%s.txt"%(num_data, num_users, method), 'w') as f:
    for i in range(num_users):
        dict_users[i] = [int(index) for index in dict_users[i]]
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(num_classes)] )
        cnts_dict[i] = cnts
        # f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))  
        # print("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))  

    for i in range(num_users):
        dict_users[i]= list(dict_users[i])
    server_idx =list(server_idx)
    return dict_users, server_idx, cnts_dict, proportions_dict




def svhn_noniid(dataset, num_users, user_split=1, alpha =None, shard_per_user=None, proportions_dict=None):
    """
    Sample non-I.I.D client data from SVHN dataset
    :param dataset:
    :param num_users:
    :return:
    """
    
    total_number_samples= len(dataset)
    num_classes = len(np.unique(dataset.labels))
   

    num_user_data = int( user_split * total_number_samples)
    labels = np.array(dataset.labels)
    _lst_sample = 2
    if alpha is not None:
        method= 'dir'
    elif shard_per_user is not None:
        method= 'shard'
    
    print("svhn_noniid", total_number_samples,num_classes,method) 
    
    
    if method=="shard":
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        idxs_dict = {}
        server_idx = np.random.choice(list(set(range(total_number_samples))) , total_number_samples- num_user_data , replace=False)
        local_idx = [i for i in range(total_number_samples) if i not in server_idx]
        
        for i in local_idx:
            label = torch.tensor(dataset.labels[i]).item()
            if label not in idxs_dict.keys():
                idxs_dict[label] = []
            idxs_dict[label].append(i)

       
        shard_per_class = int(shard_per_user * num_users / num_classes)
        for label in idxs_dict.keys():
            x = idxs_dict[label]
            num_leftover = len(x) % shard_per_class
            leftover = x[-num_leftover:] if num_leftover > 0 else []
            x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
            x = x.reshape((shard_per_class, -1))
            x = list(x)

            for i, idx in enumerate(leftover):
                x[i] = np.concatenate([x[i], [idx]])
            idxs_dict[label] = x


        if proportions_dict is None:
            proportions_dict = list(range(num_classes)) * shard_per_class
            np.random.shuffle(proportions_dict)
            proportions_dict = np.array(proportions_dict).reshape((num_users, -1))

        # Divide and assign
        for i in range(num_users):
            rand_set_label = proportions_dict[i]
            rand_set = []
            for label in rand_set_label:
                idx = np.random.choice(len(idxs_dict[label]), replace=False)
                rand_set.append(idxs_dict[label].pop(idx))
            dict_users[i] = np.concatenate(rand_set)
    
    elif method == "dir":
        if proportions_dict is None:
            proportions_dict= {k: None for k in range(num_classes)}

        y_train = labels
        
        least_idx = np.zeros((num_users, num_classes, _lst_sample), dtype=int)
        for i in range(num_classes):
            idx_i = np.random.choice(np.where(labels==i)[0], num_users*_lst_sample, replace=False)
            least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
        least_idx = np.reshape(least_idx, (num_users, -1))
        
        least_idx_set = set(np.reshape(least_idx, (-1)))
        #least_idx_set = set([])
        
        server_idx = np.random.choice(list(set(range(total_number_samples))-least_idx_set), total_number_samples-num_user_data, replace=False)
        local_idx = np.array([i for i in range(total_number_samples) if i not in server_idx and i not in least_idx_set])
    
        print(len(server_idx), len(local_idx), len(least_idx_set) )
        N = y_train.shape[0]
        net_dataidx_map = {}
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        
        idx_batch = [[] for _ in range(num_users)]
        # for each class in the dataset
        for k in range(num_classes):
            idx_k_select = np.where(y_train == k)[0]
        
            idx_k =[]
            for id in idx_k_select:
                if id in local_idx:
                    idx_k.append(id)

            np.random.shuffle(idx_k)
            if proportions_dict[k] is not None:
                proportions = proportions_dict[k]
            else:
                proportions = np.random.dirichlet(np.repeat(alpha, num_users))
                proportions_dict[k] = proportions
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/num_users) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
           

        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            dict_users[j] = idx_batch[j]  
            dict_users[j] = np.concatenate((dict_users[j], least_idx[j]), axis=0)          

    cnts_dict = {}
    
    for i in range(num_users):
        dict_users[i] = [int(index) for index in dict_users[i]]
        labels_i = labels[dict_users[i]]
        cnts = np.array([np.count_nonzero(labels_i == j ) for j in range(num_classes)] )
        cnts_dict[i] = cnts
    

    for i in range(num_users):
        dict_users[i]= list(dict_users[i])
    server_idx =list(server_idx)
    return dict_users, server_idx, cnts_dict, proportions_dict




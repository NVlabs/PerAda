# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

import os 
from torchvision import  datasets
from PIL import Image
import numpy as np 

class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root :str, name :str,
                 transform=None, target_transform=None):
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        self.targets = self.targets.astype(int)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)



class CIFAR101(datasets.VisionDataset):
    def __init__(self, root :str,
                 transform=None, target_transform=None):
        super(CIFAR101, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )

       
        data_path = os.path.join(root, 'cifar10.1_v6_data.npy')
        target_path = os.path.join(root,  'cifar10.1_v6_labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        self.targets = self.targets.astype(int)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)

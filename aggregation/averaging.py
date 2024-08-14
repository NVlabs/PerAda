# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

import torch

def FedAveraging(state_dict_lst , weights= [] , update_layer_name= None):
    if update_layer_name is not None: # only aggregatate update_layer_name 
        print("FedAveraging update",update_layer_name)
        def is_update(name):
            for update_name in update_layer_name:
                if update_name in name:
                    return True
            return False

    w_avg = {}
    for k in state_dict_lst[0].keys():
        w_avg[k] = torch.zeros_like(state_dict_lst[0][k])
      

    for k in w_avg.keys():
        if update_layer_name is not None:
            if is_update(k)==False:
                w_avg[k]= state_dict_lst[0][k] 
                continue
        
        params = [s[k] for s in state_dict_lst]
        w_avg[k] = sum(1 * t for t in params)   
        w_avg[k] = torch.div(w_avg[k], len(state_dict_lst))     
    return w_avg

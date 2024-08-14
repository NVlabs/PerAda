# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file: https://github.com/facebookresearch/FL_partial_personalization/blob/main/LICENSE


from baseline_constants import DICT_FOR_CLASSES 
import torch
import torchvision.models
from torchinfo import summary
from .resnet_bn_utils import ResNetBN

class CIFARResNetBN(ResNetBN):
    def __init__(self, pretrained, num_classes=10, model='resnet18',save_activations=False):
        if 'resnet34' in model:
            layers = (3, 4, 6, 3)
        elif 'resnet18' in model:
            layers = (2, 2, 2, 2)
        else:
            raise NotImplementedError
        super().__init__(layers=layers, num_classes=num_classes, original_size=True,save_activations=save_activations)
        if pretrained:
            self.load_pretrained(model)
        self.update_layer_name = None

    @torch.no_grad()
    def load_pretrained(self, model):
        if 'resnet34' in model:
            pretrained_model = torchvision.models.resnet34(pretrained=True)

        else:
            pretrained_model = torchvision.models.resnet18(pretrained=True)
        
        pretrained_params = dict(pretrained_model.named_parameters())
        for name, param in self.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:  # do not load final layer weights
                param.copy_(pretrained_params[name])
        print('Successfully loaded weights from pretrained resnet')

    def print_summary(self, train_batch_size):
        device = next(self.parameters()).device
        print(summary(self, input_size=(train_batch_size, 3, 32, 32), device=device))
 

def get_adapter_model(net='resnet18', num_classes =10 , save_activations=False, per_dropout = 0.3):
    pretrained = True if 'pretrain' in net else False
    use_adapter = True if 'adapter' in net else False
   
    model = CIFARResNetBN(pretrained=pretrained, num_classes= num_classes ,  model=net,save_activations=save_activations)
    if use_adapter:
        # define adapter layer
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in layer.children():
                # each block is of type `ResidualBlock`
                block.add_adapters(per_dropout)
    else:
        # define output layer
        model.drop_o = torch.nn.Dropout(per_dropout)
    
    return model 

def get_model(dataset, net,save_activations=False,per_dropout = 0.3):
    dict_for_classes = DICT_FOR_CLASSES
    num_classes= dict_for_classes[dataset]
    if ('resnet18' in net) or ('resnet34' in net):
        model = get_adapter_model(net=net, num_classes =num_classes , save_activations=save_activations,per_dropout =per_dropout)
    else:
        raise NotImplementedError
   
    model=model.cuda()
    return model 


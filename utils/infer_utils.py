# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

import torch
import torchvision.transforms as transforms
import torch 
import os 
from models.model_utils import get_model
import numpy as np 
from torchvision import transforms, datasets
import random 
from torch.utils.data import DataLoader


def load_txt(path :str) -> list:
    return [line.rstrip('\n') for line in open(path)]



def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


corruptions = load_txt('data/CIFAR-10-C/corruptions.txt')

def get_cifar_dataloader (cname, datadir= 'data'):
    MEAN = [0.49139968, 0.48215841, 0.44653091]
    STD  = [0.24703223, 0.24348513, 0.26158784]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    if cname== 'natural':
        dataset = datasets.CIFAR10(os.path.join(datadir,'cifar10'),
                        train=False, transform=transform, download=True,
                    )
    elif cname== 'cifar10.1':
        from datasets.cifar import CIFAR101
        dataset = CIFAR101(
                    os.path.join(datadir, 'cifar10.1'),
                     transform=transform
                )
    else:
        from datasets.cifar import CIFAR10C
        dataset = CIFAR10C(
                    os.path.join(datadir, 'CIFAR-10-C'),  cname, 
                     transform=transform
                )
    loader = DataLoader(dataset, batch_size=50000,
                        shuffle=False, num_workers=4)
    return loader 


def get_cifar100_dataloader (cname, datadir= '../data'):
    MEAN = [0.49139968, 0.48215841, 0.44653091]
    STD  = [0.24703223, 0.24348513, 0.26158784]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    if cname== 'natural':
        dataset = datasets.CIFAR100(os.path.join(datadir,'cifar100'),
                        train=False, transform=transform, download=True,
                    )

    loader = DataLoader(dataset, batch_size=50000,
                        shuffle=False, num_workers=4)
    return loader 



def computeAUROC(dataGT, dataPRED, nnClassCount):
    # Computes area under ROC curve 
    # dataGT: ground truth data
    # dataPRED: predicted data
    outAUROC = []
    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()
    
    from sklearn.metrics import roc_auc_score

    for i in range(nnClassCount):
        try:
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
        except ValueError:
            pass
    return outAUROC


def test(model, testloader,dataset="cifar10"):
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    if dataset =='chexpert':
        y_prob = []
        y_true = []
        from sklearn import metrics
        import numpy as np 
        criterion= torch.nn.BCEWithLogitsLoss()
        sgmd =torch.nn.Sigmoid().cuda()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(testloader):
                x, y = x.to('cuda'), y.to('cuda')
                pred = model(x)
                loss = criterion(pred, y)
                y_prob.append(sgmd(pred).detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

                test_loss += loss.item()
                total += y.size(0)
                
            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            aurocMean = metrics.roc_auc_score(y_true, y_prob, average='macro')
            print('test Loss: %.3f | Acc:%.3f%%'%(test_loss/(batch_idx+1), aurocMean*100))
        return test_loss/len(testloader), aurocMean*100.

    else:
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(testloader):
                x, y = x.to('cuda'), y.to('cuda')
                pred = model(x)
                loss = criterion(pred, y)

                test_loss += loss.item()
                _, pred_c = pred.max(1)
                total += y.size(0)
                correct += pred_c.eq(y).sum().item()
            print('test Loss: %.3f | Acc:%.3f%% total: %d'%(test_loss/(batch_idx+1), 100.*correct/total, total))

        acc = 100.*correct/total
        return test_loss/len(testloader), acc


def test_all_loaders(model,all_name, loader_dict,dataset="cifar10", cor=False):
    acc_all =dict()
    acc_corruptions = 0

    for cname in all_name:
        testloader = loader_dict[cname]
        _, acc=  test(model, testloader,dataset)
        acc_all[cname]  = acc
        if cname in corruptions:
            acc_corruptions+= acc
    
    if cor:
        acc_all['cifar10-c'] = acc_corruptions *1.0 /len(corruptions)
    return acc_all


def prepare_infer_model(args):

    adapter_model = get_model (dataset= args.dataset  , net = "{}_adapter".format(args.model))
    vanilla_model =  get_model (dataset= args.dataset , net =  "{}".format(args.model))
    adapter_model = adapter_model.cuda()
    vanilla_model = vanilla_model.cuda()
    return adapter_model, vanilla_model


# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

# Part of the implementation is based on: https://github.com/epfml/federated-learning-public-code/blob/master/codes/FedDF-code/pcode/aggregation/noise_knowledge_transfer.py

import copy 
import torch 
from utils.stat_tracker import BestPerf 
import collections
import torch.nn.functional as F


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


def test(dataset, model, testloader , log=False):
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
            if log:
                print('test Loss: %.3f | Acc:%.3f%%'%(test_loss/(batch_idx+1), aurocMean*100))
        return test_loss/len(testloader), aurocMean*100.

    else:
        criterion= torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(testloader):
                x, y = x.to('cuda'), y.to('cuda')
                pred = model(x)
                loss = criterion(pred, y)

                test_loss += loss.item()
                _, pred_c = pred.max(1)
                total += y.size(0)
                correct += pred_c.eq(y).sum().item()
            if log:
                print('test Loss: %.3f | Acc:%.3f%%'%(test_loss/(batch_idx+1), 100.*correct/total))

        acc = 100.*correct/total
        return test_loss/len(testloader), acc



def divergence(student_logits, teacher, kd_KL_temperature=1, use_teacher_logits=True):
    divergence = F.kl_div(
        F.log_softmax(student_logits / kd_KL_temperature, dim=1),
        F.softmax(teacher / kd_KL_temperature, dim=1)
        if use_teacher_logits
        else teacher,
        reduction="batchmean",
    )  # forward KL
    return divergence

def check_early_stopping(
        model,
        model_ind,
        best_tracker,
        validated_perf,
        validated_perfs,
        perf_index,
        early_stopping_batches,
        log_fn=print,
        best_models=None,
    ):
    # update the tracker.
    best_tracker.update(perf=validated_perf, perf_location=perf_index)
    if validated_perfs is not None:
        validated_perfs[model_ind].append(validated_perf)

    # save the best model.
    if best_tracker.is_best and best_models is not None:
        best_models[model_ind] = copy.deepcopy(model)

    # check if we need the early stopping or not.
    if perf_index - best_tracker.get_best_perf_loc >= early_stopping_batches:
        log_fn(
            f"\tMeet the early stopping condition (batches={early_stopping_batches}): early stop!! (perf_index={perf_index}, best_perf_loc={best_tracker.get_best_perf_loc})."
        )
        return True
    else:
        return False


        
def Distillation(args, kd_server_smodel, kd_trainset , val_dataloader, local_model_list):
    kd_lr = args.kd_lr
    kd_eval_batches_freq= args.kd_eval_batches_freq 
    
    early_stopping_server_batches =args.early_stopping_server_batches
    total_n_server_pseudo_batches =args.total_n_server_pseudo_batches
    kd_KL_temperature =args.kd_KL_temperature

    
    optimizer_server_student = torch.optim.Adam(filter(lambda p: p.requires_grad, kd_server_smodel.parameters()),lr = kd_lr)
    scheduler_server_student = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_server_student,
        total_n_server_pseudo_batches,
        last_epoch=-1,
    )
    
    
    # get the init server perf.
    kd_server_smodel.eval()
    _, init_perf_on_val = test(args.dataset, kd_server_smodel,val_dataloader )
    print("init_perf_on_val: ", init_perf_on_val)

    server_best_tracker = BestPerf(best_perf=None, larger_is_better=True)
    best_models = [None]
    validated_perfs = collections.defaultdict(list)
    ## note that kd_trainset is replaced to  kd_dataloader 
    distillation_data_loader=kd_trainset 
   
    print("use kd dataset", args.kd_dataset)   

    data_iter = iter(distillation_data_loader)
    batch_id= 0



    while batch_id < total_n_server_pseudo_batches:
        try:
            batch_data = next(data_iter)
        except StopIteration:
            data_iter = iter(distillation_data_loader)
            batch_data = next(data_iter)
        

        pseudo_data_student=batch_data[0].cuda() # 0 is data, 1 is label
        pseudo_data_teacher = pseudo_data_student


        out_t= None
        
        for cli_idx in range(len(local_model_list)):     
            with torch.no_grad():
                _logits =  local_model_list[cli_idx](pseudo_data_teacher)
                
                # KL loss
                if out_t is not None:
                    out_t += _logits * 1/ len(local_model_list)
                else: 
                    out_t = _logits * 1/ len(local_model_list)
                


        kd_server_smodel.train()
        # steps on the same pseudo data
        optimizer_server_student.zero_grad()
        out_s =  kd_server_smodel(pseudo_data_student)     
   
        # KL loss
        loss = divergence( #   Distilling the Knowledge in a Neural Network
                out_s, out_t, kd_KL_temperature
            )
    
        loss.backward()
      
        optimizer_server_student.step()

        # after each batch.
        if scheduler_server_student is not None:
            scheduler_server_student.step()
        # overfit need early stop
        
        if (batch_id+1) % kd_eval_batches_freq == 0:
            kd_server_smodel.eval()
            _, validated_perf = test(args.dataset,
                 kd_server_smodel, val_dataloader,log=False
            )
            log_str = ('Server Batch[{0:03}/{1:03}] '
                    'KD:{kd_loss:.4f} ValAcc{val_acc}'.format(
                    batch_id, total_n_server_pseudo_batches ,kd_loss= loss, val_acc =validated_perf   ))
            print(log_str)

              # check early stopping.
            if check_early_stopping(
                model=kd_server_smodel,
                model_ind=0,
                best_tracker=server_best_tracker,
                validated_perf=validated_perf,
                validated_perfs=validated_perfs,
                perf_index=batch_id + 1,
                early_stopping_batches=early_stopping_server_batches,
                best_models=best_models,
            ):
                break

        batch_id += 1

    use_init_server_model = (
            True
            if init_perf_on_val  >= server_best_tracker.best_perf
            else False
        )

        # get the server model.
    if use_init_server_model:
        print("use init server model instead.")
        return None, 0
    else:
        print("use distillation model at server step {} with val performance {}".format(server_best_tracker.get_best_perf_loc,server_best_tracker.best_perf  ))
        kd_server_smodel = best_models[0] 
        return kd_server_smodel, server_best_tracker.get_best_perf_loc -1 

# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

from collections import OrderedDict
import copy
import numpy as np
import random
import time
import torch
import json
import wandb
import os


class FedBase:
    """Base class for FL algos"""

    def __init__(
        self,
        args,
        clients,
        train_data,
        test_data,
        global_model,
        kd_trainset,
        val_dataloader,
        test_dataloader,
    ):
        try:
            self.update_layer_name
        except:
            self.update_layer_name = None
        else:
            print("self.update_layer_name was defined.", self.update_layer_name)
        self.clients = clients
        self.train_data = train_data
        self.test_data = test_data

        self.args = args

        if args.load_checkpoint is not None:
            fname = os.path.join(args.load_checkpoint, "gmodel.ckpt")
            stat_dict = torch.load(fname)
            load_epoch = stat_dict["epoch"]
            global_model.load_state_dict(stat_dict["state_dict"], strict=False)
            print("load gloal model epoch {} from {}".format(fname, load_epoch, fname))

        self.global_model = copy.deepcopy(global_model)
        self.current_round = -1
        if self.args.dataset == "chexpert":
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        self.kd_trainset = kd_trainset
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.selected_clients = None

        self.local_model_list = []

        self.num_train_samples_all_clients = []
        self.num_test_samples_all_clients = []
        self.local_optimizers = []

        self.saved_results = {
            "round": [],
            "testServerCentAcc": [],
            "testUserPerLocalAcc": [],
            "testUserPerLocalAccStd": [],
            "testUserPerCentAcc": [],
            "testUserPerCentAccStd": [],
        }

        for u in clients["train_users"]:
            local_model = copy.deepcopy(global_model)
            self.num_train_samples_all_clients.append(len(train_data[u]["indices"]))
            self.num_test_samples_all_clients.append(len(test_data[u]["indices"]))
            local_optim = torch.optim.SGD(local_model.parameters(), lr=args.lr)

            self.local_model_list.append(local_model)
            self.local_optimizers.append(local_optim)

        self.train_users = clients["train_users"]
        self.test_users = clients["test_users"]

        self.global_best_acc = 0
        self.start_time = time.time()
        self.per_best_acc = [0 for u in clients["train_users"]]
        self.saved_test_per_acces = [0 for u in clients["train_users"]]
        self.saved_test_per_cent_acces = [0 for u in clients["train_users"]]

    def sample_clients(self, clients_per_round):
        return random.sample(list(range(len(self.train_users))), clients_per_round)

    def run_local_updates(
        self, u, local_model, train_dataloader_u, num_epochs, optimizer
    ):
        return self.train(local_model, train_dataloader_u, num_epochs, optimizer, u=u)

    def server_aggregation(
        self, select_model_list, num_train_samples, update_layer_name
    ):

        from aggregation.averaging import FedAveraging

        w_glob_avg = FedAveraging(
            [c.state_dict() for c in select_model_list],
            weights=[],
            update_layer_name=update_layer_name,
        )
        self.sever_load_state_dict(w_glob_avg)

    def local_finetune_one_round(self, current_round):
        self.current_round = current_round

        (
            train_acces,
            num_train_samples,
        ) = ([], [])
        self.selected_clients = list(range(len(self.train_users)))

        for u in self.selected_clients:

            local_model = self.local_model_list[u]
            # update local model

            train_dataloader_u = self.train_data[u]["dataloader"]

            u_train_loss, u_train_acc = self.run_local_updates(
                u, local_model, train_dataloader_u, 1, self.local_optimizers[u]
            )

            num_train_samples.append(len(self.train_data[u]["indices"]))
            train_acces.append(u_train_acc)

        if self.args.log_online:

            wandb.log(
                {
                    "trainUserLocalAcc": (
                        np.nan
                        if len(train_acces) == 0
                        else np.average(train_acces, weights=num_train_samples)
                    ),
                },
                step=self.current_round,
            )

        if self.current_round % self.args.eval_every == 0:
            self.save_checkpoints(-1)

    def is_global_update(self, name):
        if self.update_layer_name is None:
            return True
        else:
            for update_name in self.update_layer_name:
                if update_name in name:
                    return True
            return False

    def state_dict_sever_to_client(self):
        server_state_dict = OrderedDict(
            (n, p)
            for (n, p) in self.global_model.state_dict().items()
            if self.is_global_update(n) == True
        )

        return server_state_dict

    def sever_load_state_dict(self, w_glob_avg):

        self.global_model.load_state_dict(w_glob_avg)

    def run_one_round(self, current_round):
        self.current_round = current_round

        (
            train_acces,
            num_train_samples,
        ) = ([], [])
        self.selected_clients = self.sample_clients(self.args.clients_per_round)
        select_model_list = []

        state_dict_sever_to_client = self.state_dict_sever_to_client()
        for u in self.selected_clients:
            local_model = self.local_model_list[u]
            local_model.load_state_dict(state_dict_sever_to_client, strict=False)

            # update local model
            train_dataloader_u = self.train_data[u]["dataloader"]
            u_train_loss, u_train_acc = self.run_local_updates(
                u,
                local_model,
                train_dataloader_u,
                self.args.num_epochs,
                self.local_optimizers[u],
            )
            select_model_list.append(local_model)
            train_acces.append(u_train_acc)

            num_train_samples.append(len(self.train_data[u]["indices"]))

        # fedavg
        self.server_aggregation(
            select_model_list, num_train_samples, self.update_layer_name
        )

        _, global_acc = self.test(self.global_model, self.test_dataloader)
        print(
            "round {} select clients {} fedavg global model {} time {:.2f}".format(
                self.current_round,
                self.selected_clients,
                global_acc,
                time.time() - self.start_time,
            )
        )
        self.start_time = time.time()

        if self.args.log_online:
            wandb.log(
                {
                    "testServerCentAcc": global_acc,
                    "trainUserLocalAcc": (
                        np.nan
                        if len(train_acces) == 0
                        else np.average(train_acces, weights=num_train_samples)
                    ),
                },
                step=self.current_round,
            )

        if self.current_round % self.args.eval_every == 0:
            self.save_checkpoints(global_acc)

    def save_checkpoints(self, global_acc):
        if self.args.nologging == False:
            self.saved_results["round"].append(self.current_round)
            self.saved_results["testServerCentAcc"].append(global_acc)

            with open(
                os.path.join(self.args.output_summary_file, "results.json"), "w"
            ) as f:
                json.dump(self.saved_results, f)
            # check to save the model
            if global_acc > self.global_best_acc:
                self.global_best_acc = global_acc
                torch.save(
                    {
                        "state_dict": self.global_model.state_dict(),
                        "epoch": self.current_round,
                    },
                    os.path.join(self.args.output_summary_file, "gmodel.ckpt"),
                )

    def train(self, model, trainloader, epoch, optimizer, log=False, u=0):

        criterion = self.criterion

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr
        )
        model.zero_grad()
        model.train()

        def train_core(data_loader, model, epoch, optimizer):
            for ep in range(epoch):
                train_loss = 0
                correct = 0
                total = 0
                for batch_idx, (x, y) in enumerate(data_loader):
                    x, y = x.to("cuda"), y.to("cuda")
                    optimizer.zero_grad()
                    pred = model(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    total += y.size(0)
                    if self.args.dataset == "chexpert":
                        correct += 0
                    else:
                        _, pred_c = pred.max(1)
                        correct += pred_c.eq(y).sum().item()

                print(
                    "cli %d ep %d batch %d  train Loss: %.3f | Acc:%.3f%% | total %d"
                    % (
                        u,
                        ep,
                        batch_idx,
                        train_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        total,
                    )
                )
            optimizer.zero_grad()
            acc = 100.0 * correct / total
            return train_loss / len(data_loader), acc

        _loss, _acc = train_core(trainloader, model, epoch, optimizer)

        return _loss, _acc

    def computeAUROC(self, dataGT, dataPRED, nnClassCount):
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

    def test(self, model, testloader, log=False):
        model.eval()
        criterion = self.criterion
        test_loss = 0
        correct = 0
        total = 0
        if self.args.dataset == "chexpert":
            y_prob = []
            y_true = []
            from sklearn import metrics
            import numpy as np

            criterion = torch.nn.BCEWithLogitsLoss()
            sgmd = torch.nn.Sigmoid().cuda()
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(testloader):
                    x, y = x.to("cuda"), y.to("cuda")
                    pred = model(x)
                    loss = criterion(pred, y)
                    y_prob.append(sgmd(pred).detach().cpu().numpy())
                    y_true.append(y.detach().cpu().numpy())

                    test_loss += loss.item()
                    total += y.size(0)

                y_prob = np.concatenate(y_prob, axis=0)
                y_true = np.concatenate(y_true, axis=0)
                aurocMean = metrics.roc_auc_score(y_true, y_prob, average="macro")

                if log:
                    print(
                        "test Loss: %.3f | Acc:%.3f%%"
                        % (test_loss / (batch_idx + 1), aurocMean * 100)
                    )
            return test_loss / len(testloader), aurocMean * 100.0

        else:
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(testloader):
                    x, y = x.to("cuda"), y.to("cuda")
                    pred = model(x)
                    loss = criterion(pred, y)

                    test_loss += loss.item()
                    _, pred_c = pred.max(1)
                    total += y.size(0)
                    correct += pred_c.eq(y).sum().item()
                if log:
                    print(
                        "test Loss: %.3f | Acc:%.3f%%"
                        % (test_loss / (batch_idx + 1), 100.0 * correct / total)
                    )

            acc = 100.0 * correct / total
            return test_loss / len(testloader), acc

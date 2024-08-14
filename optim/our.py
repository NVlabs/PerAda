# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

import copy
import numpy as np
import torch
import json
import os
from .base import FedBase
import wandb


class Our(FedBase):
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
        if "fix" in args.net:  # only update the layers in "update_layer_name"
            if "adapter" in args.net:
                self.update_layer_name = [
                    "adapter",
                    "bn1",
                    "bn2",
                    "fc.weight",
                    "fc.bias",
                ]  # adapter layers, batch norm layers, fc layers
            elif "out" in args.net:
                self.update_layer_name = ["fc.weight", "fc.bias"]
            elif "inp" in args.net:
                self.update_layer_name = ["conv1.weight", "bn1.weight", "bn1.bias"]
            else:
                raise NotImplementedError

            for (
                n,
                p,
            ) in (
                global_model.named_parameters()
            ):  # fix other layers expect update_layer_name
                if (
                    self.is_global_update(n) == False
                ):  # fix the global layers; only finetune personalized layers
                    p.requires_grad_(False)

        super().__init__(
            args,
            clients,
            train_data,
            test_data,
            global_model,
            kd_trainset,
            val_dataloader,
            test_dataloader,
        )

        self.personalized_model_list = []
        self.personalized_optimizers = []
        self.personalized_privacy_engines = []

        for u in clients["train_users"]:
            per_model = copy.deepcopy(global_model)
            local_optim = torch.optim.SGD(per_model.parameters(), lr=args.lr)

            if (
                args.load_checkpoint is not None
                and "_fedavg_" not in args.load_checkpoint
            ):
                fname = os.path.join(args.load_checkpoint, "permodel_{}.ckpt".format(u))
                stat_dict = torch.load(fname)
                load_epoch = stat_dict["epoch"]
                per_model.load_state_dict(stat_dict["state_dict"], strict=False)
                print(
                    "load personalized model epoch {} from {}".format(
                        fname, load_epoch, fname
                    )
                )

            self.personalized_optimizers.append(local_optim)
            self.personalized_model_list.append(per_model)

    def is_global_update(self, name):
        for update_name in self.update_layer_name:
            if update_name in name:
                return True
        return False

    def server_distillation(
        self, init_model, kd_trainset, val_dataloader, select_model_list
    ):
        from aggregation.distillation import Distillation

        distill_model, distill_step = Distillation(
            self.args,
            copy.deepcopy(init_model),
            kd_trainset,
            val_dataloader,
            select_model_list,
        )
        if self.args.log_online:
            import wandb

            wandb.log(
                {"DistillStep": distill_step}, step=self.current_round, commit=False
            )
        if distill_model is not None:
            _, global_acc = self.test(distill_model, self.test_dataloader)
            print("global model after distillation", global_acc)
            return distill_model, global_acc
        else:
            return None, None

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

        if (
            self.args.aggregation == "kd"
            and self.current_round < self.args.kd_max_round
        ):
            distill_model, global_acc = self.server_distillation(
                self.global_model,
                self.kd_trainset,
                self.val_dataloader,
                select_model_list,
            )
            if distill_model is not None:
                self.global_model = distill_model

    def run_local_updates(
        self, u, local_model, train_dataloader_u, num_epochs, optimizer
    ):
        # update personalzied model
        per_model = self.personalized_model_list[u]
        per_optimizer = self.personalized_optimizers[u]

        u_train_per_loss, u_train_per_acc = self.train_per(
            per_model,
            local_model,
            self.args.lmbda,
            train_dataloader_u,
            num_epochs,
            per_optimizer,
            u=u,
        )

        u_train_loss, u_train_acc = self.train(
            local_model, train_dataloader_u, num_epochs, optimizer, u=u
        )
        return u_train_loss, u_train_acc

    def model_dist_norm_var(self, model, prox_center, norm=2):
        model_params = [p for (n, p) in model.named_parameters()]
        return sum(
            torch.norm(v.reshape(-1) - v1.reshape(-1)) ** 2
            for (v, v1) in zip(model_params, prox_center)
        )

    def train_per(
        self, model, global_model, lmbda, trainloader, epoch, optimizer, log=True, u=0
    ):

        criterion = self.criterion

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr
        )
        prox_center = [p.detach() for (n, p) in global_model.named_parameters()]

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
                    loss = criterion(pred, y) + lmbda * self.model_dist_norm_var(
                        model, prox_center
                    )
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
                    "personalized cli %d ep %d batch %d  train Loss: %.3f | Acc:%.3f%% | total %d"
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

    def local_finetune_one_round(self, current_round):
        self.current_round = current_round

        (
            train_acces,
            num_train_samples,
        ) = ([], [])
        self.selected_clients = list(range(len(self.train_users)))

        for u in self.selected_clients:
            train_dataloader_u = self.train_data[u]["dataloader"]
            per_model = self.personalized_model_list[u]
            per_optim = self.personalized_optimizers[u]
            # update local model
            u_train_loss, u_train_acc = self.run_local_updates(
                u, per_model, train_dataloader_u, 1, per_optim
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

    def save_checkpoints(self, global_acc):

        for u in self.selected_clients:
            test_dataloader_u = self.test_data[u]["dataloader"]
            per_model = self.personalized_model_list[u]
            u_test_per_loss, u_test_per_acc = self.test(
                per_model, test_dataloader_u
            )  # test with local testdataset
            self.saved_test_per_acces[u] = u_test_per_acc

            u_test_per_cent_loss, u_test_per_cent_acc = self.test(
                per_model, self.test_dataloader
            )  # test with centralized testdataset

            self.saved_test_per_cent_acces[u] = u_test_per_cent_acc

            if self.args.nologging == False:
                if u_test_per_acc > self.per_best_acc[u]:
                    self.per_best_acc[u] = u_test_per_acc
                    torch.save(
                        {
                            "state_dict": per_model.state_dict(),
                            "epoch": self.current_round,
                        },
                        os.path.join(
                            self.args.output_summary_file, "permodel_{}.ckpt".format(u)
                        ),
                    )

        if self.args.log_online:

            wandb.log(
                {
                    "testUserPerLocalAcc": (
                        np.nan
                        if len(self.saved_test_per_acces) == 0
                        else np.average(
                            self.saved_test_per_acces,
                            weights=self.num_test_samples_all_clients,
                        )
                    ),
                    "testUserPerCentAcc": (
                        np.nan
                        if len(self.saved_test_per_cent_acces) == 0
                        else np.average(self.saved_test_per_cent_acces)
                    ),
                },
                step=self.current_round,
            )

        if self.args.nologging == False:

            self.saved_results["round"].append(self.current_round)
            self.saved_results["testServerCentAcc"].append(global_acc)
            self.saved_results["testUserPerLocalAcc"].append(
                np.average(
                    self.saved_test_per_acces, weights=self.num_test_samples_all_clients
                )
            )
            self.saved_results["testUserPerLocalAccStd"].append(
                round(np.array(self.saved_test_per_acces).std(), 5)
            )
            self.saved_results["testUserPerCentAcc"].append(
                np.average(self.saved_test_per_cent_acces)
            )
            self.saved_results["testUserPerCentAccStd"].append(
                round(np.array(self.saved_test_per_cent_acces).std(), 5)
            )

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

# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

import numpy as np
import torch
import json
import os
from .base import FedBase
import wandb


class StandAlone(FedBase):
    """StandAlone training"""

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

        if args.load_checkpoint is not None:
            for u in self.clients["train_users"]:
                local_model = self.local_model_list[u]
                fname = os.path.join(args.load_checkpoint, "permodel_{}.ckpt".format(u))
                stat_dict = torch.load(fname)
                load_epoch = stat_dict["epoch"]
                local_model.load_state_dict(stat_dict["state_dict"], strict=False)
                print(
                    "load personalized model epoch {} from {}".format(
                        fname, load_epoch, fname
                    )
                )

    def run_one_round(self, current_round):
        self.current_round = current_round
        (
            train_acces,
            num_train_samples,
        ) = ([], [])

        self.selected_clients = self.sample_clients(self.args.clients_per_round)
        select_model_list = []
        for u in self.selected_clients:
            train_dataloader_u = self.train_data[u]["dataloader"]
            local_model = self.local_model_list[u]
            # update local model
            optimizer = self.local_optimizers[u]
            u_train_loss, u_train_acc = self.run_local_updates(
                u, local_model, train_dataloader_u, self.args.num_epochs, optimizer
            )
            select_model_list.append(local_model)
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
            per_model = self.local_model_list[u]
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

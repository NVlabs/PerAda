# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

from datasets.read_data import read_partition_data
import random
import numpy as np
import torch
from parameters import parse_args
import os
from datasets.prepare_data import get_dataset
from models.model_utils import get_model
import optim
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_pfl_optimizer(pfl_algo, **kwargs):
    if pfl_algo.lower() == "fedavg":
        return optim.FedAvg(**kwargs)
    elif pfl_algo.lower() in ["standalone"]:
        return optim.StandAlone(**kwargs)
    elif pfl_algo.lower() in ["central"]:
        return optim.Central(**kwargs)
    elif pfl_algo.lower() in ["our"]:
        return optim.Our(**kwargs)

    else:
        raise ValueError(f"Unknown PFL algorithm: {pfl_algo}")


args = parse_args()


if args.nologging == False:
    if not os.path.exists(args.output_summary_file):
        os.makedirs(args.output_summary_file)


if args.log_online:
    import wandb

    _ = os.system("wandb login {}".format(args.wandb_key))
    os.environ["WANDB_API_KEY"] = args.wandb_key
    wandb.init(project=args.project, name=os.path.basename(args.output_summary_file))
    wandb.config.update(args)

init_seed(args.seed)
# GPU setup https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
torch.backends.cudnn.benchmark = True  # faster

# prepare data

clients, kd_dataloader, train_data, test_data, val_dataloader, test_dataloader = (
    read_partition_data(
        args.dataset,
        args.num_clients,
        args.dirichlet_alpha,
        args.batch_size,
        args.test_batch_size,
        args.kd_batch_size,
        args.shard_per_user,
        img_resolution=args.img_resolution,
        kd_data_fraction=args.kd_data_fraction,
    )
)


kd_trainset = None
if args.aggregation == "kd":
    if args.kd_dataset != args.dataset:

        kd_trainset = get_dataset(
            data_name=args.kd_dataset,
            datasets_path="data",
            split="train",
            img_resolution=args.img_resolution,
        )

        kd_idx = np.random.choice(
            list(set(range(len(kd_trainset)))),
            int(len(kd_trainset) * args.kd_data_fraction),
            replace=False,
        )
        print("kd_idx len", len(kd_idx), "out of", len(kd_trainset), args.kd_dataset)
        kd_dataloader = torch.utils.data.DataLoader(
            kd_trainset,
            batch_size=args.kd_batch_size,
            pin_memory=False,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(kd_idx),
        )


# initialzae the model
global_model = get_model(
    dataset=args.dataset, net=args.net, per_dropout=args.adapter_dropout
)
pfl_args = dict(
    args=args,
    clients=clients,
    train_data=train_data,
    test_data=test_data,
    global_model=global_model,
    kd_trainset=kd_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
)

pfl_optim = get_pfl_optimizer(args.pfl_algo, **pfl_args)


if args.local_finetune:
    for com_round in range(args.num_rounds):
        pfl_optim.local_finetune_one_round(com_round)
else:
    for com_round in range(args.num_rounds):
        pfl_optim.run_one_round(com_round)

if args.log_online:
    wandb.finish()

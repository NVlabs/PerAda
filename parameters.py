from baseline_constants import AGGR_MEAN, AGGR_KD
from baseline_constants import DATASETS


import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dataset", help="name of dataset;", type=str, choices=DATASETS, required=True
    )
    parser.add_argument("--net", type=str, default="resnet18", help="model name;")
    parser.add_argument(
        "--pfl_algo", type=str, default="fedavg", help="algorithm name;"
    )

    parser.add_argument(
        "--num-rounds", help="number of rounds to simulate;", type=int, default=-1
    )
    parser.add_argument(
        "--eval-every", help="evaluate every ____ rounds;", type=int, default=2
    )
    parser.add_argument(
        "--clients-per-round",
        help="number of clients trained per round;",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--seed", help="random seed for reproducibility;", type=int, default=1
    )
    parser.add_argument(
        "--batch_size",
        help="batch size when clients train on data;",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=2048,
        help="batch size when clients test on data;",
    )

    parser.add_argument(
        "--num_epochs",
        help="number of epochs when clients train on data;",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Number of total clients in the federated learning setup.",
    )

    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=None,
        help="Alpha parameter for Dirichlet distribution to control data heterogeneity.",
    )

    parser.add_argument(
        "--shard_per_user",
        type=int,
        default=None,
        help="Number of data shards per user.", # not used if we set dirichlet_alpha
    )

    parser.add_argument(
        "-lr",
        help="learning rate for local optimizers;",
        type=float,
        default=1.0,
        required=False,
    )
    parser.add_argument(
        "--lr-decay", help="decay in learning rate", type=float, default=1.0
    )
    parser.add_argument(
        "--decay-lr-every",
        help="number of iterations to decay learning rate",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--output_summary_file",
        help="Filename to log summary of optimization performance in CSV",
        default="outputs",
    )
    parser.add_argument(
        "--validation",
        help="If specified, hold out part of training data to use as a dev set for parameter search",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--patience-iter",
        help="Number of patience rounds of no updates to wait for before giving up",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--aggregation",
        help="Aggregation technique used to combine updates or gradients",
        choices=[AGGR_MEAN, AGGR_KD],
        default=AGGR_MEAN,
    )

    parser.add_argument(
        "--kd_lr",
        help="learning rate for kd optimizers;",
        type=float,
        default=0.001,
        required=False,
    )
    parser.add_argument(
        "--kd_dataset",
        type=str,
        choices=DATASETS,
        default="cifar100",
        required=False,
        help="Dataset to be used for knowledge distillation.",
    )

    parser.add_argument(
        "--kd_max_round",
        type=int,
        default=30,
        help="Maximum number of communication rounds that use knowledge distillation.",
    )
    parser.add_argument(
        "--kd_batch_size",
        type=int,
        default=128,
        help="Batch size for knowledge distillation training.",
    )

    parser.add_argument(
        "--total_n_server_pseudo_batches",
        type=int,
        default=10000,
        help="Total number of pseudo batches used by the server.",
    )
    parser.add_argument(
        "--early_stopping_server_batches",
        type=int,
        default=1000,
        help="Number of server batches to observe for early stopping.",
    )
    parser.add_argument(
        "--kd_eval_batches_freq",
        type=int,
        default=20,
        help="Frequency of evaluation batches during knowledge distillation training.",
    )
    parser.add_argument(
        "--kd_data_fraction",
        type=float,
        default=1,
        required=False,
        help="Fraction of the data to be used for knowledge distillation.",
    )
    parser.add_argument(
        "--kd_weight_decay",
        type=float,
        default=1e-4,
        required=False,
        help="Weight decay (L2 regularization) for knowledge distillation training.",
    )
    parser.add_argument(
        "--kd_KL_temperature",
        type=float,
        default=1,
        required=False,
        help="Temperature parameter for the KL-divergence in knowledge distillation.",
    )

    parser.add_argument(
        "--lmbda",
        help="the lambda for regularization",
        type=float,  #  used in the loss
        default=0.05,
    )
    parser.add_argument(
        "--nologging",
        help="not logging the training output.",
        action="store_true",
    )

    parser.add_argument(
        "--personalized", help="run with personalization", action="store_true"
    )

    parser.add_argument(
        "--start_finetune_rounds",
        help="start of round of local finetuning",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--img_resolution",
        type=int,
        default=None,
        help="Resolution of input images.",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to load model weights from.",
    )
    parser.add_argument(
        "--local_finetune",
        action="store_true",
        help="Flag to enable local fine-tuning of the model.",
    )

    parser.add_argument(
        "--adapter_dropout",
        type=float,
        default=0.3,
        help="Dropout rate for adapters in the model.",
    )

    parser.add_argument(
        "--log_online",
        action="store_true",
        help="Flag. If set, run metrics are stored online in addition to offline logging. Should generally be set.",
    )

    parser.add_argument(
        "--wandb_key",
        default="",
        type=str,
        help="API key for W&B.",
    )
    parser.add_argument(
        "--project",
        default="pfl",
        type=str,
        help="Name of the project - relates to W&B project names. In --savename default setting part of the savename.",
    )

    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 2)
        print("Random seed not provided. Using {} as seed".format(args.seed))
    return args

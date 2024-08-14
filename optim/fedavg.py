# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/PerAda/blob/main/LICENSE

from .base import FedBase


class FedAvg(FedBase):
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

    def is_global_update(self, name):
        if self.update_layer_name is None:
            return True
        else:
            for update_name in self.update_layer_name:
                if update_name in name:
                    return True
            return False

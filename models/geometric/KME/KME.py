import logging

import numpy as np
import torch

import LPU.constants
import LPU.models.geometric.Elkan.Elkan
import LPU.scripts
import LPU.scripts.KME
import LPU.utils.utils_general
import LPU.models.geometric.KME.modified_Kernel_MPE_grad_threshold

LOG = LPU.utils.utils_general.configure_logger(__name__)

DEFAULT_CONFIG = {
    # for VGP:
    "thres_par": 0.1, 
    "lambda_0": 1.0,
    "lambda_1_increment": 0.05,
    "inducing_points_size": 32,
    "learning_rate": 0.01,
    "num_epochs": 10,
    "stop_learning_lr": 1e-6,
    "device": "cpu",
    "epoch_block": 1, # Perform validation every EPOCH_BLOCK iterations
    "kernel_mode": 2,
    "intrinsic_kernel_params": {
        "normed": False,
        "kernel_type": "laplacian",
        "heat_temp": 0.01,
        "noise_factor": 0.0,
        "amplitude": 0.5,
        "n_neighbor": 5,
        "lengthscale": 0.3,
        "neighbor_mode": "distance",
        "power_factor": 1,
        "invert_M_first": False,
    },
    "dataset_name": "animal_no_animal",  # fashionMNIST
    "dataset_kind": "LPU",
    "data_generating_process": "SB",  # either of CC (case-control) or SB (selection-bias)
    'ratios': 
    {
        # *** NOTE ***
        # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
        # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
        'test': 0.4,
        'val': 0.05,
        'holdout': .05,
        'train': .5, 
    },
    "batch_size": {
        "train": 64,
        "test": 64,
        "val": 64,
        "holdout": 64
    }
}


class KME(LPU.models.geometric.Elkan.Elkan.Elkan):
    """
    Using estimator of p(s|X) to predict p(y|X)
    """
    def __init__(self, config=None, *args, **kwargs):
        if config is None:
            config = DEFAULT_CONFIG
            LOG.info("Using default configuration.")
        self.config = config
        self.kernel_mode = config.get('kernel_mode', 1)
        LOG.info(f"kernel_mode: {self.kernel_mode}")
        super().__init__(config=self.config, **kwargs)

    def set_C(self, holdout_dataloader):        
        X_holdout = []
        l_holdout = []
        for X_holdout_batch, l_holdout_batch, _, _ in holdout_dataloader:
            X_holdout.append(X_holdout_batch)
            l_holdout.append(l_holdout_batch)
        X_holdout = np.vstack(X_holdout)
        l_holdout = np.hstack(l_holdout)
        kappa_2, kappa_1 = LPU.models.geometric.KME.modified_Kernel_MPE_grad_threshold.modified_wrapper(X_holdout, X_holdout[l_holdout==1], thres_par=self.config['thres_par'], lambda_0=self.config['lambda_0'], lambda_1=self.config['lambda_0'] + self.config['lambda_1_increment'])
        if self.kernel_mode == 1:
            C_value = l_holdout.mean() / kappa_1
        else:
            C_value = l_holdout.mean() / kappa_2
        with torch.no_grad():
            self.C.fill_(torch.tensor(C_value).to(LPU.constants.DTYPE))
            LOG.info(f"Set C to {C_value}")
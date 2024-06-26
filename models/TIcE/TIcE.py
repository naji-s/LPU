import bitarray
import logging

import numpy as np
import torch

import LPU.constants
import LPU.external_libs.SAR_PU.lib.tice.tice
import LPU.external_libs.SAR_PU.lib.tice.tice.tice
import LPU.models.geometric.Elkan.Elkan

LOG = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    # for VGP:
    "random_state": 554,
    "inducing_points_size": 32,
    "learning_rate": 0.01,
    "num_epochs": 50,
    "stop_learning_lr": 1e-6,
    "device": "cpu",
    "intrinsic_kernel_params": {
        # "normed": False,
        # "kernel_type": "laplacian",
        # "heat_temp": 0.01,
        # "noise_factor": 0.0,
        # "amplitude": 0.,
        # "n_neighbor": 5,
        # "lengthscale": 0.3,
        # "neighbor_mode": "distance",
        # "power_factor": 1,
        "invert_M_first": False,
        
        "normed": False,
        "kernel_type": "laplacian",
        "noise_factor": 0.0678138720259434,
        "amplitude": 0.12023527271263154,
        "n_neighbor": 5,
        "lengthscale": 8.725214019150634,
        "neighbor_mode": "connectivity",
        "power_factor": 1.9281582357231326,        
        },
    "max-bepp": 4,
    "maxSplits": 284,
    "minT": 9,
    "nbIts": 9,
    "out": None,
    "folds": None,
    "delta": None,
    # "max-bepp": 5,
    # "maxSplits": 500,
    "promis": False,
    "delimiter": ',',
    # "minT": 10,
    # "nbIts": 2,
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
class TIcE(LPU.models.geometric.Elkan.Elkan.Elkan):
    """
    Using estimator of p(s|X) to predict p(y|X)
    """
    def __init__(self, config, *args, **kwargs):
        self.config = config
        super().__init__(config=self.config, **kwargs)

    def set_C(self, holdout_dataloader):        
        X_holdout = []
        l_holdout = []
        for X_holdout_batch, l_holdout_batch, _, _ in holdout_dataloader:
            X_holdout.append(X_holdout_batch)
            l_holdout.append(l_holdout_batch)
        X_holdout = np.vstack(X_holdout)
        l_holdout = np.hstack(l_holdout)
        folds = np.array(map(lambda l: int(l.strip()), open(self.config['folds']).readlines())) if self.config['folds'] else np.random.randint(5, size=len(X_holdout))
        l_holdout_bitarray = bitarray.bitarray(list(l_holdout.astype(int)))
        (c_estimate, c_its_estimates) = LPU.external_libs.SAR_PU.lib.tice.tice.tice.tice(X_holdout, l_holdout_bitarray, self.config['max-bepp'], folds, self.config['delta'], nbIterations=self.config['nbIts'], maxSplits=self.config['maxSplits'], useMostPromisingOnly=self.config['promis'], minT=self.config['minT'])
        # making sure c_estimate is not too small
        c_estimate = max(c_estimate, LPU.constants.EPSILON)
        with torch.no_grad():
            self.C.fill_(c_estimate)
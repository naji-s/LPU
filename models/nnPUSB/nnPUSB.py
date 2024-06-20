"""
Implementation of the nnPU model and uPU model by inheritance from the LPUModelBase class.
Despite this, becasue he code uses Chainer, the details of traininig are not adapted other
pytorch models in this library. That is a future TODO (naji)
"""

import logging
import sys

import LPU.models.nnPU.nnPU
import LPU.models.uPU.uPU
sys.path.append('LPU/external_libs/nnPUSB')

import numpy as np
import torch
import scipy.special

import LPU.constants
import LPU.external_libs
import LPU.models.geometric.Elkan.Elkan
import LPU.models.lpu_model_base


LOG = logging.getLogger(__name__)

EPSILON = 1e-16

DEFAULT_CONFIG = {
    "device": "cpu",
    "preset": "figure1",
    "num_epochs": 100,
    "beta": 0.0,
    "gamma": 1.0,
    "learning_rate": 0.001,
    "loss": "sigmoid",
    "out": "/Users/naji/phd_codebase/LPU/scripts/nnPUSB/checkpoints",
    # "model": 'mlp',

    # Dataset configuration
    "dataset_name": "animal_no_animal",
    # "dataset_name": "mnist",
    "dataset_kind": "LPU",
    "data_generating_process": "SB",  # either of CC (case-control) or SB (selection-bias)
    "batch_size": {
        "train": 30000,
        "test": 30000,
        "val": 30000,
        "holdout": 30000
    },
    "model": "mlp",
    "resample_model": "mlp",
    "ratios": {
        # *** NOTE ***
        # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
        # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
        "test": 0.4,
        "val": 0.05,
        "holdout": 0.0,
        "train": 0.55
    },

    # "unlabeled": 100, # uncomment only if you use the original datasets for nnPUSB
    # "batch_size": {
    #     "train": 64,
    #     "test": 64,
    #     "val": 64,
    #     "holdout": 64
    # },
    # "ratios": {
    #     # *** NOTE ***
    #     # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
    #     # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
    #     "test": 0.25,
    #     "val": 0.2,
    #     "holdout": 0.05,
    #     "train": 0.5
    # }
}


class nnPUSB(LPU.models.uPU.uPU.uPU):
    def __init__(self, config, dim=None, **kwargs):
        super(nnPUSB, self).__init__(config, dim=dim, **kwargs)
        self.select_model()


    def predict_prob_y_given_X(self, X=None, f_x=None):
        density_ratio = f_x/self.prior
        threshold = self.estimate_threshold(f_x)
        # print("threshold:", threshold)
        h = torch.sigmoid(density_ratio - threshold)
        return h


    def estimate_threshold(self, f_x):
        density_ratio = f_x / self.prior
        # ascending sort
        sorted_density_ratio = np.sort(density_ratio)
        size = len(density_ratio)

        n_pi = int(size * self.prior)
        threshold = (
            sorted_density_ratio[size - n_pi] + sorted_density_ratio[size - n_pi - 1]) / 2
        return threshold
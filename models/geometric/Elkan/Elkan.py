import logging

import LPU.models.geometric.Elkan.Elkan
import LPU.models.geometric.geometric_base
import LPU.utils.manifold_utils

LOG = logging.getLogger(__name__)


import gpytorch.likelihoods 
import numpy as np

import torch


import LPU.constants
torch.set_default_dtype(LPU.constants.DTYPE)
import LPU.models
import LPU.models.lpu_model_base
import LPU.models.geometric.GVGP
import LPU.models.geometric.PsychM.PsychM

DEFAULT_CONFIG = {
    # for VGP:
    "inducing_points_size": 32,
    "learning_rate": 0.01,
    "num_epochs": 10,
    "stop_learning_lr": 1e-6,
    "device": "cpu",
    "epoch_block": 1,
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
        'train': .50, 
    },

    "batch_size": {
        "train": 64,
        "test": 64,
        "val": 64,
        "holdout": 64
    }
}


EPOCH_BLOCKS = 1
DEVICE = 'cpu'

class Elkan(LPU.models.geometric.geometric_base.GeometricGPLPUBase):
    class CustomLikelihood(gpytorch.likelihoods.Likelihood):
        def __init__(self, **kwargs):
            super().__init__()
        def update_input_data(self, X):
            pass
        def forward(self, function_samples):
            return torch.distributions.Bernoulli(logits=function_samples)
        
    def __init__(self, config, *args, **kwargs):
        self.config = config
        super().__init__(config=self.config, **kwargs)

    def predict_proba(self, X=None, f_x=None):
        if not f_x:
            self.gp_model.update_input_data(X)
            f_x = self.gp_model(X)
        l_prob = torch.sigmoid(f_x.rsample(sample_shape=torch.Size([1000]))).mean(dim=0)
        return l_prob 

    def predict_prob_y_given_X(self, X=None, f_x=None):
        return self.predict_proba(X=X, f_x=f_x) / self.C
    
    def predict_prob_l_given_y_X(self, X=None, f_x=None):
        return self.C 

    def set_C(self, holdout_dataloader):
        C_vals = []
        for holdout_X, holdout_l, _, _ in holdout_dataloader:
            C_vals.append(self.predict_proba(holdout_X[holdout_l==1]).detach().cpu().numpy())
        C_vals = np.hstack(C_vals)
        C_value = np.mean(C_vals, axis=0)
        with torch.no_grad():
            self.C.fill_(torch.tensor(C_value).to(LPU.constants.DTYPE))
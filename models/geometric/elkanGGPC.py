import logging

import lpu.models.geometric.elkanGGPC
import lpu.models.geometric.geometric_base
LOG = logging.getLogger(__name__)


import gpytorch.likelihoods 
import numpy as np

import torch


import lpu.constants
torch.set_default_dtype(lpu.constants.DTYPE)
import lpu.models
import lpu.models.lpu_model_base
import lpu.models.geometric.GVGP
import lpu.models.geometric.psychmGGPC


EPOCH_BLOCKS = 1
DEVICE = 'cpu'

class ElkanGGPC(lpu.models.geometric.geometric_base.GeometricGPLPUBase):
    class CustomLikelihood(gpytorch.likelihoods.Likelihood):
        def __init__(self, config, **kwargs):
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
        self.C = np.mean(C_vals, axis=0)
    

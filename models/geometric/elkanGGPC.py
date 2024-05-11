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

    def predict_proba(self, X):
        self.gp_model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            self.gp_model.update_input_data(X)
            self.gp_model(X)
            l_prob = torch.sigmoid(self.gp_model(X).rsample(sample_shape=torch.Size([1000]))).mean(dim=0)
        return l_prob 

    def predict_prob_y_given_X(self, X):
        return self.predict_proba(X) / self.C
    
    def predict_prob_l_given_y_X(self, X):
        return self.C 

    def set_C(self, holdout_dataloader):
        C_vals = []
        for holdout_X, holdout_l, _, _ in holdout_dataloader:
            C_vals.append(self.predict_proba(holdout_X[holdout_l==1]))
        C_vals = np.hstack(C_vals)
        self.C = np.mean(C_vals, axis=0)
    

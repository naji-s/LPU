"""
Implementation of the nnPU model and uPU model by inheritance from the LPUModelBase class.
Despite this, becasue he code uses Chainer, the details of traininig are not adapted other
pytorch models in this library. That is a future TODO (naji)
"""

import logging
import sys

import lpu.external_libs.nnPUlearning
# sys.path.append('lpu/external_libs/nnPUSB')

import numpy as np
import torch
import scipy.special

import lpu.constants
import lpu.external_libs
import lpu.models.geometric.elkanGGPC
import lpu.models.lpu_model_base
import lpu.external_libs.nnPUSB.nnPU_loss


LOG = logging.getLogger(__name__)

EPSILON = 1e-16

class uPUloss(torch.nn.Module):
    """Loss function for PU learning."""

    def __init__(self, prior, loss, gamma=1, beta=0):
        super(uPUloss, self).__init__()

        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_func = loss
        self.positive = 1
        self.unlabeled = -1

    def forward(self, x, t, return_risk_separately=False):
        t = t[:, None]
        # positive: if positive,1 else 0
        # unlabeled: if unlabeled,1 else 0
        positive, unlabeled = (t == self.positive).float(
        ), (t == self.unlabeled).float()
        n_positive, n_unlabeled = max(1., positive.sum().item()), max(
            1., unlabeled.sum().item())
        y_positive = self.loss_func(x)
        y_unlabeled = self.loss_func(-x)
        positive_risk = torch.sum(
            self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)

        objective = positive_risk + negative_risk
        if return_risk_separately:
            return positive_risk, negative_risk
        else:
            return objective

class uPU(lpu.models.lpu_model_base.LPUModelBase):
    def __init__(self, config, dim=None, **kwargs):
        super(uPU, self).__init__(**kwargs)
        self.config = config
        self.dim = dim
        self.select_model()

    def select_model(self):
        models = {
            "3lp": lpu.external_libs.nnPUSB.model.ThreeLayerPerceptron,
            "mlp": lpu.external_libs.nnPUSB.model.MultiLayerPerceptron, 
            "cnn": lpu.external_libs.nnPUSB.model.CNN
        }
        self.model = models[self.config['model']](self.dim)
    
    def set_C(self, holdout_dataloader):
        assert len(holdout_dataloader) == 1
        _, holdout_l, holdout_y = next(iter(holdout_dataloader))
        self.C = holdout_l[holdout_y == 1.].mean().detach().cpu().numpy()
        is_positive = holdout_y==1
        self.prior = is_positive.mean() if type(is_positive) == 'numpy.ndarray' else  is_positive.detach().cpu().numpy().mean()
        LOG.warning(f"\pi={self.prior} (the Y class prior) is passed using labeled data, since"
                    "the method needds it")


    def train_one_epoch(self, dataloader, loss_func=None, optimizer=None, device=None):
        self.model.train()
        for i, (X_batch, l_batch, _) in enumerate(dataloader):
            X_batch = X_batch.to(device, non_blocking=True)
            l_batch = l_batch.to(device, non_blocking=True)

            optimizer.zero_grad()            
            output = self.model(X_batch)             
            loss = loss_func(output, l_batch * 2 - 1)  
            loss.backward()
            optimizer.step()
        self.model.eval()

    def predict_prob_y_given_X(self, X):
        self.model.eval()
        # getting raw output
        size = len(X)
        f_x = np.reshape(self.model(X).detach().cpu().numpy(), size)
        density_ratio = f_x / self.prior
        # ascending sort
        sorted_density_ratio = np.sort(density_ratio)
        size = len(density_ratio)

        n_pi = int(size * self.prior)
        threshold = (
            sorted_density_ratio[size - n_pi] + sorted_density_ratio[size - n_pi - 1]) / 2
                
        return scipy.special.expit(f_x - threshold)

    def predict_prob_l_given_y_X(self, X):
        self.model.eval()
        return self.C
    
    def predict_proba(self, X):
        self.model.eval()
        return self.predict_prob_y_given_X(X) * self.C
    
    def predict_y_given_X(self, X):
        self.model.eval()
        LOG.debug("Implementation is not based on probabilities, different from usual"
                  "deal with models inheritting from LPUModelBase")
        size = len(X)

        # getting raw output
        raw = np.reshape(self.model(X).detach().cpu().numpy(), size)

        # predict with density ratio and threshold
        h = self.model.predict_with_density_threshold(raw, target=None, prior=self.prior)
        return (h + 1) // 2
    
    def predict(self, X):
        self.model.eval()
        return self.predict_y_given_X(X) > 0.5
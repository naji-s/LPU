"""
Implementation of the nnPU model and uPU model by inheritance from the LPUModelBase class.
Despite this, becasue he code uses Chainer, the details of traininig are not adapted other
pytorch models in this library. That is a future TODO (naji)
"""

import logging
import sys

import lpu.models.uPU
sys.path.append('lpu/external_libs/nnPUSB')

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

# class nnPUloss(lpu.models.uPU.uPUloss):
#     """Loss function for PU learning."""

#     def __init__(self, prior, loss, gamma=1, beta=0):
#         super(nnPUloss, self).__init__(prior=prior, loss=loss, gamma=gamma, beta=beta)
#     def forward(self, x, t):
#         positive_risk, negative_risk = super().forward(x, t, return_risk_separately=True)
#         objective = positive_risk + negative_risk
#         if negative_risk.data < -self.beta:
#             objective = positive_risk - self.beta
#             x_out = -self.gamma * negative_risk
#         else:
#             x_out = objective
#         return x_out


class nnPU(lpu.models.uPU.uPU):
    def __init__(self, config, dim=None, **kwargs):
        super(nnPU, self).__init__(config, dim=dim, **kwargs)
        self.select_model()
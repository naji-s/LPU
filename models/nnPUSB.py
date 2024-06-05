"""
Implementation of the nnPU model and uPU model by inheritance from the LPUModelBase class.
Despite this, becasue he code uses Chainer, the details of traininig are not adapted other
pytorch models in this library. That is a future TODO (naji)
"""

import logging
import sys

import lpu.models.nnPU
import lpu.models.uPU
sys.path.append('lpu/external_libs/nnPUSB')

import numpy as np
import torch
import scipy.special

import lpu.constants
import lpu.external_libs
import lpu.models.geometric.elkanGGPC
import lpu.models.lpu_model_base


LOG = logging.getLogger(__name__)

EPSILON = 1e-16



class nnPUSB(lpu.models.uPU.uPU):
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
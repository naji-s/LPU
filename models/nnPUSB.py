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


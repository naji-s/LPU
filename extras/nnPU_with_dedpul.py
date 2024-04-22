###################################################################################################
#
#   DEPRECATED. DO NOT USE.
#
#   Implementing nnPU with DEDPUL's library at 
#   https://github.com/dimonenka/DEDPUL
#
###################################################################################################
import logging
import random

import numpy as np
import pandas as pd
import sklearn.model_selection
import torch

import lpu.constants
import lpu.external_libs
import lpu.external_libs.DEDPUL
import lpu.external_libs.DEDPUL.algorithms
import lpu.external_libs.DEDPUL.NN_functions
import lpu.external_libs.DEDPUL.utils
import lpu.models.dedpul
import lpu.models.geometric.elkanGGPC
import lpu.models.lpu_model_base

LOG = logging.getLogger(__name__)

EPSILON = 1e-16
class nnPU(lpu.models.dedpul.DEDPUL):
    """
    Using estimator of p(s|X) to predict p(y|X)
    """
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, **kwargs)

    def set_C(self, holdout_dataloader, preds=None):
        try:
            assert (holdout_dataloader.batch_size == len(holdout_dataloader.dataset)), "There should be only one batch in the dataloader."
        except AssertionError as e:
            LOG.error(f"There should be only one batch in the dataloader, but {holdout_dataloader.batch_size} is smaller than {len(holdout_dataloader.dataset)}.")
            raise e
        X, l, y = next(iter(holdout_dataloader))
        self.C = l[y == 1].mean()
    
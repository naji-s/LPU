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

import LPU.constants
import LPU.external_libs
import LPU.external_libs.DEDPUL
import LPU.external_libs.DEDPUL.algorithms
import LPU.external_libs.DEDPUL.NN_functions
import LPU.external_libs.DEDPUL.utils
import LPU.models.dedpul
import LPU.models.geometric.elkanGGPC
import LPU.models.lpu_model_base

LOG = logging.getLogger(__name__)

EPSILON = 1e-16
class nnPU(LPU.models.dedpul.DEDPUL):
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
    
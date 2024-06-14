import logging

import numpy as np

import LPU.models.geometric.elkan.elkanGGPC
import LPU.utils.utils_general
import LPU.models.geometric.kme.modified_Kernel_MPE_grad_threshold

LOG = LPU.utils.utils_general.configure_logger(__name__)


class KMEModelGGPC(LPU.models.geometric.elkan.elkanGGPC.ElkanGGPC):
    """
    Using estimator of p(s|X) to predict p(y|X)
    """
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.kernel_mode = config.get('kernel_mode', 1)
        LOG.info(f"kernel_mode: {self.kernel_mode}")
        super().__init__(config=self.config, **kwargs)

    def set_C(self, holdout_dataloader):        
        X_holdout = []
        l_holdout = []
        for X_holdout_batch, l_holdout_batch, _, _ in holdout_dataloader:
            X_holdout.append(X_holdout_batch)
            l_holdout.append(l_holdout_batch)
        X_holdout = np.vstack(X_holdout)
        l_holdout = np.hstack(l_holdout)
        kappa_2, kappa_1 = LPU.models.geometric.kme.modified_Kernel_MPE_grad_threshold.modified_wrapper(X_holdout, X_holdout[l_holdout==1], thres_par=self.config['thres_par'], lambda_0=self.config['lambda_0'], lambda_1=self.config['lambda_0'] + self.config['lambda_1_increment'])
        if self.kernel_mode == 1:
            self.C = l_holdout.mean() / kappa_1
        else:
            self.C = l_holdout.mean() / kappa_2
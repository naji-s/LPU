import bitarray
import logging

import numpy as np
import LPU.external_libs.SAR_PU.lib.tice.tice
import LPU.external_libs.SAR_PU.lib.tice.tice.tice
import LPU.models.geometric.elkan.elkanGGPC

LOG = logging.getLogger(__name__)

EPSILON = 1e-16


class Tice(LPU.models.geometric.elkan.elkanGGPC.ElkanGGPC):
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
        folds = np.array(map(lambda l: int(l.strip()), open(self.config['folds']).readlines())) if self.config['folds'] else np.random.randint(5, size=len(X_holdout))
        l_holdout_bitarray = bitarray.bitarray(list(l_holdout.astype(int)))
        (c_estimate, c_its_estimates) = LPU.external_libs.SAR_PU.lib.tice.tice.tice.tice(X_holdout, l_holdout_bitarray, self.config['max-bepp'], folds, self.config['delta'], nbIterations=self.config['nbIts'], maxSplits=self.config['maxSplits'], useMostPromisingOnly=self.config['promis'], minT=self.config['minT'])

        self.C = c_estimate
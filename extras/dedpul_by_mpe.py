
import copy

import numpy as np
import torch.nn
import LPU.external_libs.DEDPUL
import LPU.external_libs.DEDPUL.algorithms
import LPU.external_libs.PU_learning.baselines
import LPU.external_libs.PU_learning.baselines.dedpul

import LPU.external_libs.PU_learning.utils
import LPU.models.lpu_model_base
import LPU.external_libs.PU_learning.estimator
import LPU.external_libs.PU_learning.train_PU
import LPU.constants
import LPU.models.MPE.MPE

    
class DEDPUL(LPU.models.MPE.MPE.MPE):
    def __init__(self, config):
        super(DEDPUL, self).__init__(config)

        self.alpha_estimate = None        

    
    def estimate_alpha(self, p_holdoutloader, u_holdoutloader):
        """
        Estimates the alpha value using the BBE estimator.

        NOTE: in the original code (https://github.com/acmi-lab/PU_learning/blob/5e5e350dc0588de95a36eb952e3cf5382e786aec/train_PU.py#L130)
        alpha is estimated using validation set, which is also used as the test set. This is not a good practice, hence below we use the holdout set.

        Args:
            net (object): The neural network model.
            self.config (dict): Configuration parameters.
            p_holdoutloader (object): DataLoader for positive holdout data.
            u_holdoutloader (object): DataLoader for unlabeled holdout data.

        Returns:
            float: The estimated alpha value.

        Raises:
            None

        """
        alpha = None
        pdata_probs = self.p_probs(p_holdoutloader)
        udata_probs, _ = self.u_probs(u_holdoutloader)

        poster = np.zeros_like(udata_probs[:,0])
        preds = np.concatenate((1.0- pdata_probs, udata_probs[:,1]),axis=0)
        targets = np.concatenate((np.zeros_like(pdata_probs), np.ones_like(udata_probs[:,1])), axis=0 )
        
        try:    
            diff = LPU.external_libs.PU_learning.baselines.dedpul.estimate_diff(preds, targets) 
            alpha, poster = LPU.external_libs.PU_learning.baselines.dedpul.estimate_poster_em(diff=diff, mode='dedpul', alpha=None)

            if alpha<=1e-4: 
                alpha, poster =  LPU.external_libs.PU_learning.baselines.dedpul.estimate_poster_dedpul(diff=diff, alpha=alpha)
        
        except: 
            alpha = 0.0
            poster = preds

        return 1 - alpha, poster    
    
    



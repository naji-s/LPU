import logging

import numpy as np
import sklearn.svm

import sarpu.PUmodels
import sarpu.pu_learning


import lpu.models.Kernel_MPE_grad_threshold
import lpu.models.geometric.elkanGGPC
import lpu.models.lpu_model_base

LOG = logging.getLogger(__name__)

EPSILON = 1e-16
class SVMPU(sklearn.svm.SVC, sarpu.PUmodels.BasePU):
    def __init__(self, tol=1e-4, C=1.0, kernel='rbf', degree=3, gamma='scale',
                class_weight=None, random_state=None,  max_iter=-1, cache_size=200,
                 decision_function_shape='ovr', verbose=0):
        sklearn.svm.SVC.__init__(self, tol=tol, C=C, 
                        class_weight=class_weight, random_state=random_state,
                         max_iter=max_iter, decision_function_shape=decision_function_shape,
                        verbose=verbose, kernel=kernel, degree=degree, gamma=gamma, probability=True)
               
        
    def fit(self, x, s, e=None, sample_weight=None):
        if e is None:
            super().fit(x,s,sample_weight)
        else:
            Xp,Yp,Wp = self._make_propensity_weighted_data(x,s,e,sample_weight)
            super().fit(Xp,Yp,Wp)



class SARPU(lpu.models.lpu_model_base.LPUModelBase):
    """
    Using estimator of p(s|X) to predict p(y|X)
    """
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.kernel_mode = config.get('kernel_mode', 1)
        LOG.info(f"kernel_mode: {self.kernel_mode}")
        super().__init__()
        self.max_iter = config.get('max_iter', 1)
        self.classification_model = SVMPU()
        self.propensity_model = None

    def train(self, dataloader):
        try:
            assert (dataloader.batch_size == len(dataloader.dataset)), "There should be only one batch in the dataloader."
        except AssertionError as e:
            LOG.error(f"There should be only one batch in the dataloader, but {dataloader.batch_size} is smaller than {len(dataloader.dataset)}.")
            raise e
        X, l, _ = next(iter(dataloader))
        X = X.numpy()
        l = l.numpy()
        propensity_attributes = np.ones_like(X[0]).astype(int)
        self.classification_model, self.propensity_model, self.results = sarpu.pu_learning.pu_learn_sar_em(X, l, classification_model=self.classification_model, propensity_attributes=propensity_attributes, max_its=self.max_iter)
        
    def predict_prob_y_given_X(self, X):
        return self.classification_model.predict_proba(X)

    def predict_prob_l_given_y_X(self, X):
        output = self.propensity_model.predict_proba(X)
        return output
    
    def loss_fn(self, X, l):
        class_probabilities = self.predict_prob_y_given_X(X)
        propensity_scores = self.predict_prob_l_given_y_X(X)
        return sarpu.pu_learning.loglikelihood_probs(class_probabilities, propensity_scores, l)

    
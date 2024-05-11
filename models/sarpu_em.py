import logging

import numpy as np
import sklearn.svm

import lpu.constants
import lpu.external_libs.SAR_PU.sarpu.sarpu.pu_learning
import lpu.models.lpu_model_base

LOG = logging.getLogger(__name__)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class BasePU:
    @staticmethod
    # changing the original code at https://github.com/ML-KULeuven/SAR-PU/blob/c51f8af2b9604c363b6d129b7ad03b8db488346f/sarpu/sarpu/PUmodels.py
    # to avoid division by zero

    def _make_propensity_weighted_data(x,s,e,sample_weight=None):
        e = e + lpu.constants.EPSILON
        weights_pos = s/e
        weights_neg = (1-s) + s*(1-1/e)
        if sample_weight is not None:
            weights_pos = sample_weight*weights_pos
            weights_neg = sample_weight*weights_neg
            
        Xp = np.concatenate([x,x])
        Yp = np.concatenate([np.ones_like(s), np.zeros_like(s)])
        Wp = np.concatenate([weights_pos, weights_neg])
        return Xp, Yp, Wp

class LogisticRegressionPU(LogisticRegression, BasePU):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):
        LogisticRegression.__init__(self,penalty=penalty, dual=dual, tol=tol, C=C, 
                         fit_intercept=fit_intercept,intercept_scaling=intercept_scaling,
                        class_weight=class_weight, random_state=random_state,
                        solver=solver, max_iter=max_iter, multi_class=multi_class,
                        verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
               
        
    def fit(self, x, s, e=None, sample_weight=None):
        if e is None:
            super().fit(x,s,sample_weight)
        else:
            Xp,Yp,Wp = self._make_propensity_weighted_data(x,s,e,sample_weight)
            super().fit(Xp,Yp,Wp)
            
        

class SVMPU(sklearn.svm.SVC, BasePU):
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

    # def train(self, dataloader):
    #     try:
    #         assert (dataloader.batch_size == len(dataloader.dataset)), "There should be only one batch in the dataloader."
    #     except AssertionError as e:
    #         LOG.error(f"There should be only one batch in the dataloader, but {dataloader.batch_size} is smaller than {len(dataloader.dataset)}.")
    #         raise e
    #     X, l, _, _ = next(iter(dataloader))
    #     X = X.numpy()
    #     l = l.numpy()
    #     propensity_attributes = np.ones_like(X[0]).astype(int)
    #     self.classification_model, self.propensity_model, self.results = lpu.external_libs.SAR_PU.sarpu.sarpu.pu_learning.pu_learn_sar_em(X, l, classification_model=self.classification_model, propensity_attributes=propensity_attributes, max_its=self.max_iter)

    def train(self, dataloader, holdout_dataloader=None):
        try:
            assert (dataloader.batch_size == len(dataloader.dataset)), "There should be only one batch in the dataloader."
        except AssertionError as e:
            LOG.error(f"There should be only one batch in the dataloader, but {dataloader.batch_size} is smaller than {len(dataloader.dataset)}.")
            raise e

        X, l, y, _ = next(iter(dataloader))
        X = X.numpy()
        l = l.numpy()
        y = y.numpy()
        propensity_attributes = np.ones_like(X[0]).astype(int)

        self.classification_model, self.propensity_model, self.results = lpu.external_libs.SAR_PU.sarpu.sarpu.pu_learning.pu_learn_sar_em(
            X, l, classification_model=self.classification_model, propensity_attributes=propensity_attributes, max_its=self.max_iter)

        scores_dict = self.calculate_probs_and_scores(X, l, y)
        
        return scores_dict

    def predict_prob_y_given_X(self, X):
        return self.classification_model.predict_proba(X)

    def predict_prob_l_given_y_X(self, X):
        output = self.propensity_model.predict_proba(X)
        return output
    
    def loss_fn(self, X, l):
        class_probabilities = self.predict_prob_y_given_X(X)
        propensity_scores = self.predict_prob_l_given_y_X(X)
        return lpu.external_libs.SAR_PU.sarpu.sarpu.pu_learning.loglikelihood_probs(class_probabilities, propensity_scores, l)

    
import logging

import numpy as np
import sklearn.svm

import LPU.constants
import LPU.external_libs.SAR_PU.sarpu.sarpu.pu_learning
import LPU.models.lpu_model_base

LOG = logging.getLogger(__name__)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class BasePU:
    @staticmethod
    # changing the original code at https://github.com/ML-KULeuven/SAR-PU/blob/c51f8af2b9604c363b6d129b7ad03b8db488346f/sarpu/sarpu/PUmodels.py
    # to avoid division by zero

    def _make_propensity_weighted_data(x,s,e,sample_weight=None):
        e = e + LPU.constants.EPSILON
        weights_pos = s/e
        weights_neg = (1-s) + s*(1-1/e)
        if sample_weight is not None:
            weights_pos = sample_weight*weights_pos
            weights_neg = sample_weight*weights_neg
        Xp = np.concatenate([x,x], axis=0)
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



class SARPU(LPU.models.lpu_model_base.LPUModelBase):
    """
    Using estimator of p(s|X) to predict p(y|X)
    """
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.kernel_mode = config.get('kernel_mode', 1)
        self.max_iter = config.get('num_epochs', 1)
        LOG.info(f"kernel_mode: {self.kernel_mode}")
        super().__init__()
        self.max_iter = config.get('num_epochs', 1)
        self.classification_model = self.choose_SAR_PU_model()
        self.propensity_model = None

    def choose_SAR_PU_model(self):
        return {'svm': SVMPU(**self.config['svm_params']),
        'logistic': LogisticRegressionPU(**self.config['logistic_params'])}[self.config['SAR_PU_classification_model']]

    def train(self, dataloader, holdout_dataloader=None):
        X = []
        l = []
        y = []
        # put all the data in one list
        for data in dataloader:
            X_batch, l_batch, y_batch, _ = data
            X.append(X_batch.numpy())
            l.append(l_batch.numpy())
            y.append(y_batch.numpy())
        # concatenate the data
        X_batch_concat = np.concatenate(X)
        l_batch_concat = np.concatenate(l)
        y_batch_concat = np.concatenate(y)

        # find out the binary type of the data
        binary_kind = set(np.unique(y))

        propensity_attributes = np.ones(X_batch_concat.shape[-1]).astype(int)
        self.classification_model, self.propensity_model, self.results = LPU.external_libs.SAR_PU.sarpu.sarpu.pu_learning.pu_learn_sar_em(
            X_batch_concat, l_batch_concat, classification_model=self.classification_model, propensity_attributes=propensity_attributes, max_its=self.max_iter)

        y_batch_concat_prob = self.predict_prob_y_given_X(X=X_batch_concat)
        l_batch_concat_prob = self.predict_proba(X=X_batch_concat)
        y_batch_concat_est = self.predict_y_given_X(X=X_batch_concat)
        l_batch_concat_est = self.predict(X=X_batch_concat)
        
        # _calculate_validation_metrics assumes that the data is in {0, 1},
        # so we need to convert the data to {0, 1} if it is in {-1, 1}
        if binary_kind == {-1, 1}:
            l_batch_concat = (l_batch_concat + 1) // 2
            y_batch_concat = (y_batch_concat + 1) // 2
        scores_dict = self._calculate_validation_metrics(
            y_batch_concat_prob, y_batch_concat, l_batch_concat_prob, l_batch_concat, l_ests=l_batch_concat_est, y_ests=y_batch_concat_est
        )

        
        return scores_dict

    def predict_prob_y_given_X(self, X=None, f_x=None):
        return self.classification_model.predict_proba(X)

    def predict_prob_l_given_y_X(self, X):
        output = self.propensity_model.predict_proba(X)
        return output
    
    def loss_fn(self, X, l):
        class_probabilities = self.predict_prob_y_given_X(X)
        propensity_scores = self.predict_prob_l_given_y_X(X)
        return LPU.external_libs.SAR_PU.sarpu.sarpu.pu_learning.loglikelihood_probs(class_probabilities, propensity_scores, l)

    def validate(self, dataloader, loss_fn=None):
        scores_dict = {}
        total_loss = 0.
        l_batch_concat = []
        y_batch_concat = []
        y_batch_concat_prob = []
        l_batch_concat_prob = []
        l_batch_concat_est = []
        y_batch_concat_est = []
        binary_kind = set(np.unique(dataloader.dataset.y))
        for batch_num, (X_batch, l_batch, y_batch, _) in enumerate(dataloader):
            loss = loss_fn(X_batch, l_batch)
            y_batch_prob = self.predict_prob_y_given_X(X_batch)
            l_batch_prob = self.predict_proba(X=X_batch)
            y_batch_est = self.predict_y_given_X(X=X_batch)
            l_batch_est = self.predict(X=X_batch)
            


            total_loss += loss.item()

            l_batch_concat.append(l_batch)
            y_batch_concat.append(y_batch)
            y_batch_concat_prob.append(y_batch_prob)
            l_batch_concat_prob.append(l_batch_prob)
            y_batch_concat_est.append(y_batch_est)
            l_batch_concat_est.append(l_batch_est)   

        y_batch_concat_prob = np.concatenate(y_batch_concat_prob)
        l_batch_concat_prob = np.concatenate(l_batch_concat_prob)
        y_batch_concat_est = np.concatenate(y_batch_concat_est)
        l_batch_concat_est = np.concatenate(l_batch_concat_est)
        y_batch_concat = np.concatenate(y_batch_concat)
        l_batch_concat = np.concatenate(l_batch_concat)

        if binary_kind == {-1, 1}:
            y_batch_concat = (y_batch_concat + 1) / 2
            l_batch_concat = (l_batch_concat + 1) / 2

        scores_dict = self._calculate_validation_metrics(
            y_batch_concat_prob, y_batch_concat, l_batch_concat_prob, l_batch_concat, l_ests=l_batch_concat_est, y_ests=y_batch_concat_est
        )

        # for score_type in scores_dict:
        #     scores_dict[score_type] = np.mean(scores_dict[score_type])
        total_loss /= (batch_num + 1)
        scores_dict['overall_loss'] = total_loss

        return scores_dict      
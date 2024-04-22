import numpy as np
import sklearn.base
import sklearn.metrics
import torch

import lpu.constants
import lpu.datasets.animal_no_animal.animal_no_animal_utils
import lpu.utils.dataset_utils  

class LPUModelBase(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _separate_labels(self, y):
        """
        Separates the labels into two separate arrays, one for l and one for y.

        Parameters:
            y (array-like): The labels.

        Returns:
            tuple: A tuple containing the separated labels l and y.
        """
        l_y_cat_transformed = y
        l = l_y_cat_transformed // 2
        y = l_y_cat_transformed % 2
        return l, y
    
    def fit(self, X, l, y=None):
        """
        Fits the model to the training data, which are pairs of input features X and labels l.
        Note that the target values y are not used in the training process, and are only used for evaluation. 

        Parameters:
            X (array-like): The input features.
            l (array-like): The labels.
            y (array-like, optional): The target values. Defaults to None.
        """
        pass

    
    def predict_prob_y_given_X(self, X):
        """
        Predicts the probability of y given x.

        Parameters:
        X (array-like): The input data.

        Returns:
            array-like: The predicted probability of y given x.
        """
        pass

    def predict_prob_l_given_y_X(self, X):
        """
        Predicts the probability of label l given input features X and y=1,
        i.e. p(l|y=1, X).

        Parameters:
            X (array-like): Input features for prediction.

        Returns:
            array-like: The predicted probability of label l given y=1 and X.
        """
        pass

    def predict(self, X, threshold=.5):
        return self.predict_proba(X) >= threshold
    
    def predict_y_given_X(self, X, threshold=.5):
        return self.predict_prob_y_given_X(X) >= threshold

    def predict_proba(self, X):
        """
        Predicts the probability of l given X, i.e. p(l|X).
        Note that this is not the same as predict_prob_l_given_y_X. 
        In fact, due to p(y=1|l=1) = 1, 

                    p(l|X) = p(l|y=1, X) * p(y=1|X)

        Parameters:
            X (array-like): The input data to predict probabilities for.

        Returns:
            array-like: An array of shape (n_samples, n_classes) containing the predicted probabilities for each class.
        """
        return self.predict_prob_y_given_X(X) * self.predict_prob_l_given_y_X(X)
    
    def _calculate_validation_metrics(self, y_probs, y_vals, l_probs, l_vals, l_ests=None, y_ests=None):
        """
        Calculates the validation metrics for the model.

        Parameters:
            y_probs (array-like): The predicted target values.
            y_vals (array-like): The true target values.
            l_probs (array-like): The predicted labels.
            l_vals (array-like): The true labels.

        Returns:
            dict: A dictionary containing the validation metrics.
        """
        metrics = {}
        metrics['y_accuracy'] = sklearn.metrics.accuracy_score(y_vals, y_ests)
        metrics['y_auc'] = sklearn.metrics.roc_auc_score(y_vals, y_probs)
        metrics['l_accuracy'] = sklearn.metrics.accuracy_score(l_vals, l_ests)
        metrics['l_auc'] = sklearn.metrics.roc_auc_score(l_vals, l_probs)
        metrics['y_ll'] = sklearn.metrics.log_loss(y_vals, y_probs)
        metrics['l_ll'] = sklearn.metrics.log_loss(l_vals, l_probs)
        return metrics
    
    def validate(self, dataloader):
        if len(dataloader) == 0:
            return None, None
        y_probs = []
        l_probs = []
        y_vals = []
        l_vals = []
        y_ests = []
        l_ests = []
        losses = []
        for X_val, l_val, y_val in dataloader:
            y_prob = self.predict_prob_y_given_X(X_val)
            l_prob = self.predict_proba(X_val)
            l_est = self.predict(X_val)
            y_est = self.predict_y_given_X(X_val)
            y_probs.append(y_prob)
            y_vals.append(y_val)
            l_probs.append(l_prob)
            l_vals.append(l_val)
            y_ests.append(y_est)
            l_ests.append(l_est)
            if hasattr(self, 'loss_fn'):
                losses.append(self.loss_fn(X_val, l_val).item())
        y_probs = np.hstack(y_probs)
        y_vals = np.hstack(y_vals).astype(int)
        l_probs = np.hstack(l_probs)
        l_vals = np.hstack(l_vals).astype(int)
        l_ests = np.hstack(l_ests).astype(int)
        y_ests = np.hstack(y_ests).astype(int)

        validation_results = self._calculate_validation_metrics(
            y_probs, y_vals, l_probs, l_vals, l_ests=l_ests, y_ests=y_ests)
        validation_results.update({'overall_loss': np.mean(losses)})
        return validation_results
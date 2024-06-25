import numpy as np
import sklearn.base
import sklearn.metrics
import torch

import LPU.constants
import LPU.datasets.animal_no_animal.animal_no_animal_utils
import LPU.utils.dataset_utils
import LPU.utils.utils_general

LOG = LPU.utils.utils_general.configure_logger(__name__)

class LPUModelBase(torch.nn.Module):
    def __init__(self, config=None, *args, **kwargs):
        super().__init__()
        if config is None:
            config = {}
            LOG.warning("No configuration provided. Using default configuration.")
            
        self.C = torch.nn.Parameter(torch.tensor(config.get('C', 1.0), dtype=LPU.constants.DTYPE), requires_grad=False)
        self.prior = torch.nn.Parameter(torch.tensor(config.get('prior', 0.5), dtype=LPU.constants.DTYPE), requires_grad=False)
        
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

    
    def predict_prob_y_given_X(self, X=None, f_x=None):
        """
        Predicts the probability of y given x.

        Parameters:
        X (array-like): The input data.

        Returns:
            array-like: The predicted probability of y given x.
        """
        pass

    def predict_prob_l_given_y_X(self, X=None, f_x=None):
        """
        Predicts the probability of label l given input features X and y=1,
        i.e. p(l|y=1, X).

        Parameters:
            X (array-like): Input features for prediction.

        Returns:
            array-like: The predicted probability of label l given y=1 and X.
        """
        pass

    def predict(self, X=None, f_x=None, threshold=.5):
        return self.predict_proba(X=X, f_x=f_x) >= threshold
    
    def predict_y_given_X(self, X=None, f_x=None, threshold=.5):
        return self.predict_prob_y_given_X(X=X, f_x=f_x) >= threshold

    def predict_proba(self, X=None, f_x=None):
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
        return self.predict_prob_y_given_X(X=X, f_x=f_x) * self.predict_prob_l_given_y_X(X=X)
    
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
        if set(np.unique(y_vals)).union(np.unique(l_vals)).issubset({-1, 1}):
            y_vals = (y_vals + 1) / 2
            l_vals = (l_vals + 1) / 2

        metrics = {}
        metrics['l_accuracy'] = sklearn.metrics.accuracy_score(l_vals, l_ests)
        # breakpoint()
        for arr in [l_vals, y_vals, l_ests, y_ests]:
            if all(arr):
                idx = np.random.randint(len(arr))
                arr[idx] = 0
            if not any(arr):
                idx = np.random.randint(len(arr))
                arr[idx] = 1
        

        metrics['l_precision'] = sklearn.metrics.precision_score(l_vals, l_ests)
        metrics['l_auc'] = sklearn.metrics.roc_auc_score(l_vals, l_probs)
        metrics['l_recall'] = sklearn.metrics.recall_score(l_vals, l_ests)
        metrics['l_f1'] = sklearn.metrics.f1_score(l_vals, l_ests)
        metrics['l_APS'] = sklearn.metrics.average_precision_score(l_vals, l_probs)
        
        metrics['y_accuracy'] = sklearn.metrics.accuracy_score(y_vals, y_ests)
        metrics['y_auc'] = sklearn.metrics.roc_auc_score(y_vals, y_probs)
        metrics['y_precision'] = sklearn.metrics.precision_score(y_vals, y_ests)
        metrics['y_recall'] = sklearn.metrics.recall_score(y_vals, y_ests)
        metrics['y_f1'] = sklearn.metrics.f1_score(y_vals, y_ests)
        metrics['y_APS'] = sklearn.metrics.average_precision_score(y_vals, y_probs)


        metrics['y_ll'] = sklearn.metrics.log_loss(y_vals, y_probs)
        metrics['l_ll'] = sklearn.metrics.log_loss(l_vals, l_probs)
        return metrics
    def calculate_probs_and_scores(self, X_batch, l_batch, y_batch):
        y_prob = self.predict_prob_y_given_X(X_batch)
        l_prob = y_prob * self.C
        l_est = l_prob > 0.5
        y_est = y_prob > 0.5
        scores = self._calculate_validation_metrics(
            y_prob, y_batch, l_prob, l_batch.cpu().numpy(),
            l_ests=l_est, y_ests=y_est.cpu().numpy()
        )
        return scores

    # def validate(self, dataloader, loss_fn=None, output_model=None):
    #     y_probs = []
    #     l_probs = []
    #     y_vals = []
    #     l_vals = []
    #     y_ests = []
    #     l_ests = []
    #     losses = []
    #     with torch.no_grad():
    #         for X_val, l_val, y_val, idx_val in dataloader:
    #             y_prob = self.predict_prob_y_given_X(X_val)
    #             l_prob = self.predict_proba(X_val)
    #             l_est = self.predict(X_val)
    #             y_est = self.predict_y_given_X(X_val)
    #             y_probs.append(y_prob)
    #             y_vals.append(y_val)
    #             l_probs.append(l_prob)
    #             l_vals.append(l_val)
    #             y_ests.append(y_est)
    #             l_ests.append(l_est)
    #             if loss_fn:
    #                 losses.append(loss_fn(output_model(X_val), l_val).item())
    #             else:
    #                 losses.append(0)
    #         y_probs = np.concatenate(y_probs)
    #         y_vals = np.concatenate(y_vals).astype(int)
    #         l_probs = np.concatenate(l_probs)
    #         l_vals = np.concatenate(l_vals).astype(int)
    #         l_ests = np.concatenate(l_ests).astype(int)
    #         y_ests = np.concatenate(y_ests).astype(int)
    #         validation_results = self._calculate_validation_metrics(
    #             y_probs, y_vals, l_probs, l_vals, l_ests=l_ests, y_ests=y_ests)
    #         validation_results.update({'overall_loss': np.mean(losses)})
    #         return validation_results
    def validate(self, dataloader, loss_fn=None, model=None):
        scores_dict = {}
        total_loss = 0.
        l = []
        y = []
        y_prob = []
        l_prob = []
        l_est = []
        y_est = []
        if hasattr(model, 'eval'):
            model.eval()
        binary_kind = set(np.unique(dataloader.dataset.y if hasattr(dataloader.dataset, 'y') else dataloader.dataset.Y))
        with torch.no_grad():
            for batch_num, (X_batch, l_batch, y_batch, _) in enumerate(dataloader):

                if self.config['data_generating_process'] == 'CC':
                    X_batch = torch.concat([X_batch, X_batch[l_batch==1]], dim=0)
                    y_batch = torch.concat([y_batch, y_batch[l_batch==1]], dim=0)
                    l_batch = torch.concat([torch.zeros_like(l_batch), torch.ones((int(l_batch.sum().detach().cpu().numpy().squeeze())))], dim=0)
                if hasattr(model, 'update_input_data'):
                    model.update_input_data(X_batch)
                f_x = model(X_batch)
                loss = loss_fn(f_x, l_batch)
                y_batch_prob = self.predict_prob_y_given_X(f_x=f_x)
                l_batch_prob = self.predict_proba(f_x=f_x)
                y_batch_est = self.predict_y_given_X(f_x=f_x)
                l_batch_est = self.predict(f_x=f_x)
                
                if isinstance(y_batch_prob, np.ndarray):
                    y_batch_prob = torch.tensor(y_batch_prob, dtype=LPU.constants.DTYPE)
                    l_batch_prob = torch.tensor(l_batch_prob, dtype=LPU.constants.DTYPE)
                    y_batch_est = torch.tensor(y_batch_est, dtype=LPU.constants.DTYPE)
                    l_batch_est = torch.tensor(l_batch_est, dtype=LPU.constants.DTYPE)


                total_loss += loss.item()

                l.append(l_batch.detach().cpu().numpy())
                y.append(y_batch.detach().cpu().numpy())
                y_prob.append(y_batch_prob.detach().cpu().numpy())
                l_prob.append(l_batch_prob.detach().cpu().numpy())
                y_est.append(y_batch_est.detach().cpu().numpy())
                l_est.append(l_batch_est.detach().cpu().numpy())   

        y_prob = np.concatenate(y_prob)
        l_prob = np.concatenate(l_prob)
        y_est = np.concatenate(y_est)
        l_est = np.concatenate(l_est)
        y = np.concatenate(y)
        l = np.concatenate(l)

        if binary_kind == {-1, 1}:
            y = (y + 1) / 2
            l = (l + 1) / 2
        scores_dict = self._calculate_validation_metrics(
            y_prob, y, l_prob, l, l_ests=l_est, y_ests=y_est
        )

        # for score_type in scores_dict:
        #     scores_dict[score_type] = np.mean(scores_dict[score_type])
        total_loss /= (batch_num + 1)
        scores_dict['overall_loss'] = total_loss

        return scores_dict    

    # def calculate_probs(self, X_batch=None, raw_output=None):
    #     y_batch_prob = self.predict_prob_y_given_X(X=X_batch, raw_output=raw_output)
    #     l_batch_prob = self.predict_proba(X=X_batch, 
    #     l_batch_est = self.predict(X_batch)
    #     y_batch_est = self.predict_y_given_X(X_batch)
    #     return y_batch_prob, l_batch_prob, l_batch_est, y_batch_est
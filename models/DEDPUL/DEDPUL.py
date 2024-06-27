from abc import abstractmethod
import logging
import random
import sys
sys.path.append('LPU/external_libs/DEDPUL')
import unittest.mock


import numpy as np
import pandas as pd
import sklearn.model_selection
import torch

import LPU.constants
import LPU.external_libs
import LPU.models.geometric.Elkan.Elkan
import LPU.models.lpu_model_base
import LPU.external_libs.DEDPUL
import LPU.external_libs.DEDPUL.algorithms
import LPU.external_libs.DEDPUL.NN_functions

DEFAULT_CONFIG = {
    'learning_rate': .01,
    'num_epochs': 10,
    'device': 'cpu',
    'dtype': LPU.constants.DTYPE,
    'train_val_ratio': .1,
    'evaluation_interval': 1,
    'epoch_blocks': 1,
    'nrep': 10,
    'dedpul_type': 'dedpul',
    'dataset_kind': 'LPU',
    'dataset_name': 'animal_no_animal',
    'cv': 5,
    'estimate_diff_options':
    {
        'MT': True, 
        'MT_coef': 0.25,
        'decay_MT_coef': False,
        'tune': False,
        'bw_mix': 0.05, 
        'bw_pos': 0.1,
        'threshold': 'mid',
        'n_gauss_mix': 20,
        'n_gauss_pos': 10,
        'bins_mix': 20,
        'bins_pos': 20,
        'k_neighbours': None,
    },
    'batch_size': 
    {
        'train': 64,
        'test': 64,
        'val': 64,
        'holdout': 64,
    },
    'ratios': 
    {
        # *** NOTE ***
        # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
        # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
        'test': 0.4,
        'val': 0.05,
        'holdout': .0,
        'train': .55, 
    },
    'train_nn_options': 
    { 
        'loss_function': 'log',
        'n_early_stop': 20,
        'disp': False,
        'beta': 0.,
        'gamma': 1.,
        'bayes_weight': 1e-5,   
    }
}


def expanding_wrapper(*args, **kwargs):
    if 'center' in kwargs:
        del kwargs['center']
    return pd.core.window.Expanding(*args, **kwargs)

LOG = logging.getLogger(__name__)

EPSILON = 1e-16




class DEDPUL(LPU.models.lpu_model_base.LPUModelBase):
    """
    Using estimator of p(s|X) to predict p(y|X)
    """
    @abstractmethod
    def modified_train_NN(mix_data, pos_data, discriminator, d_optimizer, mix_data_val=None, pos_data_val=None, mix_data_test=None, pos_data_test=None,
                    n_epochs=None, batch_size=None, n_batches=None, n_early_stop=None,
                    d_scheduler=None, training_mode='standard', disp=False, loss_function=None, nnre_alpha=None,
                    metric=None, stop_by_metric=False, bayes=False, bayes_weight=None, beta=None, gamma=None):
        """ ** NOTE **: Identical to the LPU.external_libs.DEDPUL.NN_functions.train_NN, but with some modifications to set the 
            type of variables in the model to global constant value LPU.constants.DTYPE
        Train discriminator to classify mix_data from pos_data.
        """
        if n_batches is None:
            n_batches = min(int(mix_data.shape[0] / batch_size), int(pos_data.shape[0] / batch_size))
            batch_size_mix = batch_size_pos = batch_size
        else:
            batch_size_mix, batch_size_pos = int(mix_data.shape[0] / n_batches), int(pos_data.shape[0] / n_batches)
        if mix_data_test is not None:
            data_test = np.concatenate((pos_data_test, mix_data_test))
            target_test = np.concatenate((np.zeros((pos_data_test.shape[0],)), np.ones((mix_data_test.shape[0],))))

        d_losses_train = []
        d_losses_val = []
        d_losses_test = []
        for epoch in range(n_epochs):
            d_metrics_test = []
            discriminator.train()

            d_losses_cur = []
            if d_scheduler is not None:
                d_scheduler.step()

            for i in range(n_batches):

                batch_mix = np.array(random.sample(list(mix_data), batch_size_mix))
                batch_pos = np.array(random.sample(list(pos_data), batch_size_pos))

                batch_mix = torch.as_tensor(batch_mix, dtype=LPU.constants.DTYPE)
                batch_pos = torch.as_tensor(batch_pos, dtype=LPU.constants.DTYPE)

                # Optimize D
                d_optimizer.zero_grad()

                if training_mode == 'standard':
                    if bayes:
                        loss = LPU.external_libs.DEDPUL.NN_functions.d_loss_bayes(batch_mix, batch_pos, discriminator, loss_function, bayes_weight)
                    else:
                        loss = LPU.external_libs.DEDPUL.NN_functions.d_loss_standard(batch_mix, batch_pos, discriminator, loss_function)

                else:
                    loss = LPU.external_libs.DEDPUL.NN_functions.d_loss_nnRE(batch_mix, batch_pos, discriminator, nnre_alpha, beta=beta, gamma=gamma,
                                    loss_function=loss_function)

                loss.backward()
                d_optimizer.step()
                d_losses_cur.append(loss.cpu().item())

            d_losses_train.append(np.mean(d_losses_cur))


            if mix_data_val is not None and pos_data_val is not None:
                discriminator.eval()

                if training_mode == 'standard':
                    if bayes:
                        d_losses_val.append(LPU.external_libs.DEDPUL.NN_functions.d_loss_bayes(torch.as_tensor(mix_data_val, dtype=LPU.constants.DTYPE),
                                                                torch.as_tensor(pos_data_val, dtype=LPU.constants.DTYPE),
                                                                discriminator, w=bayes_weight).item())
                    else:
                        d_losses_val.append(LPU.external_libs.DEDPUL.NN_functions.d_loss_standard(torch.as_tensor(mix_data_val, dtype=LPU.constants.DTYPE),
                                                                torch.as_tensor(pos_data_val, dtype=LPU.constants.DTYPE),
                                                                discriminator).item())
                elif training_mode == 'nnre':
                    d_losses_val.append(LPU.external_libs.DEDPUL.NN_functions.d_loss_nnRE(torch.as_tensor(mix_data_val, dtype=LPU.constants.DTYPE),
                                                        torch.as_tensor(pos_data_val, dtype=LPU.constants.DTYPE),
                                                        discriminator, nnre_alpha).item())

            if mix_data_test is not None and pos_data_test is not None:

                discriminator.eval()

                if training_mode == 'standard':
                    if bayes:
                        d_losses_test.append(round(LPU.external_libs.DEDPUL.NN_functions.d_loss_bayes(torch.as_tensor(mix_data_test, dtype=LPU.constants.DTYPE),
                                                                torch.as_tensor(pos_data_test, dtype=LPU.constants.DTYPE),
                                                                discriminator, w=bayes_weight).item(), 5))
                    else:
                        d_losses_test.append(round(LPU.external_libs.DEDPUL.NN_functions.d_loss_standard(torch.as_tensor(mix_data_test, dtype=LPU.constants.DTYPE),
                                                                torch.as_tensor(pos_data_test, dtype=LPU.constants.DTYPE),
                                                                discriminator).item(), 5))
                elif training_mode == 'nnre':
                    d_losses_test.append(round(LPU.external_libs.DEDPUL.NN_functions.d_loss_nnRE(torch.as_tensor(mix_data_test, dtype=LPU.constants.DTYPE),
                                                        torch.as_tensor(pos_data_test, dtype=LPU.constants.DTYPE),
                                                        discriminator, nnre_alpha).item(), 5))
                if metric is not None:
                    d_metrics_test.append(metric(target_test,
                                                discriminator(torch.as_tensor(data_test, dtype=LPU.constants.DTYPE)).detach().numpy()))


                if epoch >= n_early_stop:
                    if_stop = True
                    for i in range(n_early_stop):
                        if metric is not None and stop_by_metric:
                            if d_metrics_test[-i - 1] < d_metrics_test[-n_early_stop - 1]:
                                if_stop = False
                                break
                        else:
                            if d_losses_val[-i-1] < d_losses_val[-n_early_stop-1]:
                                if_stop = False
                                break
                    if if_stop:
                        break
            elif disp:
                print('epoch', epoch, ', train_loss=', d_losses_train[-1], ', val_loss=', d_losses_val[-1])

        discriminator.eval()

        return d_losses_train, d_losses_val, d_losses_test

    @abstractmethod
    def modified_estimate_preds_cv(df, target, test_df, test_target, cv=3, n_networks=1, lr=1e-4, hid_dim=32, n_hid_layers=1,
                        random_state=None, training_mode='standard', alpha=None, l2=1e-4, train_nn_options=None,
                        all_conv=False, bayes=False, bn=True):
        """ ** NOTE ** Identical to LPU.external_libs.DEDPUL.algorithms.estimate_preds_cv, but with some modifications to output the discriminator
            as well

        Estimates posterior probability y(x) of belonging to U rather than P (ignoring relative sizes of U and P);
            predictions are the average of an ensemble of n_networks neural networks;
            performs cross-val predictions to cover the whole dataset
        :param df: features, np.array (n_instances, n_features)
        :param target: binary vector, 0 if positive, 1 if unlabeled, np.array with shape (n,)
        :param cv: number of folds, int
        :param n_networks: number of neural networks in the ensemble to average results of
        :param lr: learning rate, float
        :param hid_dim: number of neurons in each hidden layer
        :param n_hid_layers: number of hidden layers in each network
        :param random_state: seed, used in data kfold split, default is None
        :param alpha: share of N in U
        :param train_nn_options: parameters for train_NN

        :return: predicted probabilities y(x) of belonging to U rather than P (ignoring relative sizes of U and P)
        """

        if train_nn_options is None:
            train_nn_options = dict()

        preds = np.zeros((n_networks, df.shape[0],))
        means = np.zeros((n_networks, df.shape[0],))
        variances = np.zeros((n_networks, df.shape[0],))

        preds_test = np.zeros((n_networks, test_df.shape[0],))
        means_test = np.zeros((n_networks, test_df.shape[0],))
        variances_test = np.zeros((n_networks, test_df.shape[0],))

        import torch.optim as optim
        mix_data_test = test_df[test_target == 1]
        pos_data_test = test_df[test_target == 0]

        all_d_loss_train = []
        all_d_loss_val = []
        all_d_loss_test = []
        for i in range(n_networks):
            kf = sklearn.model_selection.StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            for train_index, val_index in kf.split(df, target):
                train_data = df[train_index]
                train_target = target[train_index]
                mix_data = train_data[train_target == 1]
                pos_data = train_data[train_target == 0]
                val_data = df[val_index]
                val_target = target[val_index]

                mix_data_val = val_data[val_target == 1]
                pos_data_val = val_data[val_target == 0]
                if not all_conv:
                    discriminator = LPU.external_libs.DEDPUL.algorithms.get_discriminator(inp_dim=df.shape[1], out_dim=1, hid_dim=hid_dim,
                                                    n_hid_layers=n_hid_layers, bayes=bayes, bn=bn)
                else:
                    discriminator = LPU.external_libs.DEDPUL.algorithms.all_convolution(hid_dim_full=hid_dim, bayes=bayes, bn=bn)
                discriminator.to(LPU.constants.DTYPE)
                d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=l2)

                
                d_losses_train, d_losses_val, d_losses_test = DEDPUL.modified_train_NN(mix_data=mix_data, pos_data=pos_data,discriminator=discriminator,d_optimizer=d_optimizer,
                        mix_data_val=mix_data_val, pos_data_val=pos_data_val, mix_data_test=mix_data_test, pos_data_test=pos_data_test,
                        nnre_alpha=alpha,

                        d_scheduler=None, training_mode=training_mode, bayes=bayes, **train_nn_options)
                if len(all_d_loss_train):
                    min_length = min(all_d_loss_train.shape[1], len(d_losses_train))

                    all_d_loss_train = np.vstack([all_d_loss_train[:, :min_length], np.asarray([d_losses_train[:min_length]])])
                    all_d_loss_val = np.vstack([all_d_loss_val[:, :min_length], np.asarray([d_losses_val[:min_length]])])
                    all_d_loss_test = np.vstack([all_d_loss_test[:, :min_length], np.asarray([d_losses_test[:min_length]])])

                else:
                    # adding a new axis to make it a 2D array so that we can stack them
                    # in the first clause of the if statement
                    all_d_loss_train = np.array([d_losses_train])
                    all_d_loss_val = np.array([d_losses_val])
                    all_d_loss_test = np.array([d_losses_test])

                
                if bayes:
                    pred, mean, var = discriminator(
                        torch.as_tensor(val_data, dtype=LPU.constants.DTYPE), return_params=True, sample_noise=False)
                    preds[i, val_index], means[i, val_index], variances[i, val_index] = \
                        pred.detach().numpy().flatten(), mean.detach().numpy().flatten(), var.detach().numpy().flatten()
                else:
                    preds[i, val_index] = discriminator(
                        torch.as_tensor(val_data, dtype=LPU.constants.DTYPE)).detach().numpy().flatten()

                if bayes:
                    pred, mean, var = discriminator(
                        torch.as_tensor(test_df, dtype=LPU.constants.DTYPE), return_params=True, sample_noise=False)
                    preds_test[i, :], means_test[i, :], variances_test[i, :] = \
                        pred.detach().numpy().flatten(), mean.detach().numpy().flatten(), var.detach().numpy().flatten()
                else:
                    preds_test[i, :] = discriminator(
                        torch.as_tensor(test_df, dtype=LPU.constants.DTYPE)).detach().numpy().flatten()
        
        preds = preds.mean(axis=0)
        preds_test = preds_test.mean(axis=0)

        loss_dict = {}
        loss_dict['train'] = np.mean(all_d_loss_train, axis=0) 
        loss_dict['val'] = np.mean(all_d_loss_val, axis=0)
        loss_dict['test'] = np.mean(all_d_loss_test, axis=0)

        if bayes:
            means, variances = means.mean(axis=0), variances.mean(axis=0)
            means_test, variances_test = means_test.mean(axis=0), variances_test.mean(axis=0)
            return (preds, means, variances), (preds_test, means_test, variances_test), loss_dict
        else:
            return preds, preds_test, loss_dict
        
    def modified_estimate_poster_em(diff=None, preds=None, target=None, mode='dedpul', converge=True, tol=10**-5,
                        max_iterations=1000, nonconverge=True, step=0.001, max_diff=0.05, plot=False, disp=False,
                        alpha=None, alpha_as_mean_poster=True, **kwargs):
        """
        identical to https://github.com/dimonenka/DEDPUL/blob/04c028101a509b2efe3d55de457b5df92439bb59/algorithms.py#L371
        except that the model variable types has been changed to LPU.constants.DTYPE
        corrected method locations since they're being imported from LPU.external_libs.DEDPUL.algorithms
        and also removed plotting logic


        Performs Expectation-Maximization to estimate posteriors and priors alpha (if not provided) of N in U
            with either of 'en' or 'dedpul' methods; both 'converge' and 'nonconverge' are recommended to be set True for
            better estimate
        :param diff: difference of densities f_p/f_u for the sample U, np.array (n,), output of estimate_diff()
        :param preds: predictions of classifier, np.array with shape (n,)
        :param target: binary vector, 0 if positive, 1 if unlabeled, np.array with shape (n,)
        :param mode: 'dedpul' or 'en'; if 'dedpul', diff needs to be provided; if 'en', preds and target need to be provided
        :param converge: True or False; True if convergence estimate should be computed
        :param tol: tolerance of error between priors and mean posteriors, indicator of convergence
        :param max_iterations: if exceeded, search of converged alpha stops even if tol is not reached
        :param nonconverge: True or False; True if non-convergence estimate should be computed
        :param step: gap between points of the [0, 1, step] gird to choose best alpha from
        :param max_diff: alpha with difference of mean posteriors and priors bigger than max_diff cannot be chosen;
            an heuristic to choose bigger alpha
        :param plot: True or False, if True - plots ([0, 1, grid], mean posteriors - alpha) and
            ([0, 1, grid], second lag of (mean posteriors - alpha))
        :param disp: True or False, if True - displays if the algorithm didn't converge
        :param alpha: proportions of N in U; is estimated if None
        :return: tuple (alpha, poster), e.g. (priors, posteriors) of N in U for the U sample
        """
        assert converge + nonconverge, "At least one of 'converge' and 'nonconverge' has to be set to 'True'"

        if alpha is not None:
            if mode == 'dedpul':
                alpha, poster = LPU.external_libs.DEDPUL.algorithms.estimate_poster_dedpul(diff, alpha=alpha, alpha_as_mean_poster=alpha_as_mean_poster, tol=tol, **kwargs)
            elif mode == 'en':
                _, poster = LPU.external_libs.DEDPUL.algorithms.estimate_poster_en(preds, target, alpha=alpha, **kwargs)
            return alpha, poster

        # if converge:
        alpha_converge = 0
        for i in range(max_iterations):

            if mode.endswith('dedpul'):
                _, poster_converge = LPU.external_libs.DEDPUL.algorithms.estimate_poster_dedpul(diff, alpha=alpha_converge, **kwargs)
            elif mode == 'en':
                _, poster_converge = LPU.external_libs.DEDPUL.algorithms.estimate_poster_en(preds, target, alpha=alpha_converge, **kwargs)

            mean_poster = np.mean(poster_converge)
            error = mean_poster - alpha_converge

            if np.abs(error) < tol:
                break
            if np.min(poster_converge) > 0:
                break
            alpha_converge = mean_poster

        if disp:
            if i >= max_iterations - 1:
                print('max iterations exceeded')

        # if nonconverge:

        errors = np.array([])
        for alpha_nonconverge in np.arange(0, 1, step):

            if mode.endswith('dedpul'):
                _, poster_nonconverge = LPU.external_libs.DEDPUL.algorithms.estimate_poster_dedpul(diff, alpha=alpha_nonconverge, **kwargs)
            elif mode == 'en':
                _, poster_nonconverge = LPU.external_libs.DEDPUL.algorithms.estimate_poster_en(preds, target, alpha=alpha_nonconverge, **kwargs)
            errors = np.append(errors, np.mean(poster_nonconverge) - alpha_nonconverge)

        idx = np.argmax(np.diff(np.diff(errors))[errors[1: -1] < max_diff])
        alpha_nonconverge = np.arange(0, 1, step)[1: -1][errors[1: -1] < max_diff][idx]


        if ((alpha_nonconverge >= alpha_converge) or#converge and nonconverge and
            (((errors < 0).sum() > 1) and (alpha_converge < 1 - step))):
            return alpha_converge, poster_converge

        elif nonconverge:
            if mode == 'dedpul':
                _, poster_nonconverge = LPU.external_libs.DEDPUL.algorithms.estimate_poster_dedpul(diff, alpha=alpha_nonconverge, **kwargs)
            elif mode == 'en':
                _, poster_nonconverge = LPU.external_libs.DEDPUL.algorithms.estimate_poster_en(preds, target, alpha=alpha_nonconverge, **kwargs)

            if disp:
                print('didn\'t converge')
            return alpha_nonconverge, poster_nonconverge
            # return np.mean(poster_nonconverge), poster_nonconverge

        else:
            if disp:
                print('didn\'t converge')
            return None, None        


    @abstractmethod
    def modified_estimate_poster_cv(df, target, test_df, test_target, estimator='dedpul', bayes=False, alpha=None, lr=None, estimate_poster_options=None,
                        estimate_diff_options=None, estimate_preds_cv_options=None, train_nn_options=None, cv=None):
        """
        identical to 
        https://github.com/dimonenka/DEDPUL/blob/04c028101a509b2efe3d55de457b5df92439bb59/algorithms.py#L469

        just corrected module locations since they're being imported from LPU.external_libs.DEDPUL.algorithms
        or third party libraries with absolute imports. also simplified so the estimator=='dedpul' 
        and other cases are removed

        Estimates posteriors and priors alpha (if not provided) of N in U; f_u(x) = (1 - alpha) * f_p(x) + alpha * f_n(x)
        :param df: features, np.array (n_instances, n_features)
        :param target: binary vector, 0 if positive, 1 if unlabeled, np.array with shape (n,)
        :param estimator: 'dedpul', 'baseline_dedpul', 'random_dedpul ,'en', 'em_en', or 'nnre';
            'ntc_methods' for every estimate but 'nnre'
        :param alpha: share of N in U; is estimated if not provided (nnRE requires it to be provided)
        :param estimate_poster_options: parameters for estimate_poster... functions
        :param estimate_diff_options: parameters for estimate_diff
        :param estimate_preds_cv_options: parameters for estimate_preds_cv
        :param train_nn_options: parameters for train_NN
        :return: if estimator != 'ntc_methods':
            tuple (alpha, poster), e.g. (priors, posteriors) of N in U for the U sample df[target == 1]
            if estimator == 'ntc_methods':
            dictionary with such (alpha, poster) tuples as values and method names as keys
        """

        if isinstance(df, pd.DataFrame):
            df = df.values
        if isinstance(target, pd.Series):
            target = target.values
        training_mode = 'standard'

        if train_nn_options is None:
            train_nn_options = dict()

        if estimate_poster_options is None:
            estimate_poster_options = dict()

        if estimate_diff_options is None:
            estimate_diff_options = dict()

        if estimate_preds_cv_options is None:
            estimate_preds_cv_options = dict()

        # preds = estimate_preds_cv_catboost(df, target, **estimate_preds_cv_options)
        ### uncomment the line above and comment the line below for experiments with catboost instead of neural networks
        preds, preds_test, loss_dict = DEDPUL.modified_estimate_preds_cv(df=df, cv=cv, target=target, test_df=test_df, test_target=test_target, alpha=alpha, training_mode=training_mode, bayes=bayes,
                                train_nn_options=train_nn_options, lr=lr,  **estimate_preds_cv_options)
        if bayes:
            (preds, means, variances), (preds_test, means_test, variances_test), loss_dict = preds, preds_test, loss_dict

        if bayes:
            diff = LPU.external_libs.DEDPUL.algorithms.estimate_diff_bayes(means, variances, target, **estimate_diff_options)
            diff_test = LPU.external_libs.DEDPUL.algorithms.estimate_diff_bayes(means_test, variances_test, target, **estimate_diff_options)
        else:
            diff = LPU.external_libs.DEDPUL.algorithms.estimate_diff(preds, target, **estimate_diff_options)
            diff_test = LPU.external_libs.DEDPUL.algorithms.estimate_diff(preds_test, test_target, **estimate_diff_options)

        alpha, poster = DEDPUL.modified_estimate_poster_em(diff=diff, mode='dedpul', alpha=alpha, **estimate_poster_options)
        _, poster_test = DEDPUL.modified_estimate_poster_em(diff=diff_test, mode='dedpul', alpha=alpha, **estimate_poster_options)
        return alpha, poster, poster_test, preds, preds_test, loss_dict
        
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.loss_type = config.get('loss_type', None)
        self.kde_mode = config.get('kde_mode', None)
        super().__init__(**kwargs)
        self.discriminator = None
        self.nrep = config.get('nrep', 1)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.n_neurons = config.get('n_neurons', 32)
        self.MT_coef = self.config.get('MT_coef', 0.25)
        self.bw_mix = self.config.get('bw_mix', 0.05)
        self.bw_pos = self.config.get('bw_pos', 0.1)
        self.dedepul_type = self.config['dedpul_type']

    def train(self, train_dataloader, val_dataloader, test_dataloader, train_nn_options):
        all_X_train = []
        all_l_train = []
        all_y_train = []
        for X_train, l_train, y_train, _ in train_dataloader:
            all_X_train.append(X_train)
            all_l_train.append(l_train)
            all_y_train.append(y_train)
        all_X_train = torch.vstack(all_X_train)
        all_l_train = torch.hstack(all_l_train)
        all_y_train = torch.hstack(all_y_train)

        
        all_X_val = []
        all_l_val = []  
        all_y_val = []
        for X_val, l_val, y_val, _ in val_dataloader:
            all_X_val.append(X_val)
            all_l_val.append(l_val)
            all_y_val.append(y_val)
        all_X_val = torch.vstack(all_X_val)
        all_l_val = torch.hstack(all_l_val)
        all_y_val = torch.hstack(all_y_val)

        all_X_test = []
        all_l_test = []
        all_y_test = []
        for X_test, l_test, y_test, _ in test_dataloader:
            all_X_test.append(X_test)
            all_l_test.append(l_test)
            all_y_test.append(y_test)
        all_X_test = torch.vstack(all_X_test)
        all_l_test = torch.hstack(all_l_test)
        all_y_test = torch.hstack(all_y_test)

        train_size = len(all_X_train)
        val_size = len(all_X_val)
        test_size = len(all_X_test)

        concat_X = torch.vstack([all_X_train, all_X_val])
        concat_l = torch.hstack([all_l_train, all_l_val])
        
        # inverting the labels as DEDPUL assume unlabeled data has target==1,
        # while LPU datasets have lebel==0 for unlabeled data
        concat_l = 1 - concat_l
        all_l_test = 1 - all_l_test
        
        # setting the num_epochs to be what is passed in config file
        train_nn_options['n_epochs'] = self.config.get('num_epochs')
        train_nn_options['batch_size'] = self.config['batch_size']['train']

        alpha_posterior, n_in_unlabeled_posterior, n_in_unlabeled_posterior_test, preds,  preds_test, loss_dict = DEDPUL.modified_estimate_poster_cv(df=concat_X, target=concat_l, 
                                                                                                                                                     test_df=all_X_test, test_target=all_l_test,
                                                                                                                                                     estimator='dedpul', bayes=False, alpha=None, lr=self.config['learning_rate'],
                                                                                              estimate_diff_options=self.config['estimate_diff_options'],
                                                                                            train_nn_options=train_nn_options, cv=self.config['cv'])

        # inverting the labels again to match the LPU dataset assumption,
        # i.e. label==0 for unlabeled data
        concat_l = 1 - concat_l
        all_l_test = 1 - all_l_test
        
        y_probs = 1 - preds
        # concat_l == 0 since the value of unlabeledness is inverted
        y_probs[concat_l == 0] = 1 - n_in_unlabeled_posterior
        l_probs = 1 - preds

        y_probs_test = 1 - preds_test
        y_probs_test[all_l_test == 0] = 1 - n_in_unlabeled_posterior_test
        l_probs_test = 1 - preds_test

        l_ests = l_probs > .5
        y_ests = y_probs > .5
        l_ests_test = l_probs_test > .5
        y_ests_test = y_probs_test > .5


        metrics = dict()
        metrics['train'] = self._calculate_validation_metrics(y_probs[:train_size], all_y_train, l_probs[:train_size], all_l_train, l_ests=l_ests[:train_size], y_ests=y_ests[:train_size])
        metrics['val'] = self._calculate_validation_metrics(y_probs[train_size:train_size+val_size], all_y_val, l_probs[train_size:train_size+val_size], all_l_val, l_ests=l_ests[train_size:train_size+val_size], y_ests=y_ests[train_size:train_size+val_size])
        metrics['test'] = self._calculate_validation_metrics(y_probs=y_probs_test, y_vals=all_y_test, l_probs=l_probs_test, l_vals=all_l_test, l_ests=l_ests_test, y_ests=y_ests_test)
        for key in metrics:
            metrics[key].update({'overall_loss': loss_dict[key][-1]})
        return metrics

        
    

    def loss_fn(self, X_batch, l_batch):
        return LPU.external_libs.DEDPUL.NN_functions.d_loss_standard(X_batch[l_batch == 1], X_batch[l_batch == 0], self.discriminator, self.loss_type)
    

    # def validate(self, dataloader, holdoutloader):
    #     y_probs = []
    #     l_probs = []
    #     y_vals = []
    #     l_vals = []
    #     y_ests = []
    #     l_ests = []
    #     losses = []
    #     # self.set_C(dataloader)
    #     holdout_all_X = []
    #     holdout_all_l = []
    #     holdout_all_y = []
    #     for X_val, l_val, _, _ in holdoutloader:
    #         holdout_all_X.append(X_val)
    #         holdout_all_l.append(l_val)
    #     holdout_all_X = torch.vstack(holdout_all_X)
    #     holdout_all_l = torch.hstack(holdout_all_l).detach()
    #     holdout_all_X = holdout_all_X[holdout_all_l == 1]
    #     holdout_all_l = torch.zeros(((holdout_all_l==1).sum().to(int)))

    #     all_X = []
    #     all_l = []
    #     all_y = []
    #     for X_val, l_val, y_val, idx_val in dataloader:
    #         all_X.append(X_val)
    #         all_l.append(l_val)
    #         all_y.append(y_val)
    #     all_X = torch.vstack(all_X)
    #     all_y = torch.hstack(all_y).detach().cpu().numpy()
    #     all_l = torch.hstack(all_l).detach()
    #     all_l = torch.ones_like(all_l)

    #     all_X = torch.vstack((holdout_all_X, all_X))
    #     all_l = torch.hstack((holdout_all_l, all_l))
    #     preds = self.discriminator(all_X).detach().cpu().numpy().flatten()
    #     diff = LPU.external_libs.DEDPUL.algorithms.estimate_diff(preds=preds, target=all_l, bw_mix=self.bw_mix, bw_pos=self.bw_pos, kde_mode=self.kde_mode, threshold=self.threshold,
    #                 MT=False, MT_coef=self.MT_coef, tune=False, decay_MT_coef=False, n_gauss_mix=20, n_gauss_pos=10,bins_mix=20, bins_pos=20, k_neighbours=None)
    #     self.alpha, poster =  LPU.external_libs.DEDPUL.algorithms.estimate_poster_em(diff, preds, self.dedepul_type, alpha_as_mean_poster=True)
    #     y_probs = poster[len(holdout_all_l):]
    #     l_probs = preds[len(holdout_all_l):]
    #     l_vals = all_l[len(holdout_all_l):]
    #     y_vals = all_y[len(holdout_all_l):]
    #     # self.C = self.holdout_l_mean * (1-self.alpha)
    #     # y_prob = self.modified_estimate_preds_cv(X_val, 1-l_val, train_nn_options=None)
    #     # if hasattr(self, 'loss_fn'):
    #     #     losses.append(self.loss_fn(all_X, all_l).item())
    #     # else:
    #     #     losses.append(0)

    #     # y_probs = np.hstack(y_probs)
    #     # y_vals = np.hstack(y_vals).astype(int)
    #     # l_probs = np.hstack(l_probs)
    #     # l_vals = np.hstack(l_vals).astype(int)
    #     l_ests = l_probs > self.threshold
    #     y_ests = y_probs > .5

    #     validation_results = self._calculate_validation_metrics(
    #         y_probs, y_vals, l_probs, l_vals, l_ests=l_ests, y_ests=y_ests)
    #     validation_results.update({'overall_loss': np.mean(losses)})
    #     return validation_results
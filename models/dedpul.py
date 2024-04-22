from abc import abstractmethod
import logging
import random

import numpy as np
import pandas as pd
import sklearn.model_selection
import torch

import lpu.constants
import lpu.external_libs
import lpu.models.geometric.elkanGGPC
import lpu.models.lpu_model_base

LOG = logging.getLogger(__name__)

EPSILON = 1e-16




class DEDPUL(lpu.models.lpu_model_base.LPUModelBase):
    """
    Using estimator of p(s|X) to predict p(y|X)
    """
    @abstractmethod
    def modified_train_NN(mix_data, pos_data, discriminator, d_optimizer, mix_data_test=None, pos_data_test=None,
                n_epochs=200, batch_size=64, n_batches=None, n_early_stop=5,
                d_scheduler=None, training_mode='standard', disp=False, loss_function=None, nnre_alpha=None,
                metric=None, stop_by_metric=False, bayes=False, bayes_weight=1e-5, beta=0, gamma=1):
        """ ** NOTE **: Identical to the lpu.external_libs.DEDPUL.NN_functions.train_NN, but with some modifications to set the 
            type of variables in the model to global constant value lpu.constants.DTYPE
        Train discriminator to classify mix_data from pos_data.
        """
        d_losses_train = []
        d_losses_test = []
        d_metrics_test = []
        if n_batches is None:
            n_batches = min(int(mix_data.shape[0] / batch_size), int(pos_data.shape[0] / batch_size))
            batch_size_mix = batch_size_pos = batch_size
        else:
            batch_size_mix, batch_size_pos = int(mix_data.shape[0] / n_batches), int(pos_data.shape[0] / n_batches)
        if mix_data_test is not None:
            data_test = np.concatenate((pos_data_test, mix_data_test))
            target_test = np.concatenate((np.zeros((pos_data_test.shape[0],)), np.ones((mix_data_test.shape[0],))))

        for epoch in range(n_epochs):

            discriminator.train()

            d_losses_cur = []
            if d_scheduler is not None:
                d_scheduler.step()

            for i in range(n_batches):

                batch_mix = np.array(random.sample(list(mix_data), batch_size_mix))
                batch_pos = np.array(random.sample(list(pos_data), batch_size_pos))

                batch_mix = torch.as_tensor(batch_mix, dtype=lpu.constants.DTYPE)
                batch_pos = torch.as_tensor(batch_pos, dtype=lpu.constants.DTYPE)

                # Optimize D
                d_optimizer.zero_grad()

                if training_mode == 'standard':
                    if bayes:
                        loss = lpu.external_libs.DEDPUL.NN_functions.d_loss_bayes(batch_mix, batch_pos, discriminator, loss_function, bayes_weight)
                    else:
                        loss = lpu.external_libs.DEDPUL.NN_functions.d_loss_standard(batch_mix, batch_pos, discriminator, loss_function)

                else:
                    loss = lpu.external_libs.DEDPUL.NN_functions.d_loss_nnRE(batch_mix, batch_pos, discriminator, nnre_alpha, beta=beta, gamma=gamma,
                                    loss_function=loss_function)

                loss.backward()
                d_optimizer.step()
                d_losses_cur.append(loss.cpu().item())

            d_losses_train.append(round(np.mean(d_losses_cur).item(), 5))

            if mix_data_test is not None and pos_data_test is not None:

                discriminator.eval()

                if training_mode == 'standard':
                    if bayes:
                        d_losses_test.append(round(lpu.external_libs.DEDPUL.NN_functions.d_loss_bayes(torch.as_tensor(mix_data_test, dtype=lpu.constants.DTYPE),
                                                                torch.as_tensor(pos_data_test, dtype=lpu.constants.DTYPE),
                                                                discriminator, w=bayes_weight).item(), 5))
                    else:
                        d_losses_test.append(round(lpu.external_libs.DEDPUL.NN_functions.d_loss_standard(torch.as_tensor(mix_data_test, dtype=lpu.constants.DTYPE),
                                                                torch.as_tensor(pos_data_test, dtype=lpu.constants.DTYPE),
                                                                discriminator).item(), 5))
                elif training_mode == 'nnre':
                    d_losses_test.append(round(lpu.external_libs.DEDPUL.NN_functions.d_loss_nnRE(torch.as_tensor(mix_data_test, dtype=lpu.constants.DTYPE),
                                                        torch.as_tensor(pos_data_test, dtype=lpu.constants.DTYPE),
                                                        discriminator, nnre_alpha).item(), 5))
                if metric is not None:
                    d_metrics_test.append(metric(target_test,
                                                discriminator(torch.as_tensor(data_test, dtype=lpu.constants.DTYPE)).detach().numpy()))

                if disp:
                    if not metric:
                        print('epoch', epoch, ', train_loss=', d_losses_train[-1], ', test_loss=', d_losses_test[-1])
                    else:
                        print('epoch', epoch, ', train_loss=', d_losses_train[-1], ', test_loss=', d_losses_test[-1],
                            'test_metric=', d_metrics_test[-1])

                if epoch >= n_early_stop:
                    if_stop = True
                    for i in range(n_early_stop):
                        if metric is not None and stop_by_metric:
                            if d_metrics_test[-i - 1] < d_metrics_test[-n_early_stop - 1]:
                                if_stop = False
                                break
                        else:
                            if d_losses_test[-i-1] < d_losses_test[-n_early_stop-1]:
                                if_stop = False
                                break
                    if if_stop:
                        break
            elif disp:
                print('epoch', epoch, ', train_loss=', d_losses_train[-1])

        discriminator.eval()

        return d_losses_train, d_losses_test

    @abstractmethod
    def modified_estimate_preds_cv(df, target, cv=3, n_networks=1, lr=1e-4, hid_dim=32, n_hid_layers=1,
                        random_state=None, training_mode='standard', alpha=None, l2=1e-4, train_nn_options=None,
                        all_conv=False, bayes=False, bn=True):
        """ ** NOTE ** Identical to lpu.external_libs.DEDPUL.algorithms.estimate_preds_cv, but with some modifications to output the discriminator
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

        import torch.optim as optim

        for i in range(n_networks):
            kf = sklearn.model_selection.StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
            for train_index, test_index in kf.split(df, target):
                train_data = df[train_index]
                train_target = target[train_index]
                mix_data = train_data[train_target == 1]
                pos_data = train_data[train_target == 0]
                test_data = df[test_index]
                test_target = target[test_index]

                mix_data_test = test_data[test_target == 1]
                pos_data_test = test_data[test_target == 0]
                if not all_conv:
                    discriminator = lpu.external_libs.DEDPUL.algorithms.get_discriminator(inp_dim=df.shape[1], out_dim=1, hid_dim=hid_dim,
                                                    n_hid_layers=n_hid_layers, bayes=bayes, bn=bn)
                else:
                    discriminator = lpu.external_libs.DEDPUL.algorithms.all_convolution(hid_dim_full=hid_dim, bayes=bayes, bn=bn)
                d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=l2)

                DEDPUL.modified_train_NN(mix_data, pos_data, discriminator, d_optimizer,
                        mix_data_test, pos_data_test, nnre_alpha=alpha,
                        d_scheduler=None, training_mode=training_mode, bayes=bayes, **train_nn_options)
                if bayes:
                    pred, mean, var = discriminator(
                        torch.as_tensor(test_data, dtype=lpu.constants.DTYPE), return_params=True, sample_noise=False)
                    preds[i, test_index], means[i, test_index], variances[i, test_index] = \
                        pred.detach().numpy().flatten(), mean.detach().numpy().flatten(), var.detach().numpy().flatten()
                else:
                    preds[i, test_index] = discriminator(
                        torch.as_tensor(test_data, dtype=lpu.constants.DTYPE)).detach().numpy().flatten()

        preds = preds.mean(axis=0)
        if bayes:
            means, variances = means.mean(axis=0), variances.mean(axis=0)
            return preds, means, variances
        else:
            return preds, discriminator 
    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.loss_type = config.get('loss_type', None)
        self.kde_mode = config.get('kde_mode', None)
        super().__init__(**kwargs)
        self.descriminator = None
        self.nrep = config.get('nrep', 1)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.n_neurons = config.get('n_neurons', 32)

    def set_C(self, holdout_dataloader, preds, dedepul_type='dedpul'):
        try:
            assert (holdout_dataloader.batch_size == len(holdout_dataloader.dataset)), "There should be only one batch in the dataloader."
        except AssertionError as e:
            LOG.error(f"There should be only one batch in the dataloader, but {holdout_dataloader.batch_size} is smaller than {len(holdout_dataloader.dataset)}.")
            raise e
        X, l, _ = next(iter(holdout_dataloader))
        X = X.numpy()
        X = pd.DataFrame(X)
        threshold = self.config.get('threshold', preds[l==1].mean()+preds[l==0].mean()/2)
        MT_coef = self.config.get('MT_coef', 0.25)
        bw_mix = self.config.get('bw_mix', 0.05)
        bw_pos = self.config.get('bw_pos', 0.1)
        diff = lpu.external_libs.DEDPUL.algorithms.estimate_diff(preds, l, bw_mix=bw_mix, bw_pos=bw_pos, kde_mode=self.kde_mode, threshold=threshold,
                     MT=False, MT_coef=MT_coef, tune=False, decay_MT_coef=False, n_gauss_mix=20, n_gauss_pos=10,bins_mix=20, bins_pos=20, k_neighbours=None)
        self.alpha, _ =  lpu.external_libs.DEDPUL.algorithms.estimate_poster_em(diff, preds, dedepul_type, alpha_as_mean_poster=True)

        self.C = l.mean() / (1-self.alpha)

    def train(self, dataloader):
        try:
            assert (dataloader.batch_size == len(dataloader.dataset)), "There should be only one batch in the dataloader."
        except AssertionError as e:
            LOG.error(f"There should be only one batch in the dataloader, but {dataloader.batch_size} is smaller than {len(dataloader.dataset)}.")
            raise e
        X, l, _ = next(iter(dataloader))
        preds, self.discriminator = DEDPUL.modified_estimate_preds_cv(X, l, train_nn_options={
                                        'n_epochs': 250, 'loss_function': 'log', 'batch_size': 64,
                                        'n_batches': None, 'n_early_stop': 7, 'disp': False}, lr=self.learning_rate,
                                )   
        return preds

    def predict_proba(self, X):
        self.discriminator.eval()
        return self.discriminator(torch.as_tensor(X, dtype=lpu.constants.DTYPE)).detach().numpy().flatten()

    def predict_prob_y_given_x(self, X):
        return self.predict_proba(X) / self.C
    
    def predict_prob_l_given_y_x(self, X):
        return self.C 

    def loss_fn(self, X_batch, l_batch):
        return lpu.external_libs.DEDPUL.NN_functions.d_loss_standard(X_batch[l_batch == 0], X_batch[l_batch == 1], self.discriminator, self.loss_type)
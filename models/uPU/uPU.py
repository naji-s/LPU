"""
Implementation of the nnPU model and uPU model by inheritance from the LPUModelBase class.
Despite this, becasue he code uses Chainer, the details of traininig are not adapted other
pytorch models in this library. That is a future TODO (naji)
"""

import logging
import sys

import LPU.external_libs.nnPUlearning
# sys.path.append('LPU/external_libs/nnPUSB')

import numpy as np
import torch
import scipy.special

import LPU.constants
import LPU.external_libs
import LPU.models.geometric.Elkan.Elkan
import LPU.models.lpu_model_base
import LPU.models.uPU.uPU

DEFAULT_CONFIG = {
    "device": "cpu",
    "dataset_name": "animal_no_animal",
    "labeled": 0,
    "unlabeled": 0,
    "epoch": 100,
    "beta": 0.0,
    "gamma": 1.0,
    "learning_rate": 0.0001,
    "loss": "sigmoid",
    "model": "mlp",
    "out": "LPU/scripts/uPU/checkpoints",
    "data_generating_process": "SB",  # either of CC (case-control) or SB (selection-bias)
    "dataset_kind": "LPU",
    "batch_size": {
        "train": 64,
        "test": None,
        "val": None,
        "holdout": None
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

}

LOG = logging.getLogger(__name__)

EPSILON = 1e-16

import numpy as np
import torch
from torch import nn
import torch.nn.functional


class MyClassifier(torch.nn.Module):

    def zero_one_loss(self, h, t, is_logistic=False):
        self.eval()
        positive = 1
        negative = 0 if is_logistic else -1

        n_p = (t == positive).sum()
        n_n = (t == negative).sum()
        size = n_p + n_n

        t_p = ((h == positive) * (t == positive)).sum()
        t_n = ((h == negative) * (t == negative)).sum()
        f_p = n_n - t_n
        f_n = n_p - t_p

        # print("size:{0},t_p:{1},t_n:{2},f_p:{3},f_n:{4}".format(
        #     size, t_p, t_n, f_p, f_n))

        presicion = (0.0 if t_p == 0 else t_p/(t_p+f_p))
        recall = (0.0 if t_p == 0 else t_p/(t_p+f_n))

        return presicion, recall, 1 - (t_p+t_n)/size

    def error(self, DataLoader, is_logistic=False, device='cpu'):
        presicion = []
        recall = []
        error_rate = []
        self.eval()
        for data, target in DataLoader:
            data = data.to(device, non_blocking=True)
            t = target.detach().cpu().numpy()
            size = len(t)
            if is_logistic:
                h = np.reshape(torch.sigmoid(
                    self(data)).detach().cpu().numpy(), size)
                h = np.where(h > 0.5, 1, 0).astype(np.int32)
            else:
                h = np.reshape(torch.sign(
                    self(data)).detach().cpu().numpy(), size)

            result = self.zero_one_loss(h, t, is_logistic)
            presicion.append(result[0])
            recall.append(result[1])
            error_rate.append(result[2])

        return sum(presicion)/len(presicion), sum(recall)/len(recall), sum(error_rate)/len(error_rate)


class ThreeLayerPerceptron(MyClassifier, nn.Module):
    def __init__(self, dim):
        super(ThreeLayerPerceptron, self).__init__()

        self.input_dim = dim
        self.l1 = nn.Linear(dim, 100)
        self.l2 = nn.Linear(100, 1)

        self.af = torch.nn.functional.relu

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.l1(x)
        x = self.af(x)
        x = self.l2(x)
        return x


class MultiLayerPerceptron(MyClassifier, nn.Module):
    def __init__(self, dim):
        super(MultiLayerPerceptron, self).__init__()

        self.input_dim = dim

        self.l1 = nn.Linear(dim, 300, bias=False)
        self.b1 = nn.BatchNorm1d(300)
        self.l2 = nn.Linear(300, 300, bias=False)
        self.b2 = nn.BatchNorm1d(300)
        self.l3 = nn.Linear(300, 300, bias=False)
        self.b3 = nn.BatchNorm1d(300)
        self.l4 = nn.Linear(300, 300, bias=False)
        self.b4 = nn.BatchNorm1d(300)
        self.l5 = nn.Linear(300, 1)
        self.af = torch.nn.functional.relu

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        h = self.l1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.l2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.l3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.l4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.l5(h)
        return h


class CNN(MyClassifier, nn.Module):
    def __init__(self, dim):
        super(CNN, self).__init__()

        self.af = torch.nn.functional.relu
        self.input_dim = dim

        self.conv1 = nn.Conv2d(3, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(192, 10, 1)
        self.b1 = nn.BatchNorm2d(96)
        self.b2 = nn.BatchNorm2d(96)
        self.b3 = nn.BatchNorm2d(96)
        self.b4 = nn.BatchNorm2d(192)
        self.b5 = nn.BatchNorm2d(192)
        self.b6 = nn.BatchNorm2d(192)
        self.b7 = nn.BatchNorm2d(192)
        self.b8 = nn.BatchNorm2d(192)
        self.b9 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(640, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.conv2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.conv3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.conv4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.conv5(h)
        h = self.b5(h)
        h = self.af(h)
        h = self.conv6(h)
        h = self.b6(h)
        h = self.af(h)
        h = self.conv7(h)
        h = self.b7(h)
        h = self.af(h)
        h = self.conv8(h)
        h = self.b8(h)
        h = self.af(h)
        h = self.conv9(h)
        h = self.b9(h)
        h = self.af(h)

        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        h = self.af(h)
        h = self.fc2(h)
        h = self.af(h)
        h = self.fc3(h)
        return h
    
def select_loss(loss_name):
    losses = {
        "sigmoid": lambda x: torch.sigmoid(-x)}
    return losses[loss_name]


class uPUloss(torch.nn.Module):
    """Loss function for PU learning."""

    def __init__(self, prior, loss, gamma=1, beta=0):
        super(uPUloss, self).__init__()

        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss_fn = loss
        self.positive = 1
        self.unlabeled = -1

    def forward(self, x, t, return_risk_separately=False):
        t = t[:, None]
        # positive: if positive,1 else 0
        # unlabeled: if unlabeled,1 else 0
        positive, unlabeled = (t == self.positive).float(
        ), (t == self.unlabeled).float()
        n_positive, n_unlabeled = max(1., positive.sum().item()), max(
            1., unlabeled.sum().item())
        y_positive = self.loss_fn(x)
        y_unlabeled = self.loss_fn(-x)
# ?        breakpoint()
        positive_risk = torch.sum(
            self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum(
            (unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)

        objective = positive_risk + negative_risk
        if return_risk_separately:
            return positive_risk, negative_risk
        else:
            return objective

class uPU(LPU.models.lpu_model_base.LPUModelBase):
    def __init__(self, config, dim=None, **kwargs):
        super(uPU, self).__init__(**kwargs)
        self.config = config
        self.dim = dim
        self.select_model()

    def select_model(self):
        models = {
            "3mlp": ThreeLayerPerceptron,
            "mlp": MultiLayerPerceptron, 
            "cnn": CNN
        }
        self.model = models[self.config['model']](self.dim).to(LPU.constants.DTYPE)
    
    def set_C(self, holdout_dataloader):
        X_all = []
        y_all = []
        l_all = []
        for X, l, y, _ in holdout_dataloader:
            X_all.append(X)
            y_all.append(y)
            l_all.append(l)
        X_all = torch.cat(X_all)
        y_all = torch.cat(y_all)
        l_all = torch.cat(l_all)
        with torch.no_grad():        
            self.C.fill_(l_all[y_all == 1.].mean())
            is_positive = (y_all==1).to(LPU.constants.DTYPE)
            self.prior.fill_(is_positive.mean())
        LOG.warning(f"\pi={self.prior} (the Y class prior) is passed using labeled data, since"
                    "the method needds it")


    def train_one_epoch(self, dataloader, loss_fn=None, optimizer=None, device=None):
        self.model.train()
        scores_dict = {}
        overal_loss = 0
        l_batch_concat = []
        y_batch_concat = []
        y_batch_concat_prob = []
        l_batch_concat_prob = []
        l_batch_concat_est = []
        y_batch_concat_est = []

        binary_kind = set(np.unique(dataloader.dataset.y))
        for batch_num, (X_batch, l_batch, y_batch, _) in enumerate(dataloader):
            X_batch = X_batch.to(device)
            l_batch = l_batch.to(device)
            y_batch = y_batch.to(device)
            
            # if self.config['data_generating_process'] == 'CC':
            #     X_batch = torch.concat([X_batch, X_batch[l_batch==1]], dim=0)
            #     y_batch = torch.concat([y_batch, y_batch[l_batch==1]], dim=0)
            #     l_batch = torch.concat([torch.zeros_like(l_batch), torch.ones((int(l_batch.sum().detach().cpu().numpy().squeeze())))], dim=0)

            f_x = self.model(X_batch)   

            ########################################################################
            # calculating score-related values
            ########################################################################

            detached_f_x = f_x.clone().detach().cpu()
            y_batch_prob = self.predict_prob_y_given_X(f_x=detached_f_x)
            l_batch_prob = self.predict_proba(f_x=detached_f_x)
            y_batch_est = self.predict_y_given_X(f_x=detached_f_x)
            l_batch_est = self.predict(f_x=detached_f_x)

            if isinstance(y_batch_prob, np.ndarray):
                y_batch_prob = torch.tensor(y_batch_prob, dtype=LPU.constants.DTYPE)
                l_batch_prob = torch.tensor(l_batch_prob, dtype=LPU.constants.DTYPE)
                y_batch_est = torch.tensor(y_batch_est, dtype=LPU.constants.DTYPE)
                l_batch_est = torch.tensor(l_batch_est, dtype=LPU.constants.DTYPE)



            y_batch_concat.append(y_batch)
            y_batch_concat_prob.append(y_batch_prob)
            y_batch_concat_est.append(y_batch_est)

            l_batch_concat_prob.append(l_batch_prob)
            l_batch_concat.append(l_batch)
            l_batch_concat_est.append(l_batch_est)

            ########################################################################
            # calculating loss & backpropagation
            ########################################################################

            optimizer.zero_grad()            
            # calculating the loss only for the current batch
            loss = loss_fn(f_x, l_batch)
            loss.backward()
            optimizer.step()
            overal_loss += loss.item()
            
        y_batch_concat_prob = np.concatenate(y_batch_concat_prob)
        l_batch_concat_prob = np.concatenate(l_batch_concat_prob)
        y_batch_concat_est = np.concatenate(y_batch_concat_est)
        l_batch_concat_est = np.concatenate(l_batch_concat_est)
        y_batch_concat = np.concatenate(y_batch_concat)
        l_batch_concat = np.concatenate(l_batch_concat)
        if binary_kind == {-1, 1}:
            l_batch_concat = (l_batch_concat + 1) // 2
            y_batch_concat = (y_batch_concat + 1) // 2
        scores_dict = self._calculate_validation_metrics(
            y_batch_concat_prob, y_batch_concat, l_batch_concat_prob, l_batch_concat, l_ests=l_batch_concat_est, y_ests=y_batch_concat_est
        )

        # average loss
        scores_dict['overall_loss'] = overal_loss / (batch_num + 1)
        return scores_dict



    def predict_prob_y_given_X(self, X=None, f_x=None):
        if f_x is None:
            f_x = self.model(X)
        return torch.sigmoid(f_x)



    def predict_y_given_X(self, X=None, f_x=None, threshold=0.5):
        """
        Predicts the class label given. 

        Args:
            X (np.ndarray): input data
            f_x (np.ndarray): raw output of the model (which is the unnormalized y=1|X)

        Returns:
            np.ndarray: predicted class labels
        """
        self.model.eval()
        LOG.debug("Implementation is not based on probabilities, different from usual"
                  "deal with models inheritting from LPUModelBase")
        if X is not None and f_x is not None:
            raise ValueError("Only one of X and scores should be provided")
        if X is not None:
            f_x = self.model(X)
        return self.predict_prob_y_given_X(f_x=f_x) > threshold


    
    def predict_prob_l_given_y_X(self, X):
        self.model.eval()
        return self.C
    
    def predict_proba(self, X=None, f_x=None):
        self.model.eval()
        if X is not None and f_x is not None:
            raise ValueError("Only one of X and scores should be provided")
        if X is not None:
            f_x = self.model(X)
        return self.predict_prob_y_given_X(f_x=f_x) * self.C
    
    
    def predict(self, X=None, f_x=None, threshold=0.5):
        self.model.eval()
        if X is not None and f_x is not None:
            raise ValueError("Only one of X and scores should be provided")
        if X is not None:
            f_x = self.model(X).detach().cpu().numpy()
        return self.predict_proba(f_x=f_x) > threshold


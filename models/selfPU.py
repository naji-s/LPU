import logging
import os
import sys


import sklearn.metrics
import torch.utils.data
import unittest.mock

import lpu.external_libs.Self_PU.functions
import lpu.external_libs.Self_PU.mean_teacher
import lpu.external_libs.Self_PU.models
import lpu.external_libs.Self_PU.utils



import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import lpu.constants
import lpu.external_libs
import lpu.models.geometric.elkanGGPC
import lpu.models.lpu_model_base
import lpu.external_libs.Self_PU.datasets
import lpu.external_libs.Self_PU.train
import lpu.external_libs.Self_PU.mean_teacher.losses
import lpu.external_libs.Self_PU.mean_teacher.ramps

torch.set_default_dtype(lpu.constants.DTYPE)

LOG = logging.getLogger(__name__)

EPSILON = 1e-16


def make_dataset(dataset, n_labeled, n_unlabeled, mode="train", pn=False, seed = None):
        
    def make_PU_dataset_from_binary_dataset(x, y, l):
        # labels = np.unique(y)
        X, Y, L = np.asarray(x, dtype=lpu.constants.NUMPY_DTYPE), np.asarray(y, dtype=int), np.asarray(l, dtype=int)
        perm = np.random.permutation(len(X))
        X, Y, L = X[perm], Y[perm], L[perm]

        prior = l.mean() / l[y==1].mean() 
        ids = np.arange(len(X))
        return X, L, Y, ids, prior


    def make_PN_dataset_from_binary_dataset(x, y):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        if seed is not None:
            np.random.seed(seed)
        perm = np.random.permutation(len(X))
        X, Y = X[perm], Y[perm]
        n_p = (Y == positive).sum()
        n_n = (Y == negative).sum()
        Xp = X[Y == positive][:n_p]
        Xn = X[Y == negative][:n_n]
        X = np.asarray(np.concatenate((Xp, Xn)), dtype=np.float32)
        Y = np.asarray(np.concatenate((np.ones(n_p), -np.ones(n_n))), dtype=np.int32)
        ids = np.array([i for i in range(len(X))])
        return X, Y, Y, ids

        

    (_trainX, _trainY, _trainL), (_testX, _testY, _testL) = dataset
    prior = None
    if (mode == 'train'):
        if not pn:
            X, Y, T, ids, prior = make_PU_dataset_from_binary_dataset(_trainX, _trainY, _trainL)
        else:
            raise NotImplementedError("Not implemented yet.")
    else:
        X, Y, T, ids  = make_PN_dataset_from_binary_dataset(_testX, _testY)
    return X, Y, T, ids, prior

class selfPUModifiedDataset(lpu.external_libs.Self_PU.datasets.MNIST_Dataset):
    def __init__(self, trainX, trainY, trainL, testX, testY, testL, type="noisy", split="train", mode=None, ids=None, pn=False, increasing=False, replacement=True, top = 0.5, flex = 0, pickout=True, seed = None):
        # super().__init__(trainX, trainY, testX, testY, type=type, split=split, mode=mode, ids=ids, pn=pn, increasing=increasing, replacement=replacement, top=top, flex=flex, pickout=pickout, seed=seed)
        n_labeled = trainL.sum().to(int)
        n_unlabeled = len(trainL) - n_labeled

        # super().__init__(n_labeled, n_unlabeled, trainX, trainY, testX, testY, type=type, split=split, mode=mode, ids=ids, pn=pn, increasing=increasing, replacement=replacement, top = top, flex = flex, pickout=pickout, seed = seed)
        # if split == "train":
        #     self.X, self.Y, self.T, self.oids, self.prior = lpu.external_libs.Self_PU.datasets.make_dataset(((trainX, trainY), (testX, testY)), n_labeled, n_unlabeled, mode=split, pn=pn, seed = seed)
        # elif split == "test":
        #     self.X, self.Y, self.T, self.oids, self.prior = lpu.external_libs.Self_PU.datasets.make_dataset(((trainX, trainY), (testX, testY)), n_labeled, n_unlabeled, mode=split, pn=pn, seed = seed)
        # else:
        #     raise ValueError("split should be either 'train' or 'test'")
        trainY = 2 * trainY - 1
        trainL = 2 * trainL - 1
        testY = 2 * testY - 1
        testL = 2 * testL - 1

        if split == "train":
            self.X, self.Y, self.T, self.oids, self.prior = make_dataset(((trainX, trainY, trainL), (testX, testY, testL)), n_labeled, n_unlabeled, mode=split, pn=pn, seed = seed)
        elif split == "test":
            self.X, self.Y, self.T, self.oids, self.prior = make_dataset(((trainX, trainY, trainL), (testX, testY, testL)), n_labeled, n_unlabeled, mode=split, pn=pn, seed = seed)
        else:
            raise ValueError("split should be either 'train' or 'test'")
        assert np.all(self.oids == np.linspace(0, len(self.X) - 1, len(self.X)))
        self.clean_ids = []
        # self.Y_origin = self.Y
        self.P = self.Y.copy()
        self.type = type
        if (ids is None):
            self.ids = self.oids
        else:
            self.ids = np.array(ids)

        self.split = split
        self.mode = mode
        self.pos_ids = self.oids[self.Y == 1]

        self.pid = self.pos_ids
        if len(self.ids) != 0:
            self.uid = np.intersect1d(self.ids[self.Y[self.ids] == -1], self.ids)
        else:
            self.uid = []
        print(len(self.uid))
        print(len(self.pid))
        self.sample_ratio = len(self.uid) // len(self.pid)  + 1
        print("SAMPLE RATIO:", self.sample_ratio)
        print("origin:", len(self.pos_ids), len(self.ids), type)
        self.increasing = increasing
        self.replacement = replacement
        self.top = top
        self.flex = flex
        self.pickout = pickout

        self.pick_accuracy = []
        self.result = -np.ones(len(self))
        self.result = -np.ones(len(trainX) + len(testX))

class selfPU(lpu.models.lpu_model_base.LPUModelBase):
    """
    """
    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 32)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.model = None
        self.epochs = config.get('epochs', 200)
        self.ema_decay = None
        self.soft_label = config.get('soft_label', False)
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 4)
        self.model_dir = config.get('model_dir', 'lpu/scripts/selfPU')

    def set_C(self, holdout_dataloader):
        try:
            assert (holdout_dataloader.batch_size == len(holdout_dataloader.dataset)), "There should be only one batch in the dataloader."
        except AssertionError as e:
            LOG.error(f"There should be only one batch in the dataloader, but {holdout_dataloader.batch_size} is smaller than {len(holdout_dataloader.dataset)}.")
            raise e
        X, l, y = next(iter(holdout_dataloader))
        self.C = l[y == 1].mean()


    def predict_proba(self, X):
        self.model.eval()
        return torch.sigmoid(self.model(torch.as_tensor(X, dtype=lpu.constants.DTYPE))).detach().numpy().flatten()

    def predict_prob_y_given_x(self, X):
        return self.predict_proba(X) / self.C
    
    def predict_prob_l_given_y_x(self, X):
        return self.C 

    def loss_fn(self, X_batch, l_batch):
        return torch.tensor(0.0)
        # return lpu.external_libs.DEDPUL.NN_functions.d_loss_standard(X_batch[l_batch == 0], X_batch[l_batch == 1], self.discriminator, self.loss_type)


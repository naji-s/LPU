import sys

import lpu.external_libs.PU_learning.utils
sys.path.append('lpu/external_libs/PU_learning')

import torch.cuda
import torch.nn
import numpy as np

import lpu.constants
import lpu.utils.utils_general
import lpu.external_libs.PU_learning
import lpu.external_libs.PU_learning.algorithm
import lpu.external_libs.PU_learning.baselines
import lpu.external_libs.PU_learning.data_helper
import lpu.external_libs.PU_learning.estimator
import lpu.external_libs.PU_learning.train_PU
import lpu.external_libs.PU_learning.data_helper
import lpu.external_libs.PU_learning.helper
import lpu.datasets.LPUDataset
import lpu.datasets.dataset_utils
import lpu.external_libs.PU_learning.model_helper
import lpu.models.lpu_model_base

import types
import unittest.mock

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, dim):
        super(MultiLayerPerceptron, self).__init__()

        self.input_dim = dim

        self.l1 = torch.nn.Linear(dim, 300, bias=False)
        self.b1 = torch.nn.BatchNorm1d(300)
        self.l2 = torch.nn.Linear(300, 300, bias=False)
        self.b2 = torch.nn.BatchNorm1d(300)
        self.l3 = torch.nn.Linear(300, 300, bias=False)
        self.b3 = torch.nn.BatchNorm1d(300)
        self.l4 = torch.nn.Linear(300, 300, bias=False)
        self.b4 = torch.nn.BatchNorm1d(300)
        self.l5 = torch.nn.Linear(300, 2)
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
    
class MPE(lpu.models.lpu_model_base.LPUModelBase):
    def __init__(self, config):
        super(MPE, self).__init__()
        self.config = config
        self.alpha_estimate = 0.0
        self.device = config['device']
        self.true_alpha = config['alpha']
        self.beta = config['beta']

    def initialize_model(self, dim):
        self.net = MultiLayerPerceptron(dim=dim)
        if self.config['device'].startswith('cuda'):
            self.net = torch.nn.DataParallel(self.net)
            torch.cudnn.benchmark = True
        
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
        pos_probs = lpu.external_libs.PU_learning.algorithm.p_probs(self.net, self.device, p_holdoutloader)
        unlabeled_probs, unlabeled_targets = lpu.external_libs.PU_learning.algorithm.u_probs(self.net, self.device, u_holdoutloader)

        mpe_estimate, _, _ = lpu.external_libs.PU_learning.estimator.BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
        return mpe_estimate
    
    def estimate_alpha(
        self, 
        p_holdoutloader, 
        u_holdoutloader):
        pos_probs = lpu.external_libs.PU_learning.estimator.p_probs(self.net, self.device, p_holdoutloader)
        unlabeled_probs, unlabeled_targets = lpu.external_libs.PU_learning.estimator.u_probs(self.net, self.device, u_holdoutloader)
        """

        """

        mpe_estimate, _, _ = lpu.external_libs.PU_learning.estimator.BBE_estimator(pos_probs, unlabeled_probs, unlabeled_targets)
        return mpe_estimate


    def warm_up_one_epoch(self, epoch, p_trainloader, u_trainloader, optimizer, criterion, u_validloader):
        train_acc = lpu.external_libs.PU_learning.algorithm.train(epoch, self.net, p_trainloader, u_trainloader, \
        optimizer=optimizer, criterion=criterion, device=self.device, show_bar=self.config['show_bar'])

        valid_acc = lpu.external_libs.PU_learning.algorithm.validate(epoch, self.net, u_validloader, \
                criterion=criterion, device=self.device, threshold=0.5*self.beta/(self.beta + (1-self.beta)*self.true_alpha),show_bar=self.config['show_bar'])
        
        return train_acc, valid_acc 
    

    def train_on_epoch(self, epoch, p_trainloader, u_trainloader, optimizer, criterion, alpha_used, train_unlabeled_size):
        keep_samples, _ = lpu.external_libs.PU_learning.train_PU.rank_inputs(epoch, self.net, u_trainloader, self.device,\
            alpha_used, u_size=train_unlabeled_size)
        
        train_acc = lpu.external_libs.PU_learning.train_PU.train_PU_discard(epoch, self.net,  p_trainloader, u_trainloader,\
            optimizer, criterion, self.device, keep_sample=keep_samples,show_bar=self.config['show_bar'])
        return train_acc
    def predict_proba(self, X):
        return self.net(X)

LEARNING_RATE = 0.01
INDUCING_POINTS_SIZE = 32
BATCH_SIZE = 32
TRAIN_VAL_RATIO = .1
ELKAN_HOLD_OUT_SIZE = 0.1
TRAIN_TEST_RATIO = .5


def create_mpe_datasets(lpu_dataset, double_unlabeled=False):
    _, _, _, _, train_indices, test_indices, val_indices, holdout_indices = lpu.datasets.dataset_utils.create_stratified_splits(lpu_dataset, train_val_ratio=TRAIN_VAL_RATIO, batch_size=BATCH_SIZE, hold_out_size=ELKAN_HOLD_OUT_SIZE, train_test_ratio=TRAIN_TEST_RATIO, return_indices=True)
    unlabeled_train_indices = train_indices[lpu_dataset.l[train_indices]==0]
    unlabeled_val_indices = val_indices[lpu_dataset.l[val_indices]==0]
    unlabeled_holdout_indices = holdout_indices[lpu_dataset.l[holdout_indices]==0]
    unlabeled_test_indices = test_indices[lpu_dataset.l[test_indices]==0]

    p_train_indices = train_indices[lpu_dataset.l[train_indices]==1]
    p_val_indices = val_indices[lpu_dataset.l[val_indices]==1]
    p_holdout_indices = holdout_indices[lpu_dataset.l[holdout_indices]==1]
    p_test_indices = test_indices[lpu_dataset.l[test_indices]==1]

    X_tensor = torch.tensor(lpu_dataset.X, dtype=lpu.constants.DTYPE)

    PDataset_train = lpu.external_libs.PU_learning.helper.PosData(
        transform=lpu_dataset.transform, target_transform=None, data=X_tensor[p_train_indices], index=np.arange(len(p_train_indices)))
    UDataset_train = lpu.external_libs.PU_learning.helper.UnlabelData(
        transform=lpu_dataset.transform, target_transform=None, 
        pos_data=X_tensor[train_indices][lpu_dataset.y[train_indices]==1], 
        neg_data=X_tensor[train_indices][lpu_dataset.y[train_indices]==0],
        index=np.arange(len(train_indices)))
    
    PDataset_test = lpu.external_libs.PU_learning.helper.PosData(
        transform=lpu_dataset.transform, target_transform=None, data=X_tensor[p_test_indices], index=np.arange(len(p_test_indices)))
    UDataset_test = lpu.external_libs.PU_learning.helper.UnlabelData(
        transform=lpu_dataset.transform, target_transform=None, 
        pos_data=X_tensor[unlabeled_test_indices][lpu_dataset.y[unlabeled_test_indices]==1], 
        neg_data=X_tensor[unlabeled_test_indices][lpu_dataset.y[unlabeled_test_indices]==0],
        index=np.arange(len(unlabeled_test_indices)))
    
    PDataset_val = lpu.external_libs.PU_learning.helper.PosData(
        transform=lpu_dataset.transform, target_transform=None, data=X_tensor[p_val_indices], index=np.arange(len(p_val_indices)))
    UDataset_val = lpu.external_libs.PU_learning.helper.UnlabelData(
        transform=lpu_dataset.transform, target_transform=None, 
        pos_data=X_tensor[val_indices][lpu_dataset.y[val_indices]==1], 
        neg_data=X_tensor[val_indices][lpu_dataset.y[val_indices]==0],
        index=np.arange(len(val_indices)))
    
    PDataset_holdout = lpu.external_libs.PU_learning.helper.PosData(
        transform=lpu_dataset.transform, target_transform=None, data=X_tensor[p_holdout_indices], index=np.arange(len(p_holdout_indices)))
    UDataset_holdout = lpu.external_libs.PU_learning.helper.UnlabelData(
        transform=lpu_dataset.transform, target_transform=None, 
        pos_data=X_tensor[holdout_indices][lpu_dataset.y[holdout_indices]==1], 
        neg_data=X_tensor[holdout_indices][lpu_dataset.y[holdout_indices]==0],
        index=np.arange(len(holdout_indices)))

    return PDataset_train, UDataset_train, PDataset_test, UDataset_test, PDataset_val, UDataset_val, PDataset_holdout, UDataset_holdout

def main():
    yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/mpe_config.yaml'
    config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    alpha_estimate = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal')
#    passing X to initialize_inducing_points to extract the initial values of inducing points

    (PDataset_train, UDataset_train, 
        PDataset_test, UDataset_test, 
            PDataset_val, UDataset_val, 
                PDataset_holdout, UDataset_holdout) = create_mpe_datasets(lpu_dataset)
    p_trainloader = torch.utils.data.DataLoader(PDataset_train, batch_size=config['batch_size'], shuffle=True)
    u_trainloader = torch.utils.data.DataLoader(UDataset_train, batch_size=config['batch_size'], shuffle=True)
    p_validloader = torch.utils.data.DataLoader(PDataset_val, batch_size=config['batch_size'], shuffle=True)
    u_validloader = torch.utils.data.DataLoader(UDataset_val, batch_size=config['batch_size'], shuffle=True)
    p_holdoutloader = torch.utils.data.DataLoader(PDataset_holdout, batch_size=config['batch_size'], shuffle=True)
    u_holdoutloader = torch.utils.data.DataLoader(UDataset_holdout, batch_size=config['batch_size'], shuffle=True)
    p_testloader = torch.utils.data.DataLoader(PDataset_test, batch_size=config['batch_size'], shuffle=True)
    u_testloader = torch.utils.data.DataLoader(UDataset_test, batch_size=config['batch_size'], shuffle=True)

    mpe_model = MPE(config)
    mpe_model.initialize_model(lpu_dataset.X.shape[1])


    train_unlabeled_size = len(u_trainloader.dataset)

    if config['optimizer']=="SGD":
        optimizer = torch.optim.SGD(mpe_model.net.parameters(), lr=config['lr'], momentum=config['momentum'], weight_decay=config['wd'])
    elif config['optimizer']=="Adam":
        optimizer = torch.optim.Adam(mpe_model.net.parameters(), lr=config['lr'],weight_decay=config['wd'])
    elif config['optimizer']=="AdamW": 
        optimizer = torch.optim.AdamW(mpe_model.net.parameters(), lr=config['lr'])

    ## Train in the begining for warm start
    if config['warm_start']: 
        # outfile.write("Warm_start: \n")
        for epoch in range(config['warm_start_epochs']): 
            train_acc, valid_acc = mpe_model.warm_up_one_epoch(epoch=epoch, p_trainloader=p_trainloader, u_trainloader=u_trainloader, optimizer=optimizer, criterion=criterion, u_validloader=u_validloader)
            if config['estimate_alpha']: 
                alpha_estimate = mpe_model.estimate_alpha(p_holdoutloader=p_holdoutloader, u_holdoutloader=u_holdoutloader)
            print ("Epoch: ", epoch, "Alpha: ", alpha_estimate)

    alpha_used = alpha_estimate

    for epoch in range(config['epochs']):
        if config['use_alpha']: 
            alpha_used =  alpha_estimate
        else:
            alpha_used = config['alpha']
        train_acc = mpe_model.train_on_epoch(epoch=epoch, p_trainloader=p_trainloader, 
                       u_trainloader=u_trainloader, optimizer=optimizer, criterion=criterion, 
                       alpha_used=alpha_used, train_unlabeled_size=train_unlabeled_size)

        valid_acc = lpu.external_libs.PU_learning.algorithm.validate(epoch, mpe_model.net, u_validloader, 
                                                                     criterion=criterion, device=mpe_model.device, 
                                                                     threshold=0.5*config['beta']/(config['beta'] + (1-config['beta'])*config['alpha']), 
                                                                     show_bar=config['show_bar'])
        
        if config['estimate_alpha']: 
            mpe_model.estimate_alpha(p_holdoutloader, u_holdoutloader)
        print ("Epoch: ", epoch, "Train Acc: ", train_acc, "Valid Acc: ", valid_acc)

if __name__ == "__main__":
    # yaml_file_path = '/Users/naji/phd_codebase/lpu/configs/mpe_config.yaml'
    # config = lpu.utils.utils_general.load_and_process_config(yaml_file_path)
    main()
    # lpu_dataset = lpu.datasets.LPUDataset.LPUDataset(dataset_name='animal_no_animal', normalize=False, invert_l=False)
    # train_loader, test_loader, val_loader, holdout_loader = lpu.datasets.dataset_utils.create_stratified_splits(lpu_dataset, train_val_ratio=TRAIN_VAL_RATIO, batch_size=len(lpu_dataset), hold_out_size=HOLD_OUT_SIZE, train_test_ratio=TRAIN_TEST_RATIO)
    # passing X to initialize_inducing_points to extract the initial values of inducing points
    # nnPU_model = lpu.models.nnPU.nnPU(self.config)
    # args = types.SimpleNamespace(**config)
    # sys.path.append('/Users/naji/phd_codebase/lpu/external_libs/nnPUSB')
    # main(args, self.config)
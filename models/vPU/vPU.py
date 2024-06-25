
import logging
import math

import numpy as np
import torch.nn
import torchvision.models


import LPU.external_libs
import LPU.external_libs.distPU.customized
import LPU.external_libs.distPU.customized.mixup
import LPU.external_libs.distPU.losses
import LPU.external_libs.distPU.losses.distributionLoss
import LPU.external_libs.distPU.losses.entropyMinimization
import LPU.external_libs.distPU.losses.factory
import LPU.external_libs.distPU.models.modelForCIFAR10
import LPU.external_libs.distPU.models.modelForFMNIST
import LPU.external_libs.vpu
import LPU.external_libs.vpu.model
import LPU.external_libs.vpu.model.model_cifar
import LPU.external_libs.vpu.utils
import LPU.external_libs.vpu.utils.func
import LPU.models.distPU
import LPU.models.lpu_model_base
import LPU.constants
import LPU.external_libs.distPU.models.factory
import LPU.external_libs.distPU.models.modelForCIFAR10
import LPU.external_libs.distPU.models.modelForFMNIST
import LPU.utils.auxiliary_models

DEFAULT_CONFIG = {
    "device": "cpu",
    "dataset_name": "animal_no_animal",  # could also be fashionMNIST
    "dataset_kind": "LPU",
    "val_iterations": 30,
    "num_labeled": 3000,
    "learning_rate": 0.0001,
    "epochs": 50,
    "mix_alpha": 0.3,
    "data_generating_process": "SB",  # either of CC (case-control) or SB (selection-bias)
    "lam": 0.3,
    "ratios": {
        # *** NOTE ***
        # TRAIN_RATIO == 1. - HOLDOUT_RATIO - TEST_RATIO - VAL_RATIO
        # i.e. test_ratio + val_ratio + holdout_ratio + train_ratio == 1
        'test': 0.4,
        'val': 0.2,
        'holdout': .0,
        'train': .4, 
    },
    "batch_size": {
        "train": 64,
        "test": 64,
        "val": 16,
        "holdout": 64
    }
}


LOG = logging.getLogger(__name__)


def cal_val_var(config, model_phi, val_p_loader, val_u_loader):
    """
    Calculate variational loss on the validation set, which consists of only positive and unlabeled data.

    :param model_phi: current \Phi model.
    :param val_p_loader: loader for positive data in the validation set.
    :param val_u_loader: loader for the whole validation set (including positive and unlabeled data).
    """

    # set the model to evaluation mode
    model_phi.eval()

    # feed the validation set to the model and calculate variational loss
    with torch.no_grad():
        for idx, unlabeled_tuple in enumerate(val_u_loader):
            if config['dataset_kind'] in ['LPU', 'MPE']:
                _, data_u, _, _ = unlabeled_tuple
            else:
                data_u, _ = unlabeled_tuple
            if torch.cuda.is_available():
                data_u = data_u.cuda()
            data_u = data_u.to(LPU.constants.DTYPE)
            output_phi_u_curr = model_phi(data_u)
            if idx == 0:
                output_phi_u = output_phi_u_curr
            else:
                output_phi_u = torch.cat((output_phi_u, output_phi_u_curr))
        for idx, p_tuple in enumerate(val_p_loader):
            if config['dataset_kind'] in ['LPU', 'MPE']:
                _, data_p, _ = p_tuple
            else:
                data_p, _ = p_tuple
            if torch.cuda.is_available():
                data_p = data_p.cuda()
            data_p = data_p.to(LPU.constants.DTYPE)
            output_phi_p_curr = model_phi(data_p)
            if idx == 0:
                output_phi_p = output_phi_p_curr
            else:
                output_phi_p = torch.cat((output_phi_p, output_phi_p_curr))
        log_phi_p = output_phi_p[:, 1]
        log_phi_u = output_phi_u[:, 1]
        var_loss = torch.logsumexp(log_phi_u, dim=0) - math.log(len(log_phi_u)) - torch.mean(log_phi_p)
        return var_loss.item()
    
def get_model_by_dataset(dataset_name):
    """ Returns the model class based on the dataset name. """    
    dataset_to_model = {
        'cifar10': LPU.external_libs.vpu.model.model_cifar.NetworkPhi,
        'fashionMNIST': LPU.external_libs.vpu.model.model_fashionmnist.NetworkPhi,
        'stl10': LPU.external_libs.vpu.model.model_stl.NetworkPhi,
        'pageblocks': LPU.external_libs.vpu.model.model_vec.NetworkPhi,
        'grid': LPU.external_libs.vpu.model.model_vec.NetworkPhi,
        'avila': LPU.external_libs.vpu.model.model_vec.NetworkPhi,
        'animal_no_animal': LPU.utils.auxiliary_models.MultiLayerPerceptron
    }
    return dataset_to_model.get(dataset_name)

class vPU(LPU.models.lpu_model_base.LPUModelBase):
    def __init__(self, config, **kwargs):
        super(vPU, self).__init__()
        self.config = config
        self.device = config.get('device')
        if self.config['dataset_kind'] not in ['LPU', 'MPE']:
            kwargs.pop('input_dim')
        self.model = get_model_by_dataset(config['dataset_name'])(**kwargs).to(self.device).to(LPU.constants.DTYPE)

    def train_one_epoch(self, config, opt_phi, p_loader, u_loader):
        """
        One epoch of the training of VPU.

        :param config: arguments.
        :param model_phi: current model \Phi.
        :param opt_phi: optimizer of \Phi.
        :param p_loader: loader for the labeled positive training data.
        :param u_loader: loader for training data (including positive and unlabeled)
        """

        # set the model to train mode
        self.model.train()
        total_phi_loss = 0
        total_var_loss = 0
        total_reg_loss = 0
        loader_length = 1

        # Initialize variables to store concatenated data and targets
        train_data_all = None
        train_target_all = None
        f_x_all = None

        for batch_idx, (unlabeled_tuple, positive_tuple) in enumerate(zip(u_loader, p_loader)):
            if self.config['dataset_kind'] in ['LPU', 'MPE']:
                _, data_u, _, u_true_target = unlabeled_tuple
                _, data_p, _ = positive_tuple
            else:
                data_u, u_true_target = unlabeled_tuple
                data_p, _ = positive_tuple

            data_all = torch.concat((data_p, data_u))
            data_all = data_all.to(self.config['device']).to(LPU.constants.DTYPE)
            output_phi_all = self.model(data_all)
            log_phi_all = output_phi_all[:, 1]
            idu_p = slice(0, len(data_p))
            idu_u = slice(len(data_p), len(data_all))

            if self.config['data_generating_process'] == 'SB':
                y = torch.cat((torch.zeros(len(data_p)), u_true_target))
                l = torch.cat((torch.zeros(len(data_p)), torch.ones(len(data_u))))
                f_x = log_phi_all
            else:
                y = u_true_target
                l = torch.zeros(len(data_u))
                f_x = log_phi_all[idu_u]

            log_phi_u = log_phi_all[idu_u]
            log_phi_p = log_phi_all[idu_p]
            output_phi_u = output_phi_all[idu_u]
            var_loss = torch.logsumexp(log_phi_u, dim=0) - math.log(len(log_phi_u)) - 1 * torch.mean(log_phi_p)

            # perform Mixup and calculate the regularization
            target_u = output_phi_u[:, 1].exp()
            target_p = torch.ones(len(data_p))
            target_p = target_p.cuda() if torch.cuda.is_available() else target_p
            rand_perm = torch.randperm(data_p.size(0))
            data_p_perm, target_p_perm = data_p[rand_perm], target_p[rand_perm]
            m = torch.distributions.beta.Beta(config['mix_alpha'], config['mix_alpha'])
            lam = m.sample()
            data = lam * data_u + (1 - lam) * data_p_perm
            target = lam * target_u + (1 - lam) * target_p_perm
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            data = data.to(LPU.constants.DTYPE)
            out_log_phi_all = self.model(data)
            reg_mix_log = ((torch.log(target) - out_log_phi_all[:, 1]) ** 2).mean()
            total_reg_loss += reg_mix_log.item()
            # calculate gradients and update the network
            phi_loss = var_loss + config['lam'] * reg_mix_log
            total_phi_loss += phi_loss.item()
            total_var_loss += var_loss.item()
            opt_phi.zero_grad()
            phi_loss.backward()
            opt_phi.step()
            loader_length += 1

            # Concatenate data and targets
            if train_data_all is None:
                train_data_all = data_all
                y_all = y
                l_all = l
                f_x_all = f_x
            else:
                train_data_all = torch.cat((train_data_all, data_all))
                y_all = torch.cat((y_all, y))
                l_all = torch.cat((l_all, l))
                f_x_all = torch.cat((f_x_all, f_x))
        # inverting labels since MPE loaders have inverse labeling, i.e. 0 for positive and 1 for negative
        y_all = 1 - y_all
        # Calculate scores on concatenated data
        train_data_all = train_data_all.to(LPU.constants.DTYPE)
        train_y_probs = torch.exp(f_x_all).detach().cpu().numpy()
        train_l_probs = np.random.uniform(0, 1, len(train_y_probs))
        train_l_ests = (train_l_probs > 0.5).astype(int)
        train_y_ests = (train_y_probs > 0.5).astype(int)
        train_results = super()._calculate_validation_metrics(
                y_probs=train_y_probs, y_vals=y_all, l_probs=train_l_probs, l_vals=l_all, l_ests=train_l_ests, y_ests=train_y_ests)
        train_results.update({'overall_loss': total_var_loss / loader_length})
        train_results.update({'phi_loss': total_phi_loss / loader_length})
        train_results.update({'reg_loss': total_reg_loss / loader_length})
        return train_results

                       
    def validate(self, train_p_loader, train_u_loader, val_p_loader, val_u_loader, epoch, train_phi_loss, var_loss, train_reg_loss, test_loader=None):
        
        self.model.eval()

        # calculate variational loss of the validation set consisting of PU data
        val_var = cal_val_var(self.config, self.model, val_p_loader, val_u_loader)

        
        # max_phi is needed for normalization
        train_log_max_phi = -math.inf
        for idx, (train_p_tuple, train_u_tuple) in enumerate(zip(train_p_loader, train_u_loader)):
            if self.config['dataset_kind'] in ['LPU', 'MPE']:
                _, train_data_u, _, _ = train_u_tuple
            else:
                train_data_p, _ = train_p_tuple
                train_data_u, _ = train_u_tuple
            train_data = train_data_u
            # this line is about the 
            train_data = train_data.to(LPU.constants.DTYPE)
            if torch.cuda.is_available():
                train_data = train_data.cuda()
            train_log_max_phi = max(train_log_max_phi, self.model(train_data)[:, 1].max())
        
        for idx, (val_u_tuple, val_p_tuple) in enumerate(zip(val_u_loader, val_p_loader)):
            if self.config['dataset_kind'] in ['LPU', 'MPE']:
                _, val_data_u, _, val_u_true_target = val_u_tuple
                _, val_data_p, _ = val_p_tuple
            else:
                val_data_u, val_u_true_target = val_u_tuple
            if self.config['data_generating_process'] == 'SB':
                val_data = torch.cat((val_data_p, val_data_u))
                y_val = torch.cat((torch.zeros(len(val_data_p)), val_u_true_target))
            else:
                val_data = val_data_u
                y_val = val_u_true_target
            # inverting labels since MPE loaders have inverse labeling, i.e. 0 for positive and 1 for negative
            y_val = 1 - y_val
            if torch.cuda.is_available():
                val_data = val_data.cuda()
                y_val = y_val.cuda()
            val_data = val_data.to(LPU.constants.DTYPE)
            val_log_phi = self.model(val_data)[:, 1]
            val_log_phi -= train_log_max_phi
            if idx == 0:
                val_log_phi_all = val_log_phi
                y_all_vall = y_val
                val_l_vals_all = np.concatenate([np.ones(len(val_data_p)), np.zeros(len(val_data_u))])
            else:
                val_log_phi_all = torch.cat((val_log_phi_all, val_log_phi))
                y_all_vall = torch.cat((y_all_vall, y_val))
                val_l_vals_all = np.concatenate([val_l_vals_all, np.concatenate([np.ones(len(val_data_p)), np.zeros(len(val_data_u))])])

        val_y_probs = torch.exp(val_log_phi_all).detach().cpu().numpy()
        y_all_vall = y_all_vall.cpu().numpy()
        val_l_vals = np.random.randint(0, 2, len(y_all_vall))
        val_l_probs = np.random.uniform(0, 1, len(y_all_vall))
        val_l_ests = (val_l_probs > val_l_vals.mean()).astype(int)
        val_y_ests = (val_y_probs > 0.5).astype(int)
        validation_results = super()._calculate_validation_metrics(
                y_probs=val_y_probs,y_vals=y_all_vall, l_probs=val_l_probs, l_vals=val_l_vals, l_ests=val_l_ests, y_ests=val_y_ests)
        validation_results.update({'overall_loss': val_var})
        # validation_results.update({'phi_loss': val_phi})
        # validation_results.update({'reg_loss': train_reg_loss})
        validation_results.update({'l_accuracy': np.nan, 
                                    'l_precision': np.nan, 
                                    'l_auc': np.nan, 
                                    'l_recall': np.nan, 
                                    'l_f1': np.nan, 
                                    'l_APS': np.nan,
                                    'l_ll': np.nan})
        return validation_results

    
            
    def predict_prob_l_given_y_u(self, X):
        return self.C
        

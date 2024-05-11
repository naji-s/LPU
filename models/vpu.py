
import logging
import math

import numpy as np
import torch.nn
import torchvision.models


import lpu.external_libs
import lpu.external_libs.distPU.customized
import lpu.external_libs.distPU.customized.mixup
import lpu.external_libs.distPU.losses
import lpu.external_libs.distPU.losses.distributionLoss
import lpu.external_libs.distPU.losses.entropyMinimization
import lpu.external_libs.distPU.losses.factory
import lpu.external_libs.distPU.models.modelForCIFAR10
import lpu.external_libs.distPU.models.modelForFMNIST
import lpu.external_libs.vpu
import lpu.external_libs.vpu.model
import lpu.external_libs.vpu.model.model_cifar
import lpu.external_libs.vpu.utils
import lpu.external_libs.vpu.utils.func
import lpu.models.distPU
import lpu.models.lpu_model_base
import lpu.constants
import lpu.external_libs.distPU.models.factory
import lpu.external_libs.distPU.models.modelForCIFAR10
import lpu.external_libs.distPU.models.modelForFMNIST
import lpu.utils.auxiliary_models

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
            data_u = data_u.to(lpu.constants.DTYPE)
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
            data_p = data_p.to(lpu.constants.DTYPE)
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
        'cifar10': lpu.external_libs.vpu.model.model_cifar.NetworkPhi,
        'fashionMNIST': lpu.external_libs.vpu.model.model_fashionmnist.NetworkPhi,
        'stl10': lpu.external_libs.vpu.model.model_stl.NetworkPhi,
        'pageblocks': lpu.external_libs.vpu.model.model_vec.NetworkPhi,
        'grid': lpu.external_libs.vpu.model.model_vec.NetworkPhi,
        'avila': lpu.external_libs.vpu.model.model_vec.NetworkPhi,
        'animal_no_animal': lpu.utils.auxiliary_models.MultiLayerPerceptron
    }
    return dataset_to_model.get(dataset_name)

class VPU(lpu.models.lpu_model_base.LPUModelBase):
    def __init__(self, config, **kwargs):
        super(VPU, self).__init__()
        self.config = config
        self.device = config.get('device')
        if self.config['dataset_kind'] not in ['LPU', 'MPE']:
            kwargs.pop('input_dim')
        self.model = get_model_by_dataset(config['dataset_name'])(**kwargs).to(self.device).to(lpu.constants.DTYPE)

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
        for batch_idx, (unlabled_tuple, positive_tuple) in enumerate(zip(u_loader, p_loader)):
            if self.config['dataset_kind'] in ['LPU', 'MPE']:
                _, data_u, _, _ = unlabled_tuple
                _, data_p, _ = positive_tuple
            else:
                data_u, _ = unlabled_tuple
                data_p, _ = positive_tuple
            if torch.cuda.is_available():
                data_p, data_u = data_p.cuda(), data_u.cuda()
            # calculate the variational loss
            data_all = torch.cat((data_p, data_u))
            data_all = data_all.to(lpu.constants.DTYPE)

            output_phi_all = self.model(data_all)
            log_phi_all = output_phi_all[:, 1]
            idu_p = slice(0, len(data_p))
            idu_u = slice(len(data_p), len(data_all))
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
            # breakpoint()
            data = lam * data_u + (1 - lam) * data_p_perm
            target = lam * target_u + (1 - lam) * target_p_perm
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            data = data.to(lpu.constants.DTYPE)
            out_log_phi_all = self.model(data)
            reg_mix_log = ((torch.log(target) - out_log_phi_all[:, 1]) ** 2).mean()
            total_reg_loss += reg_mix_log.item()
            # calculate gradients and update the network
            phi_loss = var_loss + config['lam'] * reg_mix_log
            total_phi_loss += phi_loss
            total_var_loss += var_loss
            opt_phi.zero_grad()
            phi_loss.backward()
            opt_phi.step()
            loader_length += 1
            # update the utilities for analysis of the model
            # reg_avg.update(reg_mix_log.item())
            # phi_loss_avg.update(phi_loss.item())
            # var_loss_avg.update(var_loss.item())
            # phi_p, phi_u = log_phi_p.exp(), log_phi_u.exp()
            # phi_p_avg.update(phi_p.mean().item(), len(phi_p))
            # phi_u_avg.update(phi_u.mean().item(), len(phi_u))

        return total_phi_loss / loader_length, total_var_loss / loader_length, total_reg_loss / loader_length
                       
    def validate(self, train_p_loader, train_u_loader, val_p_loader, val_u_loader, epoch, phi_loss, var_loss, reg_loss, test_loader=None):
        
        self.model.eval()

        # calculate variational loss of the validation set consisting of PU data
        val_var = cal_val_var(self.config, self.model, val_p_loader, val_u_loader)

        
        # max_phi is needed for normalization
        log_max_phi = -math.inf
        for idx, (p_tuple, u_tuple) in enumerate(zip(train_p_loader, train_u_loader)):
            if self.config['dataset_kind'] in ['LPU', 'MPE']:
                _, data_u, _, _ = u_tuple
            else:
                data_p, _ = p_tuple
                data_u, _ = u_tuple
            data = data_u
            # this line is about the 
            data = data.to(lpu.constants.DTYPE)
            if torch.cuda.is_available():
                data = data.cuda()
            log_max_phi = max(log_max_phi, self.model(data)[:, 1].max())

        for idx, (u_tuple, p_tuple) in enumerate(zip(val_u_loader, val_p_loader)):
            if self.config['dataset_kind'] in ['LPU', 'MPE']:
                _, data_u, _, u_true_target = u_tuple
                _, data_p, _ = p_tuple
            else:
                data_u, u_true_target = u_tuple
            data = data_u
            # inverting labels since MPE loaders have inverse labeling, i.e. 0 for positive and 1 for negative
            target = 1 - u_true_target
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data = data.to(lpu.constants.DTYPE)
            log_phi = self.model(data)[:, 1]
            log_phi -= log_max_phi
            if idx == 0:
                log_phi_all = log_phi
                target_all = target
                l_vals_all = np.concatenate([np.ones(len(data_p)), np.zeros(len(data_u))])
            else:
                log_phi_all = torch.cat((log_phi_all, log_phi))
                target_all = torch.cat((target_all, target))
                l_vals_all = np.concatenate([l_vals_all, np.concatenate([np.ones(len(data_p)), np.zeros(len(data_u))])])

        if test_loader:
            # feed test set to the model and calculate accuracy and AUC
            with torch.no_grad():
                for idx, (data, target) in enumerate(test_loader):
                    if torch.cuda.is_available():
                        data, target = data.cuda(), target.cuda()
                    data = data.to(lpu.constants.DTYPE)
                    log_phi = self.model(data)[:, 1]
                    log_phi -= log_max_phi
                    if idx == 0:
                        log_phi_all = log_phi
                        target_all = target
                    else:
                        log_phi_all = torch.cat((log_phi_all, log_phi))
                        target_all = torch.cat((target_all, target))
        y_probs = torch.exp(log_phi_all).detach().cpu().numpy()
        y_vals = target_all.cpu().numpy()
        l_vals = np.random.randint(0, 2, len(y_vals))
        l_probs = np.random.uniform(0, 1, len(y_vals))
        l_ests = (l_probs > l_vals.mean()).astype(int)
        y_ests = (y_probs > 0.5).astype(int)
        validation_results = super()._calculate_validation_metrics(
                y_probs, y_vals, l_probs, l_vals, l_ests=l_ests, y_ests=y_ests)
        validation_results.update({'val overall_loss': val_var})
        validation_results.update({'train phi_loss': phi_loss})
        validation_results.update({'train var_loss': var_loss})
        validation_results.update({'train reg_loss': reg_loss})
        validation_results.update({'l_accuracy': np.nan, 
                                    'l_precision': np.nan, 
                                    'l_auc': np.nan, 
                                    'l_recall': np.nan, 
                                    'l_f1': np.nan, 
                                    'l_APS': np.nan,
                                    'l_ll': np.nan})
        return validation_results



    # def predict_prob_y_given_u(self, X):
    #     self.model.eval()
    #     log_max_phi = -math.inf
    #     return np.exp(max(log_max_phi, self.model(X)[:, 1].max()))
    
            
    def predict_prob_l_given_y_u(self, X):
        return self.C
        
    # def set_C(self, l_mean, p_loader, u_loader):
    #     log_max_phi = -np.inf
    #     for idx, (p_tuple, u_tuple) in enumerate(zip(p_loader, u_loader)):
    #         if self.config['dataset_kind'] in ['LPU', 'MPE']:
    #             _, data_p, _ = p_tuple
    #             _, data_u, _, _ = u_tuple
    #         else:
    #             data_p, _ = p_tuple
    #             data_u, _ = u_tuple
    #         if self.config['data_generating_process'] == 'SB':
    #             data = torch.cat((data_p, data_u))
    #         elif self.config['data_generating_process'] == 'CC':
    #             data = data_u
    #         else:
    #             raise ValueError('data_generating_process must be one of "SB" or "CC"')
    #         if torch.cuda.is_available():
    #             data = data.cuda()
    #         log_max_phi = max(log_max_phi, self.model(data)[:, 1].max())
    #     self.alpha = torch.exp(log_max_phi).mean().detach().cpu().numpy()
    #     self.C = l_mean * self.alpha
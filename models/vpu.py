
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
import lpu.models.distPU
import lpu.models.lpu_model_base
import lpu.constants
import lpu.external_libs.distPU.models.factory
import lpu.external_libs.distPU.models.modelForCIFAR10
import lpu.external_libs.distPU.models.modelForFMNIST
import lpu.utils.auxiliary_models

LOG = logging.getLogger(__name__)


def get_model_by_dataset(dataset):
    """ Returns the model class based on the dataset name. """
    dataset_to_model = {
        'cifar10': lpu.external_libs.vpu.model.model_cifar.NetworkPhi,
        'fashionMNIST': lpu.external_libs.vpu.model.model_fashionmnist.NetworkPhi,
        'stl10': lpu.external_libs.vpu.model.model_stl.NetworkPhi,
        'pageblocks': lpu.external_libs.vpu.model.model_vec.NetworkPhi,
        'grid': lpu.external_libs.vpu.model.model_vec.NetworkPhi,
        'avila': lpu.external_libs.vpu.model.model_vec.NetworkPhi
    }
    return dataset_to_model.get(dataset)

class VPU(lpu.models.lpu_model_base.LPUModelBase):
    def __init__(self, config):
        super(VPU, self).__init__()
        self.config = config
        self.device = config.get('device')
        self.model = get_model_by_dataset(config['dataset'])()

    def train_one_epoch(self, config, opt_phi, p_loader, x_loader):
        """
        One epoch of the training of VPU.

        :param config: arguments.
        :param model_phi: current model \Phi.
        :param opt_phi: optimizer of \Phi.
        :param p_loader: loader for the labeled positive training data.
        :param x_loader: loader for training data (including positive and unlabeled)
        """

        # set the model to train mode
        self.model.train()

        for batch_idx, (unlabled_tuple, positive_tuple) in enumerate(zip(x_loader, p_loader)):
            if self.config['dataset_kind'] in ['LPU', 'MPE']:
                _, data_x, _, _ = unlabled_tuple
                _, data_p, _ = positive_tuple
            else:
                data_x, _ = unlabled_tuple
                data_p, _ = positive_tuple
            if torch.cuda.is_available():
                data_p, data_x = data_p.cuda(), data_x.cuda()

            # calculate the variational loss
            data_all = torch.cat((data_p, data_x))
            output_phi_all = self.model(data_all)
            log_phi_all = output_phi_all[:, 1]
            idx_p = slice(0, len(data_p))
            idx_x = slice(len(data_p), len(data_all))
            log_phi_x = log_phi_all[idx_x]
            log_phi_p = log_phi_all[idx_p]
            output_phi_x = output_phi_all[idx_x]
            var_loss = torch.logsumexp(log_phi_x, dim=0) - math.log(len(log_phi_x)) - 1 * torch.mean(log_phi_p)

            # perform Mixup and calculate the regularization
            target_x = output_phi_x[:, 1].exp()
            target_p = torch.ones(len(data_p), dtype=torch.float32)
            target_p = target_p.cuda() if torch.cuda.is_available() else target_p
            rand_perm = torch.randperm(data_p.size(0))
            data_p_perm, target_p_perm = data_p[rand_perm], target_p[rand_perm]
            m = torch.distributions.beta.Beta(config['mix_alpha'], config['mix_alpha'])
            lam = m.sample()
            data = lam * data_x + (1 - lam) * data_p_perm
            target = lam * target_x + (1 - lam) * target_p_perm
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            out_log_phi_all = self.model(data)
            reg_mix_log = ((torch.log(target) - out_log_phi_all[:, 1]) ** 2).mean()

            # calculate gradients and update the network
            phi_loss = var_loss + config['lam'] * reg_mix_log
            opt_phi.zero_grad()
            phi_loss.backward()
            opt_phi.step()

            # update the utilities for analysis of the model
            # reg_avg.update(reg_mix_log.item())
            # phi_loss_avg.update(phi_loss.item())
            # var_loss_avg.update(var_loss.item())
            phi_p, phi_x = log_phi_p.exp(), log_phi_x.exp()
            # phi_p_avg.update(phi_p.mean().item(), len(phi_p))
            # phi_x_avg.update(phi_x.mean().item(), len(phi_x))

        # return phi_loss_avg.avg, var_loss_avg.avg, reg_avg.avg, phi_p_avg.avg, phi_x_avg.avg


    def predict_prob_y_given_X(self, X):
        self.model.eval()
        with torch.no_grad():
            if type(X) == np.ndarray:
                X = torch.tensor(X, dtype=lpu.constants.DTYPE)
            X = X.to(self.device)
            outputs = self.model(X)
            predicted_prob  = torch.nn.functional.sigmoid(outputs)
        return predicted_prob.cpu().numpy().squeeze()
    
            
    def predict_prob_l_given_y_X(self, X):
        return self.C
        
    def set_C(self, holdout_dataloader):
        X, l, y, _ = next(iter(holdout_dataloader))
        self.C = l[y == 1].mean().detach().cpu().numpy()
        self.prior = y.mean().detach().cpu().numpy()


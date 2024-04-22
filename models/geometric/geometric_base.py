import logging

import gpytorch
import numpy as np
import sklearn.base
import sklearn.metrics
import torch


import lpu.constants
import lpu.models.geometric.GVGP
import lpu.models.lpu_model_base

DEVICE = 'cpu'
EPOCH_BLOCKS = 1
INDUCING_POINTS_SIZE = 32
INTRINSIC_KERNEL_PARAMS = {
    'normed': False,
    'kernel_type': 'laplacian',
    'heat_temp': .01,
    'noise_factor': 0.,
    'amplitude': 0.5,
    'n_neighbor': 5,
    'lengthscale': 0.3,
    'neighbor_mode': 'distance',
    'power_factor': 1,
    'invert_M_first': False
}

LOG = logging.getLogger(__name__)


class GeometricGPLPUBase(lpu.models.lpu_model_base.LPUModelBase):
    def __init__(self, config, inducing_points_initial_vals=None, *args, **kwargs):
        super().__init__()
        self.config = config
        self.device = config.get('device', DEVICE)
        duplicate_keys = set(config.keys()) & set(kwargs.keys())
        if duplicate_keys:
            raise ValueError(f"Duplicate arguments have been set through both keyword arguments and the config file: {duplicate_keys}")
        
        if inducing_points_initial_vals is None:
            raise ValueError("inducing_points_initial_vals must be provided")
        
        if inducing_points_initial_vals is not None:
            self.inducing_points = torch.nn.Parameter(inducing_points_initial_vals)
            self.inducing_points_size = inducing_points_initial_vals.size(0)
        else:
            self.inducing_points_size = kwargs.get('inducing_points_size', config.get('inducing_points_size', INDUCING_POINTS_SIZE))
            self.inducing_points = torch.nn.Parameter(torch.zeros(self.inducing_points_size))

        # Extract training_size first from kwargs and, if no there, then from config
        self.training_size = kwargs.get('training_size', config.get('training_size', None))
        
        if self.training_size is None:
            raise ValueError(
                "training_size is not defined in the config. "
                "gpytorch.mlls.PredictiveLogLikelihood (which is used in this implementation) "
                "requires the number of training data points to be passed as an argument. "
                "Please define training_size in the config."
            )

        # creating the geometric VGP gp_model
        intrinsic_kernel_params = self._create_intrinsinc_kernel_params_from_config(
        )
        self.gp_model = lpu.models.geometric.GVGP.GeometricVGP(
            inducing_points=self.inducing_points,
            intrinsic_kernel_params=intrinsic_kernel_params).to(self.device)
        self.likelihood = self.CustomLikelihood(config, **kwargs).to(DEVICE).to(dtype=lpu.constants.DTYPE)
        self.mll = gpytorch.mlls.PredictiveLogLikelihood(
            self.likelihood,
            model=self.gp_model,
            num_data=self.training_size,
            beta=1).to(self.device)


    def _create_intrinsinc_kernel_params_from_config(self):
        """
        Extract intrinsic kernel parameters from `config` and return them as a dictionary
        """
        return self.config.get('intrinsic_kernel_params', INTRINSIC_KERNEL_PARAMS)


    def loss_fn(self, X_batch, l_batch):
        self.gp_model.update_input_data(X_batch)
        self.likelihood.update_input_data(X_batch)
        output = self.gp_model(X_batch)
        loss = -self.mll(output, l_batch)
        return loss

    def train_one_epoch(self, dataloader, optimizer):
        self.gp_model.train()
        self.likelihood.train()
        mini_batch_counter = 0
        total_loss = 0.
        for batch_idx, (X_batch, l_batch, _) in enumerate(
                dataloader):
            X_batch.to(self.device)
            l_batch.to(self.device)
            LOG.debug(f"Batch {batch_idx} is being processed now")
            loss = self.loss_fn(X_batch, l_batch)
            optimizer.zero_grad()
            total_loss += loss
            loss.backward()
            optimizer.step()

            mini_batch_counter += 1
        self.gp_model.eval()
        self.likelihood.eval()
        total_loss = total_loss.item()
        return total_loss

import abc
import logging

import gpytorch
import numpy as np
import sklearn.base
import sklearn.metrics
import torch
import abc


import LPU.constants
import LPU.models.geometric.GVGP
import LPU.models.lpu_model_base
import LPU.utils.utils_general
import LPU.utils.manifold_utils

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

LOG = LPU.utils.utils_general.configure_logger(__name__)


class GeometricGPLPUBase(LPU.models.lpu_model_base.LPUModelBase):

    class CustomLikelihood(gpytorch.likelihoods.Likelihood, metaclass=abc.ABCMeta):
        def __init__(self, **kwargs):
            super().__init__()

        @abc.abstractmethod
        def forward(self, function_samples, **kwargs):
            raise NotImplementedError("forward method must be implemented in the subclass")

        @abc.abstractmethod
        def update_input_data(self, X):
            raise NotImplementedError("update_input_data method must be implemented in the subclass")
        
    def __init__(self, config, inducing_points_initial_vals=None, *args, **kwargs):
        super().__init__()
        self.config = config
        self.device = config.get('device', DEVICE)
        self.cholesky_max_tries = config.get('cholesky_max_tries', 10)
        duplicate_keys = set(config.keys()) & set(kwargs.keys())
        if duplicate_keys:
            raise ValueError(f"Duplicate arguments have been set through both keyword arguments and the config file: {duplicate_keys}")
                
        if inducing_points_initial_vals is not None:
            self.inducing_points = torch.nn.Parameter(inducing_points_initial_vals)
            self.inducing_points_size = inducing_points_initial_vals.size(0)
        else:
            input_dim = config.get('input_dim', None)
            if input_dim is None:
                raise ValueError("input_dim is not defined in the config. "
                                 "Since inducing_points_initial_vals is not provided," 
                                 "the model needs to know the input dimension."
                                 "Please define input_dim in the config.")
            
            self.inducing_points_size = kwargs.get('inducing_points_size', config.get('inducing_points_size', INDUCING_POINTS_SIZE))
            self.inducing_points = torch.nn.Parameter(torch.zeros(self.inducing_points_size))

        # Extract training_size first from kwargs and, if no there, then from config
        self.training_size = kwargs.get('training_size', config.get('training_size', None))
        
        # if self.training_size is None:
        #     raise ValueError(
        #         "training_size is not defined in the config. "
        #         "gpytorch.mlls.PredictiveLogLikelihood (which is used in this implementation) "
        #         "requires the number of training data points to be passed as an argument. "
        #         "Please define training_size in the config."
        #     )

        # creating the geometric VGP gp_model
        self.intrinsic_kernel_params = self._create_intrinsinc_kernel_params_from_config(
        )

        self.gp_model = LPU.models.geometric.GVGP.GeometricVGP(
            inducing_points=self.inducing_points,
            intrinsic_kernel_params=self.intrinsic_kernel_params).to(self.device)
        self.likelihood = self.CustomLikelihood(**kwargs).to(DEVICE).to(dtype=LPU.constants.DTYPE)
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


    def loss_fn(self, l_output, l_batch):
        loss = -self.mll(l_output, l_batch)
        return loss

    def train_one_epoch(self, dataloader, optimizer, holdout_dataloader=None):
        self.gp_model.train()
        self.likelihood.train()
        overrall_loss = 0.
        num_of_batches = 0
        scores_dict = {}
        l_batch_concat = []
        y_batch_concat = []
        y_batch_concat_prob = []
        l_batch_concat_prob = []
        l_batch_concat_est = []
        y_batch_concat_est = []

        with gpytorch.settings.cholesky_max_tries(self.cholesky_max_tries):
            for batch_idx, (X_batch, l_batch, y_batch, _) in enumerate(dataloader):
                X_batch.to(self.device)
                l_batch.to(self.device)
                LOG.debug(f"Batch {batch_idx} is being processed now")
                optimizer.zero_grad()
                self.likelihood.update_input_data(X_batch)
                self.gp_model.update_input_data(X_batch)
                f_x = self.gp_model(X_batch)
                loss = self.loss_fn(f_x, l_batch)
                num_of_batches += 1
                
                y_batch_prob = self.predict_prob_y_given_X(f_x=f_x)
                l_batch_prob = self.predict_proba(X=X_batch, f_x=f_x)
                y_batch_est = self.predict_y_given_X(f_x=f_x)
                l_batch_est = self.predict(X=X_batch, f_x=f_x)

                if isinstance(y_batch_prob, np.ndarray):
                    y_batch_prob = torch.tensor(y_batch_prob, dtype=LPU.constants.DTYPE)
                    l_batch_prob = torch.tensor(l_batch_prob, dtype=LPU.constants.DTYPE)
                    y_batch_est = torch.tensor(y_batch_est, dtype=LPU.constants.DTYPE)
                    l_batch_est = torch.tensor(l_batch_est, dtype=LPU.constants.DTYPE)
                loss.backward()
                overrall_loss += loss.item()
                optimizer.step()
                l_batch_concat.append(l_batch.detach().cpu().numpy())
                y_batch_concat.append(y_batch.detach().cpu().numpy())
                y_batch_concat_prob.append(y_batch_prob.detach().cpu().numpy())
                l_batch_concat_prob.append(l_batch_prob.detach().cpu().numpy())
                y_batch_concat_est.append(y_batch_est.detach().cpu().numpy())
                l_batch_concat_est.append(l_batch_est.detach().cpu().numpy())

        y_batch_concat_prob = np.concatenate(y_batch_concat_prob)
        l_batch_concat_prob = np.concatenate(l_batch_concat_prob)
        y_batch_concat_est = np.concatenate(y_batch_concat_est)
        l_batch_concat_est = np.concatenate(l_batch_concat_est)
        y_batch_concat = np.concatenate(y_batch_concat)
        l_batch_concat = np.concatenate(l_batch_concat)
        scores_dict = self._calculate_validation_metrics(
            y_batch_concat_prob, y_batch_concat, l_batch_concat_prob, l_batch_concat, l_ests=l_batch_concat_est, y_ests=y_batch_concat_est
        )

        # average loss
        scores_dict['overall_loss'] = overrall_loss / (batch_idx + 1)
        return scores_dict 
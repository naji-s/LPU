import logging
import numpy as np
import warnings

import gpytorch
import gpytorch.distributions
import gpytorch.priors
import gpytorch.variational
import torch
import torch.distributions
import torch.nn
import torch.utils.data 

import LPU.models.geometric.geometric_base
import LPU.utils.matrix_utils as matrix_utils
import LPU.constants
import LPU.models.lpu_model_base
import LPU.datasets.animal_no_animal.animal_no_animal_utils
import LPU.models.lpu_model_base
import LPU.models.geometric.PsychM.PsychM
import LPU.utils.dataset_utils
import LPU.models.geometric.GVGP
import LPU.utils.utils_general  
import LPU.utils.manifold_utils

# Set up logging configuration
# logging.basicConfig(level=logging.INFO)  # Set the logging level as per your requirement

DEFAULT_CONFIG = {
    "warm_start": False,
    "inducing_points_size": 64,
    "learning_rate": 0.01,
    "num_epochs": 100,
    "stop_learning_lr": 1e-8,
    "device": "cpu",
    "epoch_block": 1,
    "intrinsic_kernel_params": {
        "normed": False,
        "kernel_type": "laplacian",
        "noise_factor": 0.01606854415673371,
        "amplitude": 0.015389532206728609,
        "n_neighbor": 5,
        "lengthscale": 76.37052411449602,
        "neighbor_mode": "connectivity",
        "power_factor": 0.10733466786949608,
        
        # "normed": False,
        # "kernel_type": "laplacian",
        # "heat_temp": 0.01,
        # "noise_factor": 0.0,
        # "amplitude": 0.5,
        # "n_neighbor": 5,
        # "lengthscale": 0.3,
        # "neighbor_mode": "distance",
        # "power_factor": 1,
        "invert_M_first": False,
    },
    "dataset_name": "animal_no_animal",
    "dataset_kind": "LPU",
    "data_generating_process": "SB",  # either of CC (case-control) or SB (selection-bias)
    "ratios": {
        "test": 0.4,
        "val": 0.05,
        "holdout": 0.0,
        "train": 0.55
    },
    "batch_size": {
        "train": 64,
        "test": 64,
        "val": 64,
        "holdout": 64
    }
}

# Create a logger instance
LOG = LPU.utils.utils_general.configure_logger(__name__)
 

class PsychM(LPU.models.geometric.geometric_base.GeometricGPLPUBase): 
    
    class CustomLikelihood(gpytorch.likelihoods.Likelihood):
        SCALE = 1.
        def __init__(self, config=None, *args, **kwargs):
            super().__init__()
            if config is None:
                config = {}
                LOG.warning("No config provided for PsycM likelihood. Using default config.")
            self.config = config
            is_SPM = config.get('is_SPM', False)
            warm_start_params = config.get('warm_start_params', None)
            input_dim = config.get('input_dim', None)
            self.gamma_prime_mean_init = torch.tensor([self.config.get('gamma_prime_mean_init', 0.)], dtype=LPU.constants.DTYPE)
            self.lambda_prime_mean_init = torch.tensor([self.config.get('lambda_prime_mean_init', 0.)], dtype=LPU.constants.DTYPE)
            self.gamma_prime_var_init = torch.tensor(self.config.get('gamma_prime_var_init', 0), dtype=LPU.constants.DTYPE)
            self.lambda_prime_var_init = torch.tensor(self.config.get('lambda_prime_var_init', 0), dtype=LPU.constants.DTYPE)

            duplicate_keys = set(config.keys()) & set(kwargs.keys())
            if duplicate_keys:
                raise ValueError(f"Duplicate arguments have been set through both keyword arguments and the config file: {duplicate_keys}")
            
            self.is_SPM = config.get('is_SPM', kwargs.get('is_SPM', is_SPM))

            # self.gp_model = gp_model
            # applying wamr start for psychometric function params
            if warm_start_params:
                gamma, lambda_, _, _,  true_alpha, true_beta = warm_start_params
                if (gamma > 0 or lambda_ > 0) and self.is_SPM:
                    raise ValueError("Warm start params are not valid for SPM")
            else:
                gamma = lambda_ = true_alpha = true_beta = 0
                self.trainable_psychm = True

            ################################################################################################            
            # setting the mean and variance for the variational distribution for the psychometric function
            ################################################################################################            
                        
            ## setting alpha prior
            
            # Parameter for the log of the diagonal elements to ensure they are positive
            # making sure no element of diagonal is too small to avoid overflow. 
            # Assuming LPU.constants.DTYPE and LPU.constants.EPSILON are predefined constants
            dtype = LPU.constants.DTYPE
            epsilon = LPU.constants.EPSILON

            # Creating a random tensor and comparing it with EPSILON
            random_tensor = torch.randn(input_dim, dtype=dtype)
            safe_tensor = torch.max(random_tensor, torch.tensor(epsilon, dtype=dtype))

            # Setting this as a parameter
            self.log_diag = torch.nn.Parameter(safe_tensor)
            
            # Parameter for the lower triangular elements below the diagonal
            # There are input_dim * (input_dim - 1) / 2 such elements
            self.lower_tri = torch.nn.Parameter(torch.randn(input_dim * (input_dim - 1) // 2, dtype=LPU.constants.DTYPE))
            self.variational_mean_alpha = torch.nn.Parameter(torch.zeros(input_dim, dtype=LPU.constants.DTYPE))# + true_alpha)
            
            
            # setting up beta prior
            self.variational_mean_beta = torch.nn.Parameter(torch.zeros(1, dtype=LPU.constants.DTYPE).squeeze())# + true_beta)
            self.variational_covar_beta = torch.nn.Parameter(torch.zeros(1, dtype=LPU.constants.DTYPE).squeeze()) 

            # setting up gamma and lambda prior
            self.gamma_prime_mean = torch.nn.Parameter(self.gamma_prime_mean_init)# + gamma)
            self.lambda_prime_mean = torch.nn.Parameter(self.lambda_prime_mean_init)# + lambda_ )
            self.gamma_prime_var = torch.nn.Parameter(self.gamma_prime_var_init)
            self.lambda_prime_var = torch.nn.Parameter(self.lambda_prime_var_init)

        def update_input_data(self, X):
            self.X = X

        def set_gamma_lambda(self):
            """
            Set the values for gamma and lambda parameters using fill_().
            """
            gamma_prime_value = self.gamma_prime_mean
            lambda_prime_value = self.lambda_prime_mean
            temp_gamma, temp_lambda, _ = torch.softmax(torch.concat([gamma_prime_value, lambda_prime_value, torch.zeros_like(lambda_prime_value)], dim=-1), dim=0)
            self.gamma.data.fill_(temp_gamma)
            self.lambda_.data.fill_(temp_lambda)

        def forward(self, function_samples):
            # Construct the diagonal part using the exponential of the log-diagonal
            diag = torch.diag(torch.exp(self.log_diag))
            
            # Fill the lower triangular part of an initially zero matrix
            tril_indices = torch.tril_indices(row=self.log_diag.size(0), col=self.log_diag.size(0), offset=-1)
            L = torch.zeros_like(diag)
            L[tril_indices[0], tril_indices[1]] = self.lower_tri   
            # Add the diagonal to get the full lower triangular matrix
            L += diag

            self.alpha = torch.distributions.MultivariateNormal(self.variational_mean_alpha, scale_tril=L).rsample()#torch.Size([function_samples.shape[0]]))
            self.beta = torch.distributions.Normal(self.variational_mean_beta, torch.exp(self.variational_covar_beta)).rsample()#torch.Size([function_samples.shape[0]]))
            gamma_prime_sample = torch.distributions.Normal(self.gamma_prime_mean, torch.nn.functional.softplus(self.gamma_prime_var)).rsample()#torch.Size([function_samples.shape[0]]))
            lambda_prime_sample = torch.distributions.Normal(self.lambda_prime_mean, torch.nn.functional.softplus(self.lambda_prime_var)).rsample()#torch.Size([function_samples.shape[0]]))
            self.linear_response = torch.matmul(self.X, self.alpha) + self.beta
            L_s = self.linear_response
            L_t = function_samples

            ### OLDER METHOD OF CLAC
            shape = L_t.shape
            ZERO = torch.zeros(L_t.shape)
            GAMMA = torch.tile(gamma_prime_sample, shape)
            LAMBDA = torch.tile(lambda_prime_sample, shape)
            L_s = L_s.unsqueeze(0).expand(shape[0], -1)
            logit = - torch.logsumexp(torch.stack([ZERO, -L_t], dim=1), dim=1) \
                    + torch.logsumexp(torch.stack([GAMMA, GAMMA - L_s, ZERO], dim=1), dim=1) \
                    - torch.logsumexp(torch.stack([GAMMA, + LAMBDA, ZERO], dim=1), dim=1) - torch.logsumexp(torch.stack([ZERO, -L_s], dim=1), dim=1) \
            - torch.logsumexp(torch.stack([GAMMA - L_t, GAMMA - L_t - L_s, LAMBDA, LAMBDA - L_t, LAMBDA - L_s, LAMBDA - L_t - L_s,-L_t, -L_s, -L_t - L_s], dim=1), dim=1)\
            + torch.logsumexp(torch.stack([GAMMA, GAMMA - L_s, GAMMA - L_t, GAMMA - L_s - L_t, 
                                                      LAMBDA, LAMBDA - L_s, LAMBDA - L_t, LAMBDA - L_t - L_s, ZERO, -L_t, -L_s, -L_t - L_s], dim=1), dim=1)

            return torch.distributions.Bernoulli(logits=logit)
        
        def freeze_parameters(self):
            """Freeze alpha and beta parameters."""
            self.variational_mean_alpha.requires_grad = False
            self.variational_covar_alpha.requires_grad = False
            self.variational_mean_beta.requires_grad = False
            self.variational_covar_beta.requires_grad = False

        def freeze_alpha(self):
            """Freeze alpha and beta parameters."""
            self.variational_mean_alpha.requires_grad = False
            self.variational_covar_alpha.requires_grad = False

        def unfreeze_parameters(self):
            """Unfreeze alpha and beta parameters."""
            self.variational_mean_alpha.requires_grad = True
            self.variational_covar_alpha.requires_grad = True
            self.variational_mean_beta.requires_grad = True
            self.variational_covar_beta.requires_grad = True

        def set_alpha_beta(self, alpha_value, beta_value):
            """Set fixed values for alpha and beta and freeze them."""
            self.variational_mean_alpha.data = torch.zeros_like(self.variational_mean_alpha) +  alpha_value
            self.variational_mean_alpha.requires_grad = False
            
            self.variational_covar_alpha.data = torch.full_like(self.variational_covar_alpha, 0)  # Optional: Set variance to 0 to indicate no variation
            self.variational_covar_alpha.requires_grad = False
            
            self.variational_mean_beta.data = torch.zeros_like(self.variational_mean_beta) + beta_value
            self.variational_mean_beta.requires_grad = False
            
            self.variational_covar_beta.data = torch.full_like(self.variational_covar_beta, 0)  # Optional: Set variance to 0 to indicate no variation
            self.variational_covar_beta.requires_grad = False     
    
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.config = config        

    def initialize_inducing_point_values(self, X):
        # Assuming INDUCING_POINTS_SIZE is defined as an attribute or constant
        # This method selects a subset of the training data to be used as inducing points
        inducing_point_values = X[:self.inducing_points_size]
        return inducing_point_values

    def _create_intrinsinc_kernel_params_from_config(self):
        """
        Extract intrinsic kernel parameters from `config` and return them as a dictionary
        """
        return self.config.get('intrinsic_kernel_params', None)#INTRINSIC_KERNEL_PARAMS)

    
    # def _initialize_likelihood(self, inducing_points):
    #     # Assuming `sig_X_train` is a parameter you need for your likelihood
    #     # It might be derived from your data or passed directly through `config`
    #     likelihood = LPU.models.geometric.psychmGGPC.CustomLikelihood(inducing_points.shape[-1]).to(DEVICE).to(dtype=LPU.constants.DTYPE)
    #     return likelihood    
    
    # def predict(self, X, return_std=False):
    #     # Convert X to torch tensor
    #     X_tensor = torch.tensor(X, dtype=LPU.constants.DTYPE).to(self.device)

    #     # Set model and likelihood to evaluation mode
    #     self.gp_model.eval()
    #     self.likelihood.eval()

    #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #         predictions = self.gp_model(X_tensor)
    #         mean = predictions.mean

    #     # Optionally return standard deviation
    #     if return_std:
    #         std = predictions.stddev
    #         return mean.cpu().numpy(), std.cpu().numpy()
    #     else:
    #         return mean.cpu().numpy()

    def predict_prob_l_given_y_X(self, X=None):
        # gamma, lambda_, _ = torch.softmax(torch.tensor([self.likelihood.gamma_prime_mean, self.likelihood.lambda_prime_mean, 0.]), dim=0)
        # Normalize inputs for softmax to prevent underflow/overflow
        gamma_prime = self.likelihood.gamma_prime_mean
        lambda_prime = self.likelihood.lambda_prime_mean
        gamma_prime_norm, lambda_prime_norm = gamma_prime - torch.max(gamma_prime, lambda_prime), lambda_prime - torch.max(gamma_prime, lambda_prime)
        # Applying softmax on normalized values
        gamma, lambda_, _ = torch.softmax(torch.tensor([gamma_prime_norm, lambda_prime_norm, 0], dtype=LPU.constants.DTYPE), dim=0)
        
        linear_response = X @ self.likelihood.variational_mean_alpha + self.likelihood.variational_mean_beta
        output = (gamma + (1 - gamma - lambda_) * torch.sigmoid(linear_response)).cpu().detach().numpy()
        return output
        # return self.likelihood.probs.mean(axis=0).cpu().detach().numpy()

    def predict_prob_y_given_X(self, X=None, f_x=None):
        if f_x is None:
            self.gp_model.update_input_data(X)
            f_x = self.gp_model(X)
        y_prob = torch.nn.functional.sigmoid(f_x.rsample(sample_shape=torch.Size([100]))).mean(axis=0).cpu().detach().numpy()
        return y_prob

    def validate(self, dataloader, loss_fn=None):
        scores_dict = {}
        total_loss = 0.
        l_batch_concat = []
        y_batch_concat = []
        y_batch_concat_prob = []
        l_batch_concat_prob = []
        l_batch_concat_est = []
        y_batch_concat_est = []
        self.gp_model.eval()
        self.likelihood.eval()
        binary_kind = set(torch.unique(dataloader.dataset.y))
        with torch.no_grad():
            for batch_num, (X_batch, l_batch, y_batch, _) in enumerate(dataloader):
                self.gp_model.update_input_data(X_batch)
                self.likelihood.update_input_data(X_batch)
                f_x = self.gp_model(X_batch)
                self.likelihood(f_x)
                loss = loss_fn(f_x, l_batch)
                y_batch_prob = self.predict_prob_y_given_X(f_x=f_x)
                l_batch_prob = self.predict_proba(X=X_batch, f_x=f_x)
                y_batch_est = self.predict_y_given_X(f_x=f_x)
                l_batch_est = self.predict(X=X_batch, f_x=f_x)
                
                if isinstance(y_batch_prob, np.ndarray):
                    y_batch_prob = torch.tensor(y_batch_prob, dtype=LPU.constants.DTYPE)
                    l_batch_prob = torch.tensor(l_batch_prob, dtype=LPU.constants.DTYPE)
                    y_batch_est = torch.tensor(y_batch_est, dtype=LPU.constants.DTYPE)
                    l_batch_est = torch.tensor(l_batch_est, dtype=LPU.constants.DTYPE)


                total_loss += loss.item()

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

        if binary_kind == {-1, 1}:
            y_batch_concat = (y_batch_concat + 1) / 2
            l_batch_concat = (l_batch_concat + 1) / 2
        scores_dict = self._calculate_validation_metrics(
            y_batch_concat_prob, y_batch_concat, l_batch_concat_prob, l_batch_concat, l_ests=l_batch_concat_est, y_ests=y_batch_concat_est
        )

        # for score_type in scores_dict:
        #     scores_dict[score_type] = np.mean(scores_dict[score_type])
        total_loss /= (batch_num + 1)
        scores_dict['overall_loss'] = total_loss

        return scores_dict        
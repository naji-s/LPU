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
import LPU.constants
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
    "inducing_points_size": 32,
    "learning_rate": 0.01,
    "num_epochs": 10,
    "stop_learning_lr": 1e-6,
    "device": "cpu",
    "epoch_block": 1,
    "intrinsic_kernel_params": {
        "normed": False,
        "kernel_type": "laplacian",
        "heat_temp": 0.01,
        "noise_factor": 0.0,
        "amplitude": 0.5,
        "n_neighbor": 5,
        "lengthscale": 0.3,
        "neighbor_mode": "distance",
        "power_factor": 1,
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
            is_SPM = config.get('is_SPM', False)
            warm_start_params = config.get('warm_start_params', None)
            input_dim = config.get('input_dim', None)

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
            LOG.info(f"True alpha: {true_alpha}")
            LOG.info(f"True beta: {true_beta}")
            self.variational_mean_alpha = torch.nn.Parameter(torch.zeros(input_dim, dtype=LPU.constants.DTYPE))# + true_alpha)
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
            # self.variational_covar_alpha = torch.nn.Parameter(torch.zeros(input_dim))
            # self.variational_covar_alpha_L = torch.nn.Parameter(torch.zeros(input_dim))
            # self.diag = torch.nn.Parameter(torch.randn(input_dim))
            # Parameter for the lower triangular elements below the diagonal
            # There are input_dim * (input_dim - 1) / 2 such elements
            # self.lower_tri = torch.nn.Parameter(torch.randn(input_dim * (input_dim - 1) // 2))
            
            self.variational_mean_beta = torch.nn.Parameter(torch.zeros(1, dtype=LPU.constants.DTYPE).squeeze())# + true_beta)
            self.variational_covar_beta = torch.nn.Parameter(torch.zeros(1, dtype=LPU.constants.DTYPE).squeeze()) 

            # self.gamma = torch.nn.Parameter(torch.zeros(1))
            # self.lambda_ = torch.nn.Parameter(torch.zeros(1))

            self.gamma_mean = torch.nn.Parameter(torch.zeros(1, dtype=LPU.constants.DTYPE))# + gamma)
            self.lambda_mean = torch.nn.Parameter(torch.zeros(1, dtype=LPU.constants.DTYPE))# + lambda_ )
            # self.dirichlet_alpha_gamma = torch.nn.Parameter(torch.zeros(1))
            # self.dirichlet_alpha_lambda = torch.nn.Parameter(torch.zeros(1))
            # self.dirichlet_alpha_dummy = torch.ones(1)

            self.gamma_var = torch.nn.Parameter(torch.randn(1, dtype=LPU.constants.DTYPE))
            self.lambda_var = torch.nn.Parameter(torch.randn(1, dtype=LPU.constants.DTYPE))
            self.anchor_weight = torch.nn.Parameter(torch.zeros(1, dtype=LPU.constants.DTYPE).squeeze(), requires_grad=False)
            self.train_GP = torch.nn.Parameter(torch.ones(1, dtype=LPU.constants.DTYPE).squeeze(), requires_grad=False)
            # self.alpha = torch.nn.Parameter(torch.randn(input_dim))
            # self.beta = torch.nn.Parameter(torch.randn(1))
            # log sigmoid is used in the forward layer for loss to reduce floating point errors
            # self.psych_logsigmoid = torch.nn.LogSigmoid()
            # self.gp_logsigmoid = torch.nn.LogSigmoid()

            # self.sigmoid = torch.nn.Sigmoid()
            # self.softmax = torch.nn.Softmax(0)
            # self.L_param = torch.nn.Parameter(torch.randn(input_dim, input_dim))

        def update_input_data(self, X):
            self.X = X

        
        def forward(self, function_samples):
            
            # self.variational_covar_alpha = torch.matmul(L, L.transpose(-1, -2)) + torch.eye(L.shape[0]) * LPU.constants.EPSILON
            # self.alpha = torch.distributions.LowRankMultivariateNormal(loc=self.variational_mean_alpha, cov_factor=L, cov_diag=torch.eye(L.shape[0]) * LPU.constants.EPSILON).rsample()#torch.Size([function_samples.shape[0]]))
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
            gamma_sample = torch.distributions.Normal(self.gamma_mean, torch.exp(self.gamma_var)).rsample()#torch.Size([function_samples.shape[0]]))
            lambda_sample = torch.distributions.Normal(self.lambda_mean, torch.exp(self.lambda_var)).rsample()#torch.Size([function_samples.shape[0]]))
            # gamma_sample = self.gamma_var
            # lambda_sample = self.lambda_var
            self.linear_response = torch.matmul(self.X, self.alpha) + self.beta
            self.gamma, self.lambda_, _ = torch.softmax(torch.concat([gamma_sample, lambda_sample, torch.zeros_like(lambda_sample)], dim=-1), dim=0)
            self.psychm_response = self.gamma + (1 - self.gamma - self.lambda_) * torch.sigmoid(self.linear_response)
            L_s = self.linear_response
            L_t = function_samples
            # self.psychm_response = torch.sigmoid(self.linear_response)
            # probs = torch.sigmoid(self.linear_response) * torch.sigmoid(function_samples) * self.train_GP
            # other_probs = torch.exp(torch.nn.function(self.linear_response) + torch.nn.functional.logsigmoid(function_samples))
            # logsigs = torch.nn.functional.logsigmoid(self.linear_response) + torch.nn.functional.logsigmoid(function_samples)
            # return torch.distributions.Bernoulli(probs=torch.exp(logsigs))#.mean(0))
            # return torch.distributions.Bernoulli(logits=logits)#.mean(0))
            # other_probs = self.psychm_response * torch.sigmoid(function_samples)
            ########################################################################
            ########################################################################
            # def softmax_loss(self, sig_input=None, psych_input=None, l=None, params=None, is_padded=False):
            #     """sig_input needs to be padded here"""
            #     if self.is_SPM:
            #         sig_a, sig_b, psych_alpha, psych_beta = self.partition_params(params)
            #         gamma_sample, lambda_sample = self.gamma_sample_, self.lambda_sample_
            #     else:
            #         sig_a, sig_b, psych_alpha, psych_beta, gamma_sample, lambda_sample = self.partition_params(params)

            # For function_samples, assuming self.kernel_mat is already a PyTorch tensor

            # Reshaping tensors
            # self.linear_response = self.linear_response.view(-1, -1)
            # function_samples = function_samples.view(-1, 1)
            shape = self.linear_response.shape
            zero = torch.zeros(1, device=self.linear_response.device)

            # Using PyTorch's logsumexp function and other tensor operations
            # A_leftover = torch.logsumexp(torch.stack([torch.tile(gamma_sample, shape), gamma_sample - self.linear_response, torch.tile(zero, shape)], dim=1), dim=1)
            # H_F_B_G = -torch.logsumexp(torch.stack([torch.tile(zero, shape), -self.linear_response], dim=1), dim=1) \
            #         - torch.logsumexp(torch.stack([torch.tile(zero, function_samples.shape), -function_samples], dim=1), dim=1) \
            #         #- torch.logsumexp(torch.stack([torch.tile(zero, shape), torch.tile(lambda_sample, shape), torch.tile(gamma_sample, shape)], dim=1), dim=1)
            # log_like = A_leftover + H_F_B_G
            # log_like = torch.clamp(log_like, max=0.)
            # torch.where((log_like > 0) & (log_like < LPU.constants.EPSILON), torch.ones_like(log_like, device=log_like.device), log_like)
            # self.probs = torch.exp(log_like)
            # E_D = (1 - l) * torch.logsumexp(torch.stack([torch.tile(-lambda_sample, shape), -self.linear_response, -self.linear_response - lambda_sample, -function_samples, -function_samples - gamma_sample, -function_samples - lambda_sample, -self.linear_response - function_samples, -self.linear_response - function_samples - gamma_sample, -self.linear_response - function_samples - lambda_sample], dim=1), dim=1)


            ### OLDER METHOD OF CLAC
            shape = L_t.shape
            ZERO = torch.zeros(L_t.shape)
            GAMMA = torch.tile(gamma_sample, shape)
            LAMBDA = torch.tile(lambda_sample, shape)
            L_s = L_s.unsqueeze(0).expand(shape[0], -1)

            # A = L_t#torch.stack([L_t, ZERO], dim=1)
            # B = torch.logsumexp(torch.stack([GAMMA, gamma_sample + L_s, L_s], dim=1), dim=1)
            # C = torch.logsumexp(torch.stack([ZERO, GAMMA, LAMBDA, L_t, lambda_sample + L_t, L_s, L_s + gamma_sample, L_s+lambda_sample, lambda_sample + L_t + L_s], dim=1), dim=1)
            # # breakpoint()
            # D = torch.logsumexp(torch.stack([torch.tile(-lambda_sample, shape), 
            #                                  -L_s, 
            #                                  -L_s - lambda_sample, 
            #                                  -function_samples, 
            #                                  -function_samples - gamma_sample, 
            #                                  -function_samples - lambda_sample, 
            #                                  -self.linear_response - function_samples, 
            #                                  -self.linear_response - function_samples - gamma_sample, 
            #                                  -self.linear_response - function_samples - lambda_sample], dim=1), dim=1)
            # total = L_t + B - C - D
            # return torch.distributions.Bernoulli(logits=total)
            # breakpoint()
            logit = - torch.logsumexp(torch.stack([ZERO, -L_t], dim=1), dim=1) \
                    + torch.logsumexp(torch.stack([GAMMA, GAMMA - L_s, ZERO], dim=1), dim=1) \
                    - torch.logsumexp(torch.stack([GAMMA, + LAMBDA, ZERO], dim=1), dim=1) - torch.logsumexp(torch.stack([ZERO, -L_s], dim=1), dim=1) \
            - torch.logsumexp(torch.stack([GAMMA - L_t, GAMMA - L_t - L_s, LAMBDA, LAMBDA - L_t, LAMBDA - L_s, LAMBDA - L_t - L_s,-L_t, -L_s, -L_t - L_s], dim=1), dim=1)\
            + torch.logsumexp(torch.stack([GAMMA, GAMMA - L_s, GAMMA - L_t, GAMMA - L_s - L_t, 
                                                      LAMBDA, LAMBDA - L_s, LAMBDA - L_t, LAMBDA - L_t - L_s, ZERO, -L_t, -L_s, -L_t - L_s], dim=1), dim=1)

            return torch.distributions.Bernoulli(logits=logit)
            # total = torch.clamp(total, max=0.)
            # return torch.distributions.Bernoulli(probs=torch.exp(total))
            # return torch.distributions.Bernoulli(probs=other_probs)

            # return torch.distributions.Bernoulli(probs=self.probs)

            ########################################################################
            ########################################################################       

            # return torch.distributions.Bernoulli(logits=log_result)
                                                    
            # return torch.distributions.Bernoulli(probs=log_result)
            # return gpytorch.likelihoods.BernoulliLikelihood()
        
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
        gamma, lambda_, _ = torch.softmax(torch.tensor([self.likelihood.gamma_mean, self.likelihood.lambda_mean, 0.]), dim=0)
        linear_response = X @ self.likelihood.variational_mean_alpha + self.likelihood.variational_mean_beta
        output = (gamma + (1-gamma - lambda_) * torch.sigmoid(linear_response)).cpu().detach().numpy()
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
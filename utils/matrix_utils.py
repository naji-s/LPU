import gpytorch
import gpytorch.priors
import numpy as np
import scipy.linalg.lapack 
import torch

import LPU.constants
import LPU.utils.manifold_utils
# import matrix_utils
EPSILON = 1e-8

def find_closest_differentiable(D, X, temperature=0.1):
    """
    Compute a differentiable approximation of the closest row in X for each row in D.
    
    Parameters:
    - D: A tensor of shape [m, d], where each row is a d-dimensional vector.
    - X: A tensor of shape [n, d], where each row is a d-dimensional vector.
    - temperature: A scalar controlling the sharpness of the softmax. Lower values make the selection closer to hard indexing.
    
    Returns:
    - D_tilde: A tensor of shape [m, d] that is a differentiable approximation of the closest rows in X to the rows in D.
    """
    # Compute pairwise squared Euclidean distances
    diff = D.unsqueeze(1) - X.unsqueeze(0)
    distances = torch.sum(diff ** 2, dim=2)
    
    # Apply softmax to negative distances (to simulate attraction) with temperature scaling
    weights = torch.nn.functional.softmax(-distances / temperature, dim=1)
    
    # Compute the weighted sum over X to get a differentiable D_tilde
    D_tilde = torch.matmul(weights, X)
    # print ("weights shape:", weights.shape, "and distances shape:", distances.shape, "and D_tilde shape:", D_tilde.shape)
    
    return D_tilde

def find_closest(D, X):
    """
    For each row in D, find the closest row in X.
    
    Parameters:
    - D: A tensor of shape [m, d] where each row is a d-dimensional vector.
    - X: A tensor of shape [n, d] where each row is a d-dimensional vector.
    
    Returns:
    - D_tilde: A tensor of shape [m, d] where each row is the closest row in X to the corresponding row in D.
    """
    # Step 1: Compute pairwise squared Euclidean distances
    # Expand D and X to [m, 1, d] and [1, n, d] respectively and compute the difference
    # Then, sum over the last dimension to get squared distances of shape [m, n]
    diff = D.unsqueeze(1) - X.unsqueeze(0)
    distances = torch.sum(diff ** 2, dim=2)
    
    # Step 2: Find the index of the minimum distance in X for each row in D
    min_dist_indices = torch.argmin(distances, dim=1)
    
    # Step 3: Index X with the minimum distance indices to get D_tilde
    D_tilde = X[min_dist_indices]
    
    return D_tilde

def invert_psd_matrix(matrix, regularization=EPSILON):
    """
    Inverts a positive semi-definite matrix using Cholesky decomposition with regularization.

    Parameters:
    matrix (numpy.ndarray): The matrix to be inverted.
    regularization (float): A small value added to the diagonal for regularization.

    Returns:
    numpy.ndarray: The inverted matrix.
    """
    matrix = regularization * np.eye(matrix.shape[0]) + matrix
    # Perform Cholesky decomposition and invert
    try:
        cholesky_decomp = np.linalg.cholesky(matrix)
        inv_cholesky_decomp = scipy.linalg.lapack.dtrtri(cholesky_decomp, lower=False)[0]
        inverted_matrix = inv_cholesky_decomp @ inv_cholesky_decomp.T
        # inverted_matrix = np.linalg.inv(cholesky_decomp).T @ np.linalg.inv(cholesky_decomp)        

        return inverted_matrix
    except np.linalg.LinAlgError:
        # In case Cholesky decomposition fails
        print("Cholesky decomposition failed. Matrix might not be positive definite.")
        return None

def invert_psd_matrix_torch(matrix, max_retries=5, regularization=EPSILON):
    """
    Attempts to invert a positive semi-definite matrix using Cholesky decomposition with increasing regularization.

    Parameters:
    matrix (torch.Tensor): The matrix to be inverted.
    max_retries (int): Maximum number of retries with increased regularization.

    Returns:
    torch.Tensor: The inverted matrix, or None if inversion fails.
    """

    for attempt in range(max_retries):
        try:
            reg_matrix = matrix + regularization * torch.eye(matrix.size(0), device=matrix.device)
            cholesky_decomp = torch.linalg.cholesky(reg_matrix).to(matrix.device)
            cholesky_decomp_inv = torch.linalg.solve_triangular(cholesky_decomp, torch.eye(matrix.size(0), device=matrix.device), upper=False)
            inverted_matrix = cholesky_decomp_inv.T @ cholesky_decomp_inv
            return inverted_matrix
        except RuntimeError as e:
            # print(f"Attempt {attempt + 1}: Cholesky decomposition failed with regularization {regularization}. Retrying with increased regularization.")
            regularization *= 10  # Increase regularization by an order of magnitude

    print("Failed to invert matrix after maximum retries.")
    return None


class ModifiedKernel(gpytorch.kernels.Kernel):
    def __init__(self, main_kernel, D, intrinsic_kernel_params=None):
        """
        :param main_kernel: The main GPyTorch kernel (K).
        :param M: Tensor of shape (n, n).
        :param K_DD: Tensor of shape (m, m).
        :param D: Tensor of shape (m, d).
        """
        super(ModifiedKernel, self).__init__()
        self.intrinsic_kernel_params = intrinsic_kernel_params

        # K_DD = K_DD * 1e-6

        # self.register_buffer("M", torch.zeros(D.size(0), D.size(0), dtype=LPU.constants.DTYPE))
        # self.register_buffer("K_DD", K_DD)
        # self.M_outputscale = NormalPrior(10., 0.1)
        self.M_outputscale = torch.tensor(self.intrinsic_kernel_params['amplitude'], dtype=LPU.constants.DTYPE)#torch.nn.Parameter(torch.zeros(1))# NormalPrior(0.2, .01)

        # self.M_outputscale = torch.nn.Parameter(torch.zeros(1)-1)
        # self.D = D

        # self.register_buffer("D", D)
        # self.register_buffer("M", M)
        # self.register_buffer("M_inv", M_inv)
        
        # self.M_inv = M_inv
        # self.M = M
        self.main_kernel = main_kernel
        # self.D = D
        # self.M_inv_plus_K_DD__inv = M_inv_plus_K_DD__inv
        # K_DD=self.main_kernel(self.D, self.D).evaluate().detach()
        # self.register_buffer("K_DD", K_DD)
        # M_inv_plus_K_DD__inv = invert_psd_matrix_torch(self.M_inv * self.M_outputscale ** 2 + self.K_DD).detach()
        # self.register_buffer("M_inv_plus_K_DD__inv", M_inv_plus_K_DD__inv)

    def update_input_data(self, X):
        self.X = X
        W = LPU.utils.manifold_utils.build_W_torch(X, k_neighbours=self.intrinsic_kernel_params['n_neighbor'], lengthscale=self.intrinsic_kernel_params['lengthscale'], connectivity=self.intrinsic_kernel_params['neighbor_mode'])
        self.M, _= LPU.utils.manifold_utils.build_manifold_mat(W, self.intrinsic_kernel_params)

    def forward(self, x1, x2, diag=None, **params):
        """
        Compute the kernel function between x1 and x2.
        """
        length = torch.tensor(1., dtype=LPU.constants.DTYPE)
        K_x1_x2 = self.main_kernel(x1, x2) / length 

        K_X_x1 = self.main_kernel(self.X, x1) / length 
        K_X_x2 = self.main_kernel(self.X, x2) / length 

        ############################################################################################################
        # implementation where we penalize distances between the inducing points and the data points
        ############################################################################################################
        # as part of regulariziation beside smoothness with respect to the manifold
        # D_tilde = find_closest_differentiable(self.D, self.X)
        # W = LPU.utils.manifold_utils.build_W_torch(D_tilde, k_neighbours=self.intrinsic_kernel_params['n_neighbor'], lengthscale=self.intrinsic_kernel_params['lengthscale'], connectivity=self.intrinsic_kernel_params['neighbor_mode'])
        # self.M, _= LPU.utils.manifold_utils.build_manifold_mat(W, self.intrinsic_kernel_params)
        # self.M += torch.diag(torch.linalg.norm(self.D - D_tilde, dim=1) ** 2)

        self.K_XX = self.main_kernel(self.X, self.X).evaluate().detach()

        if self.intrinsic_kernel_params['invert_M_first']:
            self.M_inv = invert_psd_matrix_torch(self.M)
            self.M = self.M * self.M_outputscale ** 2
            self.M_inv = self.M / self.M_outputscale ** 2
            self.M_inv_plus_K_XX__inv = invert_psd_matrix_torch(self.M_inv / self.M_outputscale ** 2 + self.K_XX)
        else:
            self.M = self.M * self.M_outputscale ** 2
            if torch.abs(self.M_outputscale) >= LPU.constants.EPSILON:
                self.M_inv_plus_K_XX__inv = torch.linalg.solve(torch.eye(self.M.size(0), device=self.M.device) + torch.matmul(self.M, self.K_XX), self.M)
            else:
                self.M_inv_plus_K_XX__inv = torch.zeros_like(self.M, device=self.M.device)
            # print ("M_inv_plus_K_DD__inv?", torch.isnan(self.M_inv_plus_K_DD__inv).any())
    
        # self.M_inv_plus_K_DD__inv.register_hook(lambda grad: print("Gradient of M_inv_plus_K_DD__inv:", torch.isnan(grad).any()))
        # self.M_inv_plus_K_DD__inv = 
        # Determine if expansion is needed
        # if x1.dim() > self.K_DD.dim():
        #     # Calculate the size difference and the needed expansion size
        #     size_diff = x1.dim() - self.K_DD.dim()
        #     expand_size = x1.shape[:size_diff] + (-1,) * 2  # Add -1 for the last two dimensions to keep them unchanged
        #     self.K_DD = self.K_DD.expand(expand_size)
        #     self.M_inv_plus_K_DD__inv = self.M_inv_plus_K_DD__inv.expand(expand_size)
        # Ensure correct batch matrix multiplication
        modified_term = torch.matmul(torch.matmul(K_X_x1.transpose(-2, -1), self.M_inv_plus_K_XX__inv), K_X_x2)
        # modified_term.register_hook(lambda grad: print("Gradient of modified_term:", torch.isnan(grad).any()))

        output = K_x1_x2 - modified_term 
        if diag:
            output = torch.diagonal(output, dim1=-2, dim2=-1)  # Extract diagonals considering batch dimensions

        return output
    
# class VariationalGPModel(gpytorch.models.ApproximateGP):
#     def __init__(self, inducing_points, X, intrinsic_kernel_params=None):
#         self.intrinsic_kernel_params = intrinsic_kernel_params
#         variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
#         variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
#         super(VariationalGPModel, self).__init__(variational_strategy)
        
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.core_kernel = gpytorch.kernels.RBF()#lengthscale_prior=gpytorch.priors.NormalPrior(.3, .000001))
#         # self.core_kernel.lengthscale.requires_grad_(False)
#         # self.core_kernel.lengthscale_prior = np.sqrt(.1)
#         self.ambient_kernel = gpytorch.kernels.ScaleKernel(self.core_kernel)#, outputscale_prior=gpytorch.priors.NormalPrior(.25, .001))
#         # self.ambient_kernel.lengthscale.requires_grad = False
#         # self.ambient_kernel.outputscale = torch.tensor(500.25)  # Set the initial value
#         # self.ambient_kernel.raw_outputscale.requires_grad = False  # Prevent it from being updated
#         # self.ambient_kernel.raw_outputscale.requires_grad_(False)
#         # self.ambient_kernel.raw_outputscale = 1000.
#         # fixed_outputscale = torch.nn.Parameter(torch.tensor(0.6685), requires_grad=False)
#         # self.ambient_kernel.raw_outputscale = fixed_outputscale
#         # Set the outputscale to the desired fixed value
#         # self.ambient_kernel.outputscale = 30.25
#         # self.ambient_kernel.outputscale.requires_grad_(False)

#         # self.ambient_kernel.outputscale_prior = 1.#NormalPrior(.0325, 0.001)
#         # base_kernel = gpytorch.kernels.ScaleKernel(ambient_kernel)

#         # self.ambient_kernel.outputscale_prior = 0.0325  # Set the scale to a specific value                                        
#         if intrinsic_kernel_params:
#             # torch.linalg.solve_triangular(A, B)
#             # print ("M_inv_plus_K_DD__inv:", M_inv_plus_K_DD__inv)
#             self.covar_module = ModifiedKernel(main_kernel=self.ambient_kernel, D=inducing_points, X=X, intrinsic_kernel_params=intrinsic_kernel_params)
#         else:
#             self.covar_module = self.ambient_kernel
                
#     def forward(self, x):
#         # print ("mean:", mean_x)
#         covar_x = self.covar_module(x)
#         # print ("covar:", covar_x.eval())
#         self.x = x
#         mean_x = self.mean_module(x)
#         mvn = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#         # print("THE LENGTHSCALE", self.ambient_kernel.lengthscale)
#         # Check the shape of a sample        
#         # Update loss term
#         # new_added_loss_term = LogLikelihoodPosOnly(dtype=x.dtype, device=x.device)
#         # self.update_added_loss_term("random_added_loss", new_added_loss_term)        
#         return mvn
        
    

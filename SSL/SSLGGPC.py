import gpytorch
import torch
from gpytorch.likelihoods import Likelihood
import torch.distributions as distributions
import gpytorch
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import Likelihood
from gpytorch.distributions import MultivariateNormal
import torch.distributions as distributions

import gpytorch
import torch
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import LPU.models.PsychM.psychm as psychm  # Assuming this is a custom module you have for the VariationalGPModel
torch.set_default_dtype(torch.float64)
# Assuming your VariationalGPModel and INTRINSIC_KERNEL_PARAMS are defined in a module named 'your_module_name'
# from your_module_name import VariationalGPModel, INTRINSIC_KERNEL_PARAMS
INTRINSIC_KERNEL_PARAMS = {'normed': False, 'kernel_type': 'laplacian', 'heat_temp': .7, 'noise_factor': .0, 
                        'amplitude': 1., 'n_neighbor': 5, 'lengthscale': 1., 'neighbor_mode': 'connectivity',
                        'power_factor': 1}

class SSLGPC:
    class CustomLikelihood(Likelihood):
        def __init__(self, X):
            super().__init__()
            
            self.X = X  # Store X
            # log sigmoid is used in the forward layer for loss to reduce floating point errors
            self.m = torch.nn.LogSigmoid()
            
        def update_input_data(self, X):
            self.X = X

        def forward(self, function_samples):
            return torch.distributions.Bernoulli(logits=function_samples)

    def __init__(self, inducing_points, intrinsic_kernel_params=None):
        self.model = psychm.VariationalGPModel(inducing_points, intrinsic_kernel_params)

    def fit(self, X_l, Y_l, X_u, num_iter=200):

        all_X = torch.concat([X_l, X_u])
        self.likelihood = self.CustomLikelihood(all_X)
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=inducing_points.size(0))
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.05)


        self.model.train()
        self.likelihood.train()
        for i in range(num_iter):
            self.optimizer.zero_grad()
            output = self.model(X_l)
            loss = -self.mll(output, Y_l)
            loss.backward()
            self.optimizer.step()
            if (i + 1) % 10 == 0:
                print(f'Iter {i+1}/{num_iter} - Loss: {loss.item()}')

    def predict(self, X):
        self.model.eval()
        self.likelihood.eval()
        self.likelihood.update_input_data(X)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.model(X).mean
            return self.likelihood(preds).mean#.sample_n(100).mean(1)

# Usage example:
# Assuming inducing_points is defined and is a tensor
# ssl_gpc = SSLGPC(inducing_points, INTRINSIC_KERNEL_PARAMS)
# X_l, Y_l should be defined as labeled data tensors
# ssl_gpc.fit(X_l, Y_l)
# For prediction, assuming X_test is defined
# predictions = ssl_gpc.predict(X_test)

if __name__ == "__main__":
    np.random.seed(5)
    X, Y = make_moons(n_samples=200, noise=0.05)
    ind_0 = np.nonzero(Y == 0)[0]
    ind_1 = np.nonzero(Y == 1)[0]
    ind_2 = np.nonzero(Y == 1)[0]
    # Y[ind_0] = -1

    ind_l0=np.random.choice(ind_0,1,False)
    ind_u0=np.setdiff1d(ind_0,ind_l0)

    ind_l1 = np.random.choice(ind_1, 1, False)
    ind_u1 = np.setdiff1d(ind_1, ind_l1)

    ind_l2 = np.random.choice(ind_2, 1, False)
    ind_u2 = np.setdiff1d(ind_2, ind_l2)

    # Xl=torch.tensor(np.vstack([X[ind_l0,:],X[ind_l1,:], X[ind_l2, :]]), dtype=torch.float64   )
    # Yl=torch.tensor(np.hstack([Y[ind_l0],Y[ind_l1], Y[ind_l2]]), dtype=torch.float64)
    # Xu=torch.tensor(np.vstack([X[ind_u0,:],X[ind_u1,:], X[ind_l2, :]]), dtype=torch.float64)

    X_l=torch.tensor(np.vstack([X[ind_l0,:],X[ind_l1,:]]), dtype=torch.float64   )
    Y_l=torch.tensor(np.hstack([Y[ind_l0],Y[ind_l1]]), dtype=torch.float64)
    X_u=torch.tensor(np.vstack([X[ind_u0,:],X[ind_u1,:]]), dtype=torch.float64)

    # plt.subplot(1,2,1)

    # plt.scatter(X_l[:,0], X_l[:,1],marker='+',c=Y_l)
    # plt.scatter(X_u[:,0], X_u[:,1],marker='.')
    # plt.show()
    # Fit the model
    # Usage
    # Initialize inducing points
    # Generate a shuffled set of indices for X_u
    n_inducing_points = 100
    shuffled_indices = torch.randperm(X_u.size(0))

    # Select the first n_inducing_points based on the shuffled indices
    inducing_points = X_u[shuffled_indices[:n_inducing_points]]    

    # Assuming inducing_points is defined and is a tensor
    ssl_gpc = SSLGPC(inducing_points, INTRINSIC_KERNEL_PARAMS)
    # X_l, Y_l should be defined as labeled data tensors
    ssl_gpc.fit(X_l, Y_l, X_u)
    # inducing_points, gp_model, likelihood = fit(Xl, Yl, Xu, ambient_kernel_params=None, intrinsic_kernel_params=None)

    # Determine the bounds of your input space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create a meshgrid for the input space
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))

    # Flatten the grid to pass through the model
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    grid_tensor = torch.tensor(grid, dtype=torch.float64)
    # Ensure the model and likelihood are in evaluation mode
    ssl_gpc.model.eval()
    ssl_gpc.likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Get the predictive distribution for the grid points
        # ssl_gpc.likelihood.update_input_data(grid_tensor)
        # observed_pred = ssl_gpc.likelihood(ssl_gpc.model(grid_tensor).mean)
        observed_pred = ssl_gpc.predict(grid_tensor)
        # Get the predicted probabilities for class 1
        pred_probs = observed_pred.reshape(xx.shape)
    # plt.subplot(1, 2, 2)

    # Plot the original data
    plt.scatter(X_l[:, 0], X_l[:, 1], marker='+', c=Y_l, label='Labeled Data')
    plt.scatter(X_u[:, 0], X_u[:, 1], marker='.', label='Unlabeled Data')

    # Plot the decision boundary or probability contours
    # For decision boundary, use levels=[0.5] if pred_probs represents probabilities
    plt.contourf(xx, yy, pred_probs, levels=10, cmap='RdBu', alpha=0.5)
    plt.colorbar()
    plt.contour(xx, yy, pred_probs, levels=[0.5], colors='k', linestyles='--')  # Decision boundary

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('Decision Boundary and Probability Contours')
    print(ssl_gpc.predict(torch.tensor(X[:5, :])))
    plt.legend()
    plt.show()


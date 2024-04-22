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
import lpu.models.PsychM.psychm as psychm  # Assuming this is a custom module you have for the VariationalGPModel
torch.set_default_dtype(torch.float64)
# Assuming your VariationalGPModel and INTRINSIC_KERNEL_PARAMS are defined in a module named 'your_module_name'
# from your_module_name import VariationalGPModel, INTRINSIC_KERNEL_PARAMS
INTRINSIC_KERNEL_PARAMS = {'normed': False, 'kernel_type': 'laplacian', 'heat_temp': .7, 'noise_factor': .0, 
                        'amplitude': 1., 'n_neighbor': 5, 'lengthscale': 1., 'neighbor_mode': 'connectivity',
                        'power_factor': 1}

class GGPC:
    class CustomLikelihood(Likelihood):
        def __init__(self, X):
            super().__init__()
            self.X = X  
            
        def update_input_data(self, X):
            self.X = X

        def forward(self, function_samples):
            return torch.distributions.Bernoulli(logits=function_samples)

    def __init__(self, inducing_points, intrinsic_kernel_params=None):
        self.model = psychm.VariationalGPModel(inducing_points, intrinsic_kernel_params)
        self.inducing_points = inducing_points
    def fit(self, X, y, num_iter=200):

        self.likelihood = self.CustomLikelihood(X)
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.inducing_points.size(0))
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.05)
        self.likelihood.update_input_data(X)

        self.model.train()
        self.likelihood.train()
        for i in range(num_iter):
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = -self.mll(output, y)
            loss.backward()
            self.optimizer.step()
            if (i + 1) % 10 == 0:
                print(f'Iter {i+1}/{num_iter} - Loss: {loss.item()}')

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) > threshold
    
    def predict_proba(self, X):
        self.model.eval()
        self.likelihood.eval()
        X_backup = self.likelihood.X
        self.likelihood.update_input_data(X)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.model(X).mean
            output =  self.likelihood(preds).mean
        self.likelihood.update_input_data(X_backup)
        return output
# Usage example:
# Assuming inducing_points is defined and is a tensor
# ssl_gpc = SSLGPC(inducing_points, INTRINSIC_KERNEL_PARAMS)
# X_l, Y_l should be defined as labeled data tensors
# ssl_gpc.fit(X_l, Y_l)
# For prediction, assuming X_test is defined
# predictions = ssl_gpc.predict(X_test)

if __name__ == "__main__":
    import sklearn.model_selection
    np.random.seed(5)
    X, y = make_moons(n_samples=200, noise=0.05)
    X = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)    
    y_bool = y.bool()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=42)
    plt.subplot(1,2,1)

    plt.scatter(X[y_bool, 0], X[y_bool, 1],marker='+',c='blue')
    plt.scatter(X[~y_bool, 0], X[~y_bool, 1],marker='_',c='red')

    # Fit the model
    # Usage
    # Initialize inducing points
    # Generate a shuffled set of indices for X_u
    n_inducing_points = 100
    shuffled_indices = torch.randperm(X.size(0))
    # Select the first n_inducing_points based on the shuffled indices
    inducing_points = X[shuffled_indices[:n_inducing_points]]    

    # Assuming inducing_points is defined and is a tensor
    ggpc = GGPC(inducing_points, INTRINSIC_KERNEL_PARAMS)

    ggpc.fit(X_train, y_train)

    # Determine the bounds of your input space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create a meshgrid for the input space
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))

    # Flatten the grid to pass through the model
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    grid_tensor = torch.tensor(grid, dtype=torch.float64)
    # Ensure the model and likelihood are in evaluation mode
    ggpc.model.eval()
    ggpc.likelihood.eval()

    # Get the predictive distribution for the grid points
    observed_pred = ggpc.predict_proba(grid_tensor)
    pred_probs = observed_pred.reshape(xx.shape)




    # Plot the decision boundary or probability contours
    # For decision boundary, use levels=[0.5] if pred_probs represents probabilities
    plt.contourf(xx, yy, pred_probs, levels=10, cmap='RdBu', alpha=0.5)
    plt.colorbar()
    plt.contour(xx, yy, pred_probs, levels=[0.5], colors='k', linestyles='--')  # Decision boundary

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('Decision Boundary and Probability Contours, balanced data')

    # Filter the positive examples and keep all negative examples
    positive_indices = np.where(y_train == 1)[0][:2]  # Get indices of first two positive examples
    subset_indices = np.concatenate((positive_indices, np.where(y_train == 0)[0]))

    # Create the subset of X_train and y_train
    X_subset = X_train[subset_indices]
    y_subset = y_train[subset_indices]


    # Assuming inducing_points is defined and is a tensor
    ggpc_2 = GGPC(inducing_points, INTRINSIC_KERNEL_PARAMS)
    # Train the model on the subset
    ggpc_2.fit(X_subset, y_subset)
    observed_pred_2 = ggpc_2.predict_proba(grid_tensor)
    # Get the predicted probabilities for class 1
    pred_probs_2 = observed_pred_2.reshape(xx.shape)
    plt.subplot(1, 2, 2)

    # Plot the original data
    plt.scatter(X_subset[y_subset.bool(), 0], X_subset[y_subset.bool(), 1], marker='+', c='blue', label='Pos Data')
    plt.scatter(X_subset[~y_subset.bool(), 0], X_subset[~y_subset.bool(), 1], marker='_', c= 'red', label='Neg Data')

    plt.contourf(xx, yy, pred_probs_2, levels=10, cmap='RdBu', alpha=0.5)
    plt.colorbar()
    plt.contour(xx, yy, pred_probs_2, levels=[0.5], colors='k', linestyles='--')  # Decision boundary

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('Decision Boundary and Probability Contours: 2 positive examples only')

    plt.legend()
    plt.show()


import gpytorch.models
import gpytorch.kernels
import gpytorch.means
import gpytorch.variational
import gpytorch.distributions
import gpytorch.constraints
import gpytorch.priors
import LPU.utils.matrix_utils


class GeometricVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, intrinsic_kernel_params):
        self.intrinsic_kernel_params = intrinsic_kernel_params
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        # variational_strategy = gpytorch.variational.BatchDecoupledVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GeometricVGP, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.core_kernel = gpytorch.kernels.LinearKernel()#lengthscale_prior=gpytorch.priors.NormalPrior(0.3, .0001))
        # self.core_kernel.lengthscale_prior = np.sqrt(.1)
        self.ambient_kernel = gpytorch.kernels.ScaleKernel(self.core_kernel)#, outputscale_prior=gpytorch.priors.NormalPrior(0.0325, .001))#, outputscale_constraint=gpytorch.constraints.Interval(0.01, 0.1))
        # self.ambient_kernel.outputscale_prior = 1.#NormalPrior(.0325, 0.001)
        # base_kernel = gpytorch.kernels.ScaleKernel(ambient_kernel)
        # self.ambient_kernel.outputscale_prior = 0.  # Set the scale to a specific value                                        

        if intrinsic_kernel_params:
            self.covar_module = LPU.utils.matrix_utils.ModifiedKernel(main_kernel=self.ambient_kernel, D=inducing_points, intrinsic_kernel_params=intrinsic_kernel_params)
        else:
            self.covar_module = self.ambient_kernel
    def update_input_data(self, X):
        self.covar_module.update_input_data(X)

    def forward(self, x):
        covar_x = self.covar_module(x)
        self.x = x
        mean_x = self.mean_module(x)
        mvn = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return mvn

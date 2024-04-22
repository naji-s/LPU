import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import kneighbors_graph
import scipy.sparse
import old_manifold_utils



class LapSVM(object):
    def __init__(self,ambient_kernel_params, intrinsic_kernel_params):
        self.ambient_kernel_params = ambient_kernel_params
        self.intrinsic_kernel_params = intrinsic_kernel_params


    def fit(self,X,Y,X_u):
        #construct graph
        self.X=np.vstack([X,X_u])
        Y=np.diag(Y)
        W = scipy.sparse.csr_matrix(old_manifold_utils.build_W_torch(self.X, k_neighbours=self.intrinsic_kernel_params['n_neighbor'], lengthscale=self.intrinsic_kernel_params['lengthscale'], connectivity=self.intrinsic_kernel_params['neighbor_mode']).numpy())
        # Computing Graph Laplacian
        # L = sparse.diags(np.array(W.sum(0))[0]).tocsr() - W
        L, _, _ = old_manifold_utils.build_manifold_mat(W, self.intrinsic_kernel_params)

        # Computing K with k(i,j) = kernel(i, j)
        K = self.ambient_kernel_params['kernel_function'](self.X,self.X,**self.ambient_kernel_params['scikit_kernel_parameters']) * self.ambient_kernel_params['amplitude']

        l=X.shape[0]
        u=X_u.shape[0]
        # Creating matrix J [I (l x l), 0 (l x (l+u))]
        J = np.concatenate([np.identity(l), np.zeros(l * u).reshape(l, u)], axis=1)

        # Computing "almost" alpha
        almost_alpha = np.linalg.inv(2 * self.ambient_kernel_params['amplitude'] * np.identity(l + u) \
                                     + (2  / (l + u) ** 2) * L.dot(K / self.ambient_kernel_params['amplitude'])).dot(J.T).dot(Y) 

        # Computing Q
        Q = Y.dot(J).dot(K).dot(almost_alpha) / self.ambient_kernel_params['amplitude']
        Q = (Q+Q.T)/2

        del W, L, K, J

        e = np.ones(l)
        q = -e

        # ===== Objectives =====
        def objective_func(beta):
            return (1 / 2) * beta.dot(Q).dot(beta) + q.dot(beta)

        def objective_grad(beta):
            return np.squeeze(np.array(beta.T.dot(Q) + q))

        # =====Constraint(1)=====
        #   0 <= beta_i <= 1 / l
        bounds = [(0, 1 / l) for _ in range(l)]

        # =====Constraint(2)=====
        #  Y.dot(beta) = 0
        def constraint_func(beta):
            return beta.dot(np.diag(Y))

        def constraint_grad(beta):
            return np.diag(Y)

        cons = {'type': 'eq', 'fun': constraint_func, 'jac': constraint_grad}

        # ===== Solving =====
        x0 = np.zeros(l)

        beta_hat = minimize(objective_func, x0, jac=objective_grad, constraints=cons, bounds=bounds)['x']

        # Computing final alpha
        self.alpha = almost_alpha.dot(beta_hat)

        del almost_alpha, Q

        # Finding optimal decision boundary b using labeled data
        new_K = self.ambient_kernel_params['kernel_function'](self.X,X,**self.ambient_kernel_params['scikit_kernel_parameters'])
        f = np.squeeze(np.array(self.alpha)).dot(new_K)

        self.sv_ind=np.nonzero((beta_hat>1e-7)*(beta_hat<(1/l-1e-7)))[0]

        ind=self.sv_ind[0]
        self.b=np.diag(Y)[ind]-f[ind]


    def decision_function(self,X):
        new_K = self.ambient_kernel_params['kernel_function'](self.X, X, **self.ambient_kernel_params['scikit_kernel_parameters'])
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        return f+self.b


# import matrix_utils
import numpy as np
import scipy.linalg
import scipy.sparse.csgraph
import scipy.special
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import svds, eigsh
import scipy.sparse
import scipy.sparse.linalg
import torch
# import tensorflow as tf
import lpu.utils.matrix_utils as matrix_utils
import lpu.constants
# from matrix_utils import invert_mat_with_cholesky
from numpy.linalg import LinAlgError, matrix_power
DELTA = 1e-16
from scipy.linalg import cholesky
from scipy.linalg.lapack import dtrtri

def make_laplacian_moment(L, nu, p):
    if not np.allclose(np.round(p), p):
        return torch.tensor(np.asarray(scipy.linalg.fractional_matrix_power(L + nu * torch.eye(L.shape[0]), p)), dtype=lpu.constants.DTYPE)
    else:
        return torch.matrix_power(L + nu * torch.eye(L.shape[0]), int(p))
    # # Compute eigenvalues and eigenvectors
    # eigenvalues, eigenvectors = torch.linalg.eigh(L)
    # # Modify eigenvalues
    # eigenvalues = (eigenvalues + nu)**p

    # # Reconstruct the matrix
    # D_powered = torch.diag(eigenvalues)
    # L_powered = torch.matmul(torch.matmul(eigenvectors, D_powered), eigenvectors.T)

    # return L_powered


def make_heat_kernel(L, temperatrure, k=None):
    if not k:
        k = L.shape[0] // 2
        k  = 10
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    exp_D = np.exp(-eigenvalues * temperatrure)
    exp_D = np.diag(exp_D)
    exp_L = eigenvectors @ exp_D @ eigenvectors.T
    return exp_L

import torch


def build_W_torch(features, k_neighbours, lengthscale, connectivity):
    epsilon = 1e-3
    # Calculate pairwise distances using torch.cdist
    pairwise_dist = torch.cdist(features, features)

    # Find the k nearest neighbors (including self)
    try:
        distances, indices = torch.topk(pairwise_dist, k_neighbours + 1, largest=False, sorted=True)
    except RuntimeError as e:
        if 'selected index k out of range' in str(e):
            # If k is too large, reduce it to the maximum possible value
            k_neighbours = pairwise_dist.shape[1] // 2
            distances, indices = torch.topk(pairwise_dist, k_neighbours + 1, largest=False, sorted=True)
        else:
            raise RuntimeError(f"Error in topk: {e}")

    if connectivity == 'connectivity':
        # Create a connectivity matrix (1 for connected, 0 for not connected)
        W = torch.zeros_like(pairwise_dist)
        W.scatter_(1, indices, 1)
        W = W + W.t()  # Make symmetric to ensure undirected graph
        # W[W > 0] = 1  # Ensure binary connectivity (either connected or not)
        W = torch.where(W > 0, torch.tensor(1.0, device=W.device), W)  # Ensure binary connectivity (either connected or not)
        W.fill_diagonal_(0)  # Remove self-loops
        # Apply the specified transformation to the weights
        W = torch.where(W > 0, 1 / (2 * lengthscale ** 2), W)
        # print("W GRAD?", W.requires_grad)
    elif connectivity == 'distance':
        # Create a distance matrix, but only for the k nearest neighbors
        pairwise_dist = pairwise_dist ** 2 / (2 * lengthscale ** 2)
        W = torch.full_like(pairwise_dist, torch.inf)
        W.scatter_(1, indices, distances)
        W = -(W + W.t()) / 2.  # Make symmetric to ensure undirected graph
        W = torch.where(W == -torch.inf, torch.tensor(0.0, device=W.device), torch.exp(W))

    else:
        raise ValueError("connectivity must be 'connectivity' or 'distance'")

    return W


def make_laplacian(W, normed=False):
    # Step 1: Create Degree Matrix (D)
    W_sum = torch.sum(W, dim=1)  # Sum of weights/edges per node
    D = torch.diag(W_sum)

    # Step 2: Compute Laplacian (L = D - W)
    L = D - W.to_dense()

    # Step 3: Normalize if required
    if normed:
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(W_sum))
        L = torch.eye(W.shape[0]) - torch.matmul(torch.matmul(D_inv_sqrt, W), D_inv_sqrt)

    # Convert to CSC format if necessary
    # L_csc = L.to_sparse().to(torch.float64)
    return L

def build_manifold_mat(W, manifold_kernel_info):
    """
    Builds a manifold matrix based on the input weight matrix W and manifold kernel information.

    Args:
        W (numpy.ndarray): The input weight matrix.
        manifold_kernel_info (dict): A dictionary containing information about the manifold kernel. 
                                     It should have the following structure:
                                     {
                                        'normed': bool,  # Indicates if the Laplacian is normalized
                                        'kernel_type': str,  # Type of kernel, e.g., 'heat' or 'laplacian'
                                        'heat_temp': float,  # Temperature parameter for the heat kernel (required if kernel_type is 'heat')
                                        'noise_factor': float,  # Noise factor in (L + noise_factor * I)^power_factor
                                        'power_factor': int,  # Power factor in (L + noise_factor * I)^power_factor
                                        'amplitude': float  # Amplitude parameter (required if kernel_type is 'laplacian')
                                     }

    Returns:
        tuple: A tuple containing the manifold matrix, Laplacian matrix, and the inverse of the manifold matrix.
    """

    # Convert the input matrix to a Compressed Sparse Row (CSR) matrix

    # try:
        # Compute the graph Laplacian matrix
    Lap_mat = make_laplacian(W, normed=manifold_kernel_info['normed'])
    # except scipy.sparse.SparseEfficiencyWarning as e:
        # raise scipy.sparse.SparseEfficiencyWarning(f"Efficiency Warning with CSRGRAPH: {e}")
    # except Exception as e:
        # Raise a more descriptive error message for unexpected exceptions
        # raise RuntimeError(f"Unexpected error with CSRGRAPH: {e}, Matrix type: {type(W)}")

    # Determine the type of kernel to apply
    kernel_type = manifold_kernel_info['kernel_type'].lower()

    try:
        if 'heat' in kernel_type:
            # Compute the heat kernel
            manifold_mat = make_heat_kernel(Lap_mat, manifold_kernel_info['heat_temp']) 
        elif 'laplacian' in kernel_type:
            # Compute the power of the Laplacian
            manifold_mat = make_laplacian_moment(Lap_mat, manifold_kernel_info['noise_factor'], manifold_kernel_info['power_factor']) 
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}. Supported types are 'heat' and 'laplacian'")
    except np.linalg.LinAlgError as e:
        # Handle linear algebra-specific errors
        raise np.linalg.LinAlgError(f"Linear algebra error during {kernel_type} calculation: {e}")
    except Exception as e:
        # Handle other exceptions
        raise RuntimeError(f"Unexpected error during {kernel_type} calculation: {e}")
    
    # Invert the PSD matrix of the manifold
    # manifold_mat_inv = matrix_utils.invert_psd_matrix_torch(manifold_mat)
    return manifold_mat, Lap_mat




# def build_W(features, k_neighbours, lengthscale, connectivity):
    
#     try:
#         if connectivity=='connectivity':
#             W = kneighbors_graph(features, k_neighbours, mode='connectivity', include_self=False)
#             W = (((W + W.T) > 0) * 1.)
#             W.data = np.squeeze(W.data) / (2 * lengthscale ** 2)

#         elif connectivity=='distance':
#             # if type(features).__name__ != 'ndarray':
#                 # if tf.keras.backend.shape(features)[0] >= 1:
#                 #     features = features.numpy()
#             W = kneighbors_graph(features, k_neighbours, mode='distance',include_self=False)
#             W = W.maximum(W.T)
#             W.data = np.square(W.data).ravel() 
#             W.data = np.squeeze(W.data) / (2 * lengthscale ** 2)
#             W.data = tf.math.exp(-W.data).numpy()
#             W.data = np.nan_to_num(W.data)

#     except Exception as e:
#         raise type(e)(str(e) + 'error is in kneighbors_graph')

#     # THE SPARSE CASE USING TENSORFLOW. k_nn_graph IMPLEMENTATION CURRENTLY IS BROKEN
#     # W = k_nn_graph(features, k=self.manifold_kernel_k, mode=self.opt['neighbor_mode'], include_self=False)
#     # W = tf.sparse.maximum(W, tf.sparse.transpose(W))
#     # W = convert_sparse_tensor_to_csr_matrix(W
#     # checking for floating point error

#     return W

def calculate_L_p_kernel(Lap_mat, manifold_kernel_noise, manifold_kernel_power, 
                         manifold_kernel_amplitude, use_eigsh=False, return_inverse=False):
    manifold_mat_inv = None

    if use_eigsh:
        # raise NotImplementedError("eigsh is implemented with a big mistake! needs correction: SVD cannot inverse that easily! the inverse of UDU^T is not U^T *1/D* U unfortunately!") 
        try:
            largest_power = np.inf
            decomposition_dict = svd_decompose(Lap_mat, padding=True)
            U = decomposition_dict['U']
            U_T = U.T
            noisy_D = decomposition_dict['D_vec'] + manifold_kernel_noise
            D_vec_powered = noisy_D ** manifold_kernel_power
            # log_noisy_D = tf.math.log(noisy_D).numpy()
            # log_noisy_D[log_noisy_D > largest_power] = largest_power
            # log_noisy_D_too_small_p = log_noisy_D < -largest_power
            # L_p_mat = tf.linalg.matmul(tf.linalg.matmul(U, D_vec_powered), U_T).numpy()\
            #                                         * self._manifold_kernel_amplitude ** 2

            manifold_mat = U @ np.diag(D_vec_powered) @ U_T * manifold_kernel_amplitude ** 2
            if return_inverse:
                manifold_mat_inv = U @ np.diag(1. / D_vec_powered) @ U_T / manifold_kernel_amplitude ** 2

            # print ("WTF???", manifold_mat @ manifold_mat_inv)
        except Exception as e:
            raise type(e)("caclulating L_r raises this error when relying on eigsh:" + str(e))
    else:
        decomposition_dict = None
        noisy_L = Lap_mat + manifold_kernel_noise * np.eye(Lap_mat.shape[0])
        noisy_L_p = matrix_power(noisy_L, manifold_kernel_power)
        # noisy_L_p_cholesky = cholesky(noisy_L_p)
        # noisy_L_p_cholesky_inv = dtrtri(noisy_L_p_cholesky)[0]
        from scipy.sparse import csr_matrix 
        manifold_mat = noisy_L_p * manifold_kernel_amplitude ** 2
        if return_inverse:
            t = time()
            manifold_mat_inv = invert_mat_with_cholesky(noisy_L_p)
            print ("Inversion in calculate_L_p_kernel took", t-time(), "seconds...")
            manifold_mat_inv = manifold_mat_inv / manifold_kernel_amplitude ** 2


    return manifold_mat, decomposition_dict, manifold_mat_inv

def svd_decompose(Lap_mat, padding=False):
    
    try:
        svd_func = lambda x: eigsh(A=x, k=Lap_mat.shape[0]-1, maxiter=50000)
        D, U = svd_func(Lap_mat)
    except Exception as e:
        raise type(e)("svd_func in svd_decompose() raises this error:" + str(e))
    if padding:
        U = np.hstack((U, np.ones(D.shape[0] + 1).reshape((-1, 1)) / np.sqrt(D.shape[0]+1)))
        D = np.hstack((D, np.zeros(1))) 
    decomposition_dict = dict()
    decomposition_dict['D_vec'] = D
    decomposition_dict['U'] = U
    return decomposition_dict    

def invert_mat_with_cholesky(mat=None, with_scipy=True):
    counter = 0
    eps_diag = np.diag(np.ones(mat.shape[0], dtype=np.float64) * DELTA)
    while True:
        biased_mat = mat + (counter > 0) * eps_diag * 10 ** counter
        try:
            if with_scipy:
                mat_cholesky = cholesky(tf.cast(biased_mat, tf.double))
            else:
                mat_cholesky = tf.linalg.cholesky(biased_mat).numpy()
                if np.isnan(mat_cholesky).any():
                    print ("FUCK ME!")
                    raise ValueError("tf.linalg.cholesky is failing in invert_mat_with_cholesky")
        except Exception as e:
            counter += 1
            logging = tf.get_logger()
            logging.warning(str(e) + "inverting with cholesky is failing, trial num:" + str(counter))
            if counter > 20:
                unnorm_M_inv = np.eye(mat.shape[0]) * np.sqrt(DELTA)
                break
            continue
        break
    if counter <= 10:
        if with_scipy:
            mat_cholesky_inv = dtrtri(mat_cholesky)[0]
        else:
            mat_cholesky_inv = tf.linalg.triangular_solve(tf.cast(mat_cholesky, tf.double), tf.linalg.eye(mat_cholesky.shape[0], dtype=tf.double), lower=True).numpy().T

        unnorm_M_inv = mat_cholesky_inv @ mat_cholesky_inv.T
    return unnorm_M_inv

def calculate_heat_kernel(Lap_mat, lbo_temperature, manifold_kernel_amplitude, use_eigsh, return_inverse=False):
    if use_eigsh:
        # raise NotImplementedError("eigsh is implemented with a big mistake! needs correction: SVD cannot inverse that easily! the inverse of UDU^T is not U^T *1/D* U unfortunately!") 
        try:
            decomposition_dict = svd_decompose(Lap_mat)
            D_vec = decomposition_dict['D_vec']
            U = decomposition_dict['U']
            U_T = U.T
            D = tf.math.exp(-lbo_temperature  * D_vec)
        except Exception as e:
            raise type(e)(str(e)+'Exponentiating D failed')
        try:
            exp_Lap_mat = tf.linalg.matmul(tf.linalg.matmul(U, 
                                                            tf.linalg.diag(tf.experimental.numpy.ravel(D))), U_T) *\
                                                            manifold_kernel_amplitude ** 2
            if return_inverse:
                exp_Lap_mat_inv = tf.linalg.matmul(
                                                tf.linalg.matmul(U, 
                                                    tf.linalg.diag(1. / tf.experimental.numpy.ravel(D))), U_T) /\
                                                        manifold_kernel_amplitude ** 2
            else: 
                exp_Lap_mat_inv = None


        except Exception as e:
            raise type(e)(str(e)+"exp_lap_mat calculation is failing WTF?!")
    else:
        decomposition_dict = None
        try:
            exp_Lap_mat = expm(-lbo_temperature * Lap_mat) * manifold_kernel_amplitude ** 2
            if return_inverse:
                exp_Lap_mat_inv = expm(lbo_temperature * Lap_mat) / manifold_kernel_amplitude ** 2
            else:
                exp_Lap_mat_inv = None



        except Exception as e:
            raise type(e)("exponentation L using \
                                scipy.sparse.linalg.expm is raising this error:" +\
                          str(e) + " Lap_mat has type:"+str(type(Lap_mat)))            

    return exp_Lap_mat, decomposition_dict, exp_Lap_mat_inv
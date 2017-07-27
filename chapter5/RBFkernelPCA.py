# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:06:24 2017

@author: Ajou
"""
###############################################################################
#### radial basis fucntion(RBF) kernel principal component analysis ###########
###############################################################################

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.
    
    Parameters
    ----------
     X: {NumPy ndarray}, shape = [n_samples, n_features]
    
     gamma: float
         Tuning parameter of the RBF kernel
    
     n_components: int
         Number of principal components to return
    
    Returns
    ------------
     X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
         Projected dataset
      
     lambdas: list
         Eigenvalues
    """
    # Calculate pairwise squared Euclidean distances in the MxN D dataset.
    sq_dists = pdist(X, 'sqeuclidean')
    
    mat_sq_dists = squareform(sq_dists)
    ## Compute the symmetric kernel matrix
    K = exp(-gamma*mat_sq_dists)
    
    # Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    # Obtaining eigenpairs from the centered kernel matrix
    eigvals, eigvecs = eigh(K)
    
    # Collect the top k eigvectors
    alphas = np.column_stack((eigvecs[:,-i] for i in range(1, n_components+1)))
    
    # Collect the correspoding eigenvalues
    lambdas = [eigvals[-i] for i in range(1, n_components+1)]
    
    return alphas, lambdas

    
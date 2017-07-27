# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 22:39:30 2017

@author: user
"""

from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
import numpy as np
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components = 2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1],
            color='blue', marker='o', alpha=0.5) 
plt.xlabel('PC1') 
plt.ylabel('PC2') 
plt.show() 

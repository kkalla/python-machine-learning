# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 22:42:19 2017

@author: user
"""

################################################################################
#######   Supervised data compression via LDA                        ###########
################################################################################

import numpy as np
from pca import X_train_std, y_train,X, y

## Compute mean vector
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
    print('MV {}: {}\n'.format(label,mean_vecs[label-1]))
    
## Compute within-class scatter matrix
d=13 # number of features
S_W = np.zeros((d,d))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter = np.zeros((d,d))
    for row in X[y==label]:
        row, mv = row.reshape(d,1), mv.reshape(d,1)
        class_scatter += (row-mv).dot((row-mv).T)
    S_W += class_scatter
print('Within-class scatter matrix: {}x{}'.format(S_W.shape[0],S_W.shape[1]))
## number of class label
print('Class label distribution: {}'.format(np.bincount(y_train)[1:]))

## scaling individual scatter matrix Si
S_W = np.zeros((d,d))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix: {}x{}'.format(S_W.shape[0],
      S_W.shape[1]))

## Compute between-class scatter matrix
mean_overall = np.mean(X_train_std,axis=0)
S_B = np.zeros((d,d))
for i, mean_vec in enumerate(mean_vecs):
    n = X[y==i+1, :].shape[0]
    mean_vec = mean_vec.reshape(d,1)
    mean_overall = mean_overall.reshape(d,1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print('Between-class scatter matrix: {}x{}'.format(S_B.shape[0],S_B.shape[1]))

## Select linear discriminants for new feature subspace

## Computing eigenvalues and eigenvectors
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
## Sort by desc order
eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(
        len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
print('Eigenvalues in decreasing order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])
    
## plot eigenvalues
import matplotlib.pyplot as plt
tot = sum(eigen_vals.real)
discr = [(i/tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, 14), discr, alpha=0.5, align='center', 
        label='individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid', 
         label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.show()

## Make transformation matrix W
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
               eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n',w)

## Project samples onto new feature space
X_train_lda = X_train_std.dot(w)
colors = ['r','b','g']
markers = ['s','x','o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0],
                X_train_lda[y_train==l,1], c=c ,label=l, marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()
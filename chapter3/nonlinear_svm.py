# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:31:09 2017

@author: Ajou
"""
###############################################################################
### Solving nonlinear problems using a kernel SVM                          ####
###############################################################################

from svm_data import X_xor,y_xor,np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import plot_decision_regions2 as pdr2

## Scatter plot
plt.scatter(X_xor[y_xor==1,0],X_xor[y_xor==1,1],c='b',marker='x',label='1')
plt.scatter(X_xor[y_xor==-1,0],X_xor[y_xor==-1,1],c='r',marker='s',label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()

## train a kernel SVM
svm = SVC(kernel='rbf', random_state=0,gamma=0.10,C=10.0)
svm.fit(X_xor,y_xor)
pdr2.plot_decision_regions(X_xor,y_xor,classifier = svm)
plt.legend(loc='upper left')
plt.show()

## gamma is cut-off parameter for the Gaussian sphere.
## Apply RBF kernel SVM to iris dataset
from scikit_learn_ex import X_train_std,y_train, X_combined_std,y_combined
svm = SVC(kernel='rbf',random_state=0,gamma=0.2,C=1.0)
svm.fit(X_train_std,y_train)
pdr2.plot_decision_regions(X_combined_std,y_combined,classifier=svm,
                           test_idx=range(105,150))
plt.xlabel('petal length[std]')
plt.ylabel('petal width [std]')
plt.legend(loc='upper left')
plt.show()

## Increase the value of gamma
svm = SVC(kernel='rbf',random_state=0,gamma=100.0,C=1.0)
svm.fit(X_train_std,y_train)
pdr2.plot_decision_regions(X_combined_std,y_combined,classifier=svm,
                           test_idx=range(105,150))
plt.xlabel('petal length[std]')
plt.ylabel('petal width [std]')
plt.legend(loc='upper left')
plt.show()
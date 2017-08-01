# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 21:10:23 2017

@author: user
"""
## Classifying samples using PCA, Logistic regression of scikit-learn
##import os
##os.chdir('../chapter2')
import plot_decision_regions as pdr
##os.chdir('../chapter5')
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from pca import X_train_std, X_test_std, y_train, y_test
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
pdr.plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='best')
plt.show()

## plot test data
pdr.plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='best')
plt.show()

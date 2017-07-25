# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:32:21 2017

@author: An joon
"""
##############################################################################
########## K-nearest neighbors - a lazy learning algorithm  ##################
##############################################################################

from scikit_learn_ex import X_train_std, y_train, X_combined_std,y_combined
import plot_decision_regions2 as pdr2
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std,y_train)
pdr2.plot_decision_regions(X_combined_std, y_combined,classifier=knn,
                           test_idx=range(105,150))
plt.xlabel('petal lenght')
plt.ylabel('petal width')
plt.show()

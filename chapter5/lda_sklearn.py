# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 23:36:59 2017

@author: user
"""

from lda import X_train_std, y_train
from pca import X_test_std, y_test
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from pca_sklearn import pdr
import matplotlib.pyplot as plt

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
pdr.plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='best')
plt.show()

X_test_lda = lda.transform(X_test_std)
pdr.plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='best')
plt.show()
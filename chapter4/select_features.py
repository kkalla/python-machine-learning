# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 12:12:59 2017

@author: An joon Lee
"""
import numpy as np
## Selecting meaningful features
## L1 regularization
from partition_stdize import X_train_std, y_train,X_test_std, y_test,df_wine
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1',C=0.1)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std,y_train))
print('Test accuracy:', lr.score(X_test_std,y_test))
lr.intercept_
lr.coef_

## plot the regularization path
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan','magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4,6):
    lr = LogisticRegression(penalty='l1', C=10.0**(c), random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.0**(c))
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:,column],label = df_wine.columns[column+1],
             color = color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coeff')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc = 'upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38,1.03),ncol=1,fancybox=True)
plt.show()

## Sequential feature selection algorithms
## dimensionality reduction: feature selection and feature extraction
## 1. Sequential Backward Selection (SBS)
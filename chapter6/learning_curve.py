# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:00:52 2017

@author: Ajou
"""
## Use the learning curve function from scikit-learn to evaluate the model
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from load_wdbcdata import X_train, y_train
import numpy as np

pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(penalty='l2', random_state=0))])
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
                                                        X=X_train,y=y_train,
                                                        train_sizes=np.linspace(
                                                                0.1,1.0,10),
                                                        cv=10, n_jobs=1)
train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)
plt.plot(train_sizes, train_mean, color='b',marker='o',markersize=5,
         label = 'training accuracy')
plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std,
                 alpha=0.15, color='b')
plt.plot(train_sizes, test_mean, color='green', linestyle='--',marker='s', 
          markersize=5,label='validation accuracy')
plt.fill_between(train_sizes,test_mean + test_std,test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.show()

## Addressing overfitting and underfitting with validation curves
from sklearn.learning_curve import validation_curve
param_range = [0.001,0.01,0.1,1.0,10.0,100.0]
train_scores, test_scores = validation_curve(estimator=pipe_lr, X=X_train,
                                             y=y_train, param_name='clf__C',
                                             param_range=param_range,cv=10)
train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)

plt.plot(param_range, train_mean, color='b',marker='o',markersize=5,
         label = 'training accuracy')
plt.fill_between(param_range, train_mean+train_std, train_mean-train_std,
                 alpha=0.15, color='b')
plt.plot(param_range,test_mean, color='green', linestyle='--',marker='s', 
          markersize=5,label='validation accuracy')
plt.fill_between(param_range,test_mean + test_std,test_mean - test_std,
                 alpha=0.15, color='green')
plt.grid()
plt.xscale('log')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.show()

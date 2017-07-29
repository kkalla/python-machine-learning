# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:52:37 2017

@author: Ajou
"""

## Stratified K-fold Cross-validation
from load_wdbcdata import y_train, X_train
from pipeline_ex import pipe_lr
import numpy as np
from sklearn.cross_validation import StratifiedKFold
kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test],y_train[test])
    scores.append(score)
    print('Fold: {}, Class dist.: {}, Acc: {:.3f}'.format(
            k+1, np.bincount(y_train[train]), score))
    
print('CV accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores),np.std(scores)))

## Use scikit-learn scorer
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(pipe_lr,X_train,y_train,cv = 10, n_jobs=1)
print('CV accuracy scores: {}'.format(scores))
print('CV accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))
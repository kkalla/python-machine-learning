# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:33:43 2017

@author: Ajou
"""

## Combining transformers and estimators in a pipeline
from load_wdbcdata import X_train, y_train, X_test, y_test
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))])
pipe_lr.fit(X_train, y_train)
print('Test Accuracy: {:.3f}'.format(pipe_lr.score(X_test,y_test)))

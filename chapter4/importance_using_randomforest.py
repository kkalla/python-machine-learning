# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:36:53 2017

@author: An joon Lee
"""
## Assessing feature importance with random forests

## import data
from partition_stdize import X_train, y_train,df_wine

## feature importance
import numpy as np
from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1)
forest.fit(X_train,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    ##print( "%2d) %-*s %f" % (f+1,30,feat_labels[indices[f]],
    ##     importances[indices[f]]))
    print('{:2d}) {:{prec}} {:f}'.format(f+1,feat_labels[indices[f]],
         importances[indices[f]],prec=30))
    
## make a plot
import matplotlib.pyplot as plt
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),
        importances[indices],color='lightblue',align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices],rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

## select features
X_selected = forest.transform(X_train, threshold=0.15)
X_selected.shape
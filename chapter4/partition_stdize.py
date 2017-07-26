# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:51:39 2017

@author: An joon Lee
"""
## read wine dataset from UCI machine learning repo
import pandas as pd
import numpy as np
df_wine = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
        header=None)
df_wine.columns = ['Class label','Alcohol','Malic acid','Ash',
                   'Alcalinity of ash','Magnesium','Total phenols','Flavanoids',
                   'Nonflavanoid phenols','Proanthocyanins',
                   'Color intensity','Hue',
                   'OD280/OD315 of diluted wines','Proline']
print('Class labels',np.unique(df_wine['Class label']))
df_wine.head()

## randomly partition dataset into test and training dataset
## using train_test_split() from scikit-learn's cross_validation submodule
from sklearn.cross_validation import train_test_split
X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=0)

## Bringing features onto the same scale
## min-max scaling
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

## standardiztion
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std=stdsc.transform(X_test)

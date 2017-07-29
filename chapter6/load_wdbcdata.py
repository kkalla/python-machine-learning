# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 10:26:22 2017

@author: Ajou
"""
## Loading the Breast Cancer Wisconsin dataset
## detailed information:  https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
import pandas as pd
df = pd.read_csv(url, header=None)
df.head(n=5)

## transform the class labes('M' and 'B') into integers
from sklearn.preprocessing import LabelEncoder 
X = df.loc[:,2:].values
y= df.loc[:,1].values
le = LabelEncoder()
y = le.fit_transform(y)

## divide dataset into a training set(80percent of the data) and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,
                                                    random_state=1)


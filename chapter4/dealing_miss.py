# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 14:50:43 2017

@author: An joon
"""
###############################################################################
#################           Dealing with missing data                   #######
###############################################################################

import pandas as pd
from io import StringIO
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))
df

## number of missing values per column
df.isnull().sum()

## eliminatig samples or features with missing values
## by rows
df.dropna()
## by cols
df.dropna(axis=1)
## only drop rows where all columns are NaN
df.dropna(how='all')
## drop rows that have not at least 4 non-NaN values
df.dropna(thresh=4)
## only drop rows where NaN appear in specific columns
df.dropna(subset=['C'])

## Imputing missing values
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean',axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
imputed_data
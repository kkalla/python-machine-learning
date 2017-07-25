# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:28:02 2017

@author: An joon
"""
## data set
import numpy as np
np.random.seed(0)
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:, 0] >0,X_xor[:,1]>0)
y_xor = np.where(y_xor,1,-1)
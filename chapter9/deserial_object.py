# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:20:17 2017

@author: Ajou
"""
#############################
## deserialize the objects ##
#############################

import pickle
import re
import os
os.chdir('movieclassifier')
from vectorizer import vect
## deserialize
clf = pickle.load(
        open(os.path.join('pkl_objects','classifier.pkl'),
             'rb'))

import numpy as np
label = {0:'negative', 1:'positive'}
example = ['I love this movie']
X = vect.transform(example)
print('Predictions: {}\nProbability: {:.2f}%'.format(
        label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))


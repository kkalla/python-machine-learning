# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 21:27:42 2017

@author: user
"""
## Imdb data can be downloaded at
## http://ai.stanford.edu/~amaas/data/sentiment/
## Read Movie review data to pd dataframe
import pyprind
import pandas as pd
import os
##os.chdir('chapter8')
pbar = pyprind.ProgBar(50000)
labels = {'pos':1, 'neg':0}
df = pd.DataFrame()
for s in ('test','train'):
    for l in ('pos','neg'):
        path = 'data/aclImdb/{}/{}'.format(s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file),'r',encoding='UTF-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]],ignore_index=True)
            pbar.update()
df.columns = ['review','sentiment']

## Shuffling dataframe
import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('data/movie_data.csv', index=False,encoding='UTF-8')

## Check csv data
df = pd.read_csv('data/movie_data.csv')
df.head(3)

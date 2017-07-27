# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:31:26 2017

@author: An joon
"""
## file 읽고 쓰기
## file = open()
## file.read() / file.write()
## file.close()

## with open('file-name') as 'acronym':
    # 작업
    #
    #
## 1.usa.gov data from bit.ly
import json
import os
os.chdir('basic')
path = 'usagov_bitly_data2013-05-15-1368641921.txt'
records = [json.loads(line) for line in open(path,encoding='UTF8')]

import pandas as pd
data1 = pd.read_json(path,lines=True)

## Select
## .loc -> label index / .iloc -> positionally index.
data1.loc['tz']

from pandas import DataFrame, Series
df1 = DataFrame(records)
df1.ix[0:3,2]

## time-zone field
time_zone = df1.loc[:,'tz']

def get_counts(sequence):
    counts={}
    for x in sequence:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    return counts
time_zone = time_zone.dropna()

counts = get_counts(time_zone)
counts

def top_counts(count_dict, n=10):
    value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
    value_key_pairs.sort(reverse=True)
    return value_key_pairs[:n]

top10 = top_counts(counts)
top10

###############################
from collections import Counter

count2 = Counter(time_zone)
type(count2)
count2.most_common(10) # ctrl + click -> function definition 볼수있음.
###############################
import numpy as np
df1['tz'].value_counts()
clean_tz = df1['tz'].fillna('Missing')
clean_tz[clean_tz=='']='Unknown'
tz_counts = clean_tz.value_counts()
tz_counts[:10].plot(kind='barh', rot=0)

#################################


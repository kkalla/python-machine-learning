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
data1.iloc[0]

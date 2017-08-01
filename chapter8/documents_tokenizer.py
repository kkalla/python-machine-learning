# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:21:56 2017

@author: Ajou
"""
import pandas as pd
df = pd.read_csv('data/movie_data.csv')
## Processing documents into tokens
## Splitting at whitespace characters
def tokenizer(text):
    return text.split()

## Word stemming using the Porter stemmer algorithm
## NLTK website: http://www.nltk.org/book/
## Other stemming algorithms can be found at
## http://www.nltk.org/api/nltk.stem.html 
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
tokenizer_porter(df.loc[0,'review'])

## Stop-word removal
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] 
if w not in stop]
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 22:10:50 2017

@author: user
"""

## Introduction to the Bag-of-words model
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining and weather is sweet'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())
a=bag.toarray()
## The sequence of of items in bag-of-words model above is called 1-gram or 
## unigram. By using ngram_range parameter in countVectorizer class, we can use
## different n-gram model. ex) ngram_range=(2.2) => 2-gram

## Compute tf-idf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

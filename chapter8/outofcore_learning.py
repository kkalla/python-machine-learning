# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:01:10 2017

@author: Ajou
"""

#######################################################
### Working with bigger data                    #######
### Online algorithms and out-of-core learning  #######
#######################################################

## Define tokenizer() function and cleaning data
import numpy as np
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')
def tokenizer(text):
    text = re.sub('<[^>]*>','',text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower())
    text = re.sub('[\W]+',' ',text.lower()) + ' '.join(emoticons).replace('-','')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

## Define a generator function that returns one document at a time
def stream_docs(path):
    with open(path, 'r',encoding='utf-8') as csv:
        next(csv)
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label
            
## Check performance stream_docs()
next(stream_docs(path='data/movie_data.csv'))

## Define get_minibatch() that returns a particular number of documents
def get_minibatch(doc_stream, size):
    """
    Parameter
    --------
     doc_stream: A document stream from *stream_docs* functions
     
     size: A number of documents want to be returned
     
    Return
    ---------
     docs: {array-like}
         texts
     y: {array-like}
         labels
    """
    docs, y = [],[]
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None,
    return docs, y

## Use HashingVectorizer
## https://sites.google.com/site/murmurhash/
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error = 'ignore', n_features=2**21,
                         preprocessor=None,tokenizer=tokenizer)
clf = SGDClassifier(loss='log',random_state=1,n_iter=1)
doc_stream = stream_docs(path='data/movie_data.csv')

## Start out-of-core learning
import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0,1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train,y_train, classes=classes)
    pbar.update()

## Use the last 5000 documents to evaluate the performance of our model
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: {:.3f}'.format(clf.score(X_test,y_test)))

## Update model using last 5000 documents
clf = clf.partial_fit(X_test,y_test)

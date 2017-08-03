# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:48:26 2017

@author: Ajou
"""

###################
## Training a logistic regression model for document classification
##################
import pandas as pd
df = pd.read_csv('data/movie_data.csv')
from nltk.corpus import stopwords
stop = stopwords.words('english')
import time

## Split train and test set
X_train = df.loc[:25000,'review'].values
y_train = df.loc[:25000,'sentiment'].values
X_test = df.loc[25000:,'review'].values
y_test = df.loc[25000:,'sentiment'].values

## Find the optimal set of parameters
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from documents_tokenizer import tokenizer, tokenizer_porter

tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)

param_grid = [{'vect__ngram_range': [(1,1)],
                'vect__stop_words': [stop,None],
                 'vect__tokenizer': [tokenizer, tokenizer_porter],
                 'clf__penalty': ['l1','l2'],
                 'clf__C': [1.0,10.0,100.0]},
                {'vect__ngram_range': [(1,1)],
                'vect__stop_words': [stop,None],
                 'vect__tokenizer': [tokenizer, tokenizer_porter],
                 'vect__use_idf': [False],
                 'vect__norm': [None],
                 'clf__penalty': ['l1','l2'],
                 'clf__C': [1.0,10.0,100.0]}]
lr_tfidf = Pipeline([('vect',tfidf),('clf',LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,scoring='accuracy',cv=5,
                           verbose=1,n_jobs=4)
start = time.time()
gs_lr_tfidf.fit(X_train,y_train)
end = time.time()
print('Execution time: {}'.format(end - start))
## Print best parameter set
print('Best parameter set: {}'.format(gs_lr_tfidf.best_params_))

## print 5-fold cross-validation accuracy score on the train and test data
print('CV Accuracy: {:.3f}'.format(gs_lr_tfidf.best_score_))
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: {:.3f}'.format(clf.score(X_test,y_test)))

## Naive Bayes classifier
## http://arxiv.org/pdf/1410.5329v3.pdf


## save classifier
## dump - takes the object that we want to pickle
## protocol = 4 - set to choose the latest and most efficient pickle protocol
import pickle
import os
dest = os.path.join('movieclf','pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(gs_lr_tfidf, open(os.path.join(dest, 'gs_lr.pkl'),'wb'),
            protocol = 4)
pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'),'wb'),
            protocol = 4)
pickle.dump(clf,
            open(os.path.join(dest,'classifier.pkl'),'wb'),
            protocol=4)
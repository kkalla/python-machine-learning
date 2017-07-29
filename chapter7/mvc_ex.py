# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 23:04:31 2017

@author: user
"""
## 다수결 투표 분류 예제

## Load iris data
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

## make train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5,
                                                    random_state=1)

## Train three different classifiers and look at their individual performances
## via cross-validation
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
clf1 = LogisticRegression(C=0.001, random_state=0)
clf2 = DecisionTreeClassifier(max_depth = 1, criterion='entropy',random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1)

pipe1 = Pipeline([['sc', StandardScaler()],['clf',clf1]])
pipe3 = Pipeline([['sc', StandardScaler()],['clf',clf3]])
clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
print('10-fold cross validation:\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(clf, X_train, y_train, cv = 10, scoring='roc_auc')
    print('ROC AUC: {:.2f} (+/- {:.2f}) [{}]'.format(
            scores.mean(),scores.std(),label))

## Combine individual classifiers for majority rule voting
import MajorityVoteClassifier as mvc
mv_clf = mvc.MajorityVoteClassifier(classifiers = [pipe1,clf2,pipe3])
clf_labels += ['Majority Voting']
all_clf = [pipe1,clf2,pipe3,mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator = clf,X= X_train,y=y_train,cv=10,
                             scoring='roc_auc')
    print("Accuracy: {:.2f} (+/- {:.2f}) [{}]".format(
            scores.mean(),scores.std(),label))
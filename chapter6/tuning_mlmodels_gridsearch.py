# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:45:37 2017

@author: Ajou
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from load_wdbcdata import X_train, y_train, X_test, y_test
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

pipe_svc = Pipeline([('scl', StandardScaler()),
                     ('clf',SVC(random_state=1))])
param_range = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
            {'clf__C': param_range,
             'clf__gamma': param_range,
             'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid,
                  scoring = 'accuracy', cv=10, n_jobs=-1)
gs = gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)
gs.grid_scores_

clf = gs.best_estimator_
clf.fit(X_train,y_train)
print('Test accuracy: {:.3f}'.format(clf.score(X_test,y_test)))

## Nested cross-validation
from sklearn.cross_validation import cross_val_score
import numpy as np
gs = GridSearchCV(estimator = pipe_svc, param_grid = param_grid,
                  scoring='accuracy', cv=2, n_jobs=-1)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores),np.std(scores)))

## Compare an SVM model to a simple decision tree classifier
## (tune its depth parameter)
from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(
        estimator = DecisionTreeClassifier(random_state=0),
        param_grid=[{'max_depth': [1,2,3,4,5,6,7,None]}],
        scoring = 'accuracy', cv = 5)
scores = cross_val_score(gs,X_train,y_train,scoring='accuracy',cv=2)
print('CV accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores),np.std(scores)))

## decision tree
from sklearn.tree import export_graphviz
tree = DecisionTreeClassifier(max_depth=6)
tree = tree.fit(X_train,y_train)
export_graphviz(tree,max_depth=6,out_file = 'gstree.dot')
## in cmd dot -Tpng gstree.dot -o tree.png
 
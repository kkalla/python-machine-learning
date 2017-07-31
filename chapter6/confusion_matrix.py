# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 14:47:55 2017

@author: user
"""

## Confusion matrix
from load_wdbcdata import X_train, y_train,X_test,y_test
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

pipe_svc = Pipeline([('scl', StandardScaler()),
                     ('clf',SVC(random_state=1))])
pipe_svc.fit(X_train,y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred = y_pred)
print(confmat)

## make plot using matshow()
import matplotlib.pyplot as plt
fig, ax= plt.subplots(figsize=(2.5,2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i, s=confmat[i,j],va='center', ha='center')
        
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

## Precision and recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
print('Precision: {:.3f}'.format(precision_score(y_true=y_test,
      y_pred = y_pred)))
print('Recall: {:.3f}'.format(recall_score(y_true=y_test,y_pred=y_pred)))
print('F1: {:.3f}'.format(f1_score(y_true=y_test,y_pred=y_pred)))

## List of scoring parameters can be found at
## http://scikit-learn.org/stable/modules/model_evaluation.html
from sklearn.metrics import make_scorer, f1_score
from sklearn.grid_search import GridSearchCV
param_range = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_grid = [{'clf__C': param_range,
               'clf__kernel': ['linear']},
            {'clf__C': param_range,
             'clf__gamma': param_range,
             'clf__kernel': ['rbf']}]
scorer = make_scorer(f1_score, pos_label=0)
gs = GridSearchCV(estimator = pipe_svc, param_grid=param_grid,scoring=scorer,
                  cv = 10)
gs = gs.fit(X_train,y_train)

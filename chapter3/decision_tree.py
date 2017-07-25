# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:46:13 2017

@author: An joon Lee
"""

###############################################################################
###############     Decision Tree learning                              #######
###############################################################################

"""
Maximizing information gain - getting the most bang for the buck
"""
## compare three different impurity criteria
import matplotlib.pyplot as plt
import numpy as np
def gini(p):
    return (p)*(1-(p)) + (1-p)*(1-(1-p))
def entropy(p):
    return -p*np.log2(p) - (1-p)*np.log2(1-p)
def error(p):
    return 1 - np.max([p,1-p])
x = np.arange(0.0,1.0,0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent,sc_ent,gini(x),err],
                           ['Entropy','Entropy (scaled)',
                            'Gini Impurity','Misclassification Error'],
                            ['-','-','--','-.'],
                            ['black','lightgray','red','green','cyan']):
    line = ax.plot(x, i, label=lab,linestyle=ls,lw=2, color=c)
ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.15),ncol=3,fancybox=True,
          shadow = False)
ax.axhline(y=0.5, linewidth=1, color='k',linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0,1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()

"""
Building a decision tree
"""
from sklearn.tree import DecisionTreeClassifier
from scikit_learn_ex import X_train, y_train,y_combined,X_test
import plot_decision_regions2 as pdr2
X_combined = np.vstack((X_train,X_test))
tree = DecisionTreeClassifier(criterion="entropy",max_depth=3, random_state=0)
tree.fit(X_train,y_train)
pdr2.plot_decision_regions(X_combined,y_combined,tree,test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc = 'upper left')
plt.show()

## visualize using the GraphViz program
from sklearn.tree import export_graphviz
export_graphviz(tree,out_file='tree.dot',feature_names=['petal length',
                                                        'petal width'])

"""
Combining weak to strong learners via random forests
"""
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy',n_estimators=10,
                                random_state=1,n_jobs=2)
forest.fit(X_train,y_train)
pdr2.plot_decision_regions(X_combined,y_combined,classifier=forest,
                           test_idx=range(105,150))
plt.xlable('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()
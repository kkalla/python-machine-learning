# Scikit-learn example using iris data
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

## make train and test data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state = 0)

## standardize features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

## train perceptron model on standardized data
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0 = 0.1, random_state = 0)
ppn.fit(X_train_std, y_train)

## predict target
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())

## classification accuracy of the perceptron
from sklearn.metrics import accuracy_score
print('Accuracy: %.2f' % accuracy_score(y_test,y_pred))

## plot decision regions
import plot_decision_regions2 as pdr2
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))
pdr2.plot_decision_regions(X=X_combined_std, y = y_combined,
                           classifier = ppn, test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()

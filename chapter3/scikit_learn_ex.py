# Scikit-learn example using iris data
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
import matplotlib.pyplot as plt
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))
pdr2.plot_decision_regions(X=X_combined_std, y = y_combined,
                           classifier = ppn, test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()

## Training logistic regression model using scikit-learn
from sklearn.linear_model import LogisticRegression


lr = LogisticRegression(C=1000.0, random_state = 0)
lr.fit(X_train_std, y_train)
pdr2.plot_decision_regions(X_combined_std, y_combined, classifier=lr,
        test_idx = range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc = 'upper left')
plt.show()

## Predict probability
print(lr.predict_proba(X_test_std[0,:]))

## Tackling overfitting via regularization
weights, params = [],[]
for c in np.arange(-5,5):
    lr = LogisticRegression(C = 10.0**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.0**c)
weights = np.array(weights)
plt.plot(params, weights[:,0],label = 'petal length')
plt.plot(params, weights[:,1],linestyle = '--', label = 'petal width')
plt.ylabel('weight coeff')
plt.xlabel('C')
plt.legend(loc = 'upper right')
plt.xscale('log')
plt.show()

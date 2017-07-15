# Perceptron learning algorithm
# reference by python machine learning, 2016 
# Sebastian Raschka, Packt publishing


import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                'machine-learning-databases/iris/iris.data',header = None)
df.tail() # use tail() method to check

import matplotlib.pyplot as plt
import numpy as np
y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa',-1,1)
X = df.iloc[0:100,[0,2]].values

plt.scatter(X[:50,0],X[:50,1],color='red',marker = 'o', label = 'setosa')
plt.scatter(X[50:100,0],X[50:100,1],color='blue',
        marker = 'x', label = 'versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc = 'upper left')
plt.show()

## Check error plot for each epoch
import perceptron as pt
pnn = pt.Perceptron(eta=0.1,n_iter=10)
pnn.fit(X,y)
plt.plot(range(1,len(pnn.errors_)+1),pnn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

## Plot two learning rate by number of epochs
import AdalineGD as ADA
fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
ada1 = ADA.AdalineGD(n_iter=10,eta=0.01).fit(X,y)
ax[0].plot(range(1, len(ada1.cost_) + 1),
           np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')
ada2 = ADA.AdalineGD(n_iter=10,eta=0.0001).fit(X,y)
ax[1].plot(range(1, len(ada2.cost_) + 1),
           ada2.cost_, marker = 'o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

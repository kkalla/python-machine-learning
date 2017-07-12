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
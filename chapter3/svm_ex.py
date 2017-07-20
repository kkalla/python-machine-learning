## Example of using Support Vector Machine in python
from scikit_learn_ex import X_train_std, y_train, X_combined_std,y_combined
from sklearn.svm import SVC
import plot_decision_regions2 as pdr2
import matplotlib.pyplot as plt

svm = SVC(kernel = 'linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
pdr2.plot_decision_regions(X_combined_std, y_combined, classifier=svm,
        test_idx = range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc = 'bottom right')
plt.show()

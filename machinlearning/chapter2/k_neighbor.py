from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import mglearn
import matplotlib.pyplot as plt
import numpy as np

iris_dataset=load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

knn= KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

print("{: .2f}".format(np.mean(y_pred==y_test)))
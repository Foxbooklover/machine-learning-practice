from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
'''
lasso = Lasso().fit(X_train, y_train)
print(f"{lasso.score(X_train, y_train):.2f}")
print(f"{lasso.score(X_test, y_test):.2f}")
print(np.sum(lasso.coef_!=0))
'''

lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print(f"{lasso001.score(X_train, y_train):.2f}")
print(f"{lasso001.score(X_test, y_test):.2f}")
print(np.sum(lasso001.coef_!=0))
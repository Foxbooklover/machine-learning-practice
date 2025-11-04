from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
import mglearn
import matplotlib.pyplot as plt

X, y= make_blobs(random_state=43)
'''
mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0","Class 1","Class 2"])
plt.show()
'''
linear_svm = LinearSVC().fit(X,y)
print(f"Coefficient {linear_svm.coef_.shape}")
print(f"intercept {linear_svm.intercept_.shape}")

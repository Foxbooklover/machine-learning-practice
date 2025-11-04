from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
import numpy as np

'''
X,y = mglearn.datasets.make_forge()

mglearn.discrete_scatter(X[: , 0],X[: , 1],y)
plt.legend(["Class0","Class1"] , loc = 4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape:" , X.shape)

X , y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()
'''
cancer=load_breast_cancer()
#print("cancer.keys(): \n", cancer.keys())

#print("shape of cancer data: \n", cancer.data.shape)
"""
print("Sample counts per class: \n",
      {str(n): int(v) for n, v in zip(cancer.target_names, np.bincount(cancer.target))})
"""
#print(cancer.feature_names)
#print(cancer.DESCR)

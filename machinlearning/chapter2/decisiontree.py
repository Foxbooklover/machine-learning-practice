
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import export_graphviz
import numpy as np
import mglearn
import matplotlib.pyplot as plt
'''
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)
tree=DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print(f"{tree.score(X_train, y_train ): .2f}")
print(f"{tree.score(X_test, y_test ): .2f}")

'''
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)
tree=DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

print(f"{tree.score(X_train, y_train ): .3f}")
print(f"{tree.score(X_test, y_test ): .3f}")

export_graphviz(tree, out_file="tree.dot",class_names=["malignant", "benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)
##plt.show()
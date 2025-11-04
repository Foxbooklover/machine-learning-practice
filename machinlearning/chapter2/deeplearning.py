
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import mglearn
import matplotlib.pyplot as plt
'''
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

mlp= MLPClassifier(solver='lbfgs', random_state=0,hidden_layer_sizes=[10,10]).fit(X_train, y_train)
mglearn.plots.plot_2d_separator(mlp,X_train, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)
plt.show()
'''

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42).fit(X_train, y_train)

##print(f"{mlp.score(X_train, y_train):.2f}")
##print(f"{mlp.score(X_test, y_test):.2f}")

mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)

X_train_scaled = (X_train - mean_on_train)/std_on_train
X_test_scaled = (X_test - mean_on_train)/std_on_train

mlp_scaled = MLPClassifier(max_iter=1000, alpha=1, random_state=0).fit(X_train_scaled, y_train)

print(f"{mlp_scaled.score(X_train_scaled,y_train):.2f}")
print(f"{mlp_scaled.score(X_test_scaled,y_test):.2f}")
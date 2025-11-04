from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

cancer=load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=43)
logreg = LogisticRegression().fit(X_train, y_train)

print(f"train : {logreg.score(X_train, y_train):.2f}")
print(f"test : {logreg.score(X_test, y_test):.2f}")
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
'''
X, y = load_iris(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
logreg = LogisticRegression().fit(X_train,y_train)
'''

iris = load_iris()
logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
mean_score = scores.mean()

res = cross_validate(logreg, iris.data, iris.target, cv=5, 
                     return_train_score=True)
res_df = pd.DataFrame(res)
##display(res_df)

kfold = KFold(n_splits=5,shuffle=True)
scores_1 = cross_val_score(logreg, iris.data, iris.target, cv=kfold)

loo = LeaveOneOut()
scores_2 = cross_val_score(logreg, iris.data, iris.target, cv=loo)

shuffle_split = ShuffleSplit(n_splits=10, test_size=0.5, train_size=.5 random_state=0)
scores_3 = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer


pipe = Pipeline([
    ('preprocessing', StandardScaler()),
    ('classifier', SVC())])


parm_grid = [
    {'classifier': [SVC()],
     'preprocessing': [StandardScaler(), None],
     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
     'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'classifier': [RandomForestClassifier()],
     'preprocessing': [None],
     'classifier__max_features':[1, 2, 3]}
]
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target)

grid = GridSearchCV(pipe, parm_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best cross-validation accuracy: {grid.best_score_:.3f}")
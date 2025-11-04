from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import mglearn
cancer=load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)
'''
training_accuracy=[]
test_accuracy=[]

neighbors_setting=range(1,11)

for n_neighbors in neighbors_setting:
    clf=KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))


plt.plot(neighbors_setting,training_accuracy)
plt.plot(neighbors_setting,test_accuracy)

plt.show()
'''
mglearn.plots.plot_knn_regression(n_neighbors=3)
plt.show()
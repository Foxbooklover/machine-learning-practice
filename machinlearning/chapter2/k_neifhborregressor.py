from sklearn.neighbors import KNeighborsRegressor
import mglearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

#reg =KNeighborsRegressor(n_neighbors=3)

#reg.fit(X_train,y_train)

#print(reg.predict(X_test))
#print("{:.2f}".format(reg.score(X_test,y_test)))

fig, axes = plt.subplots(1,3,figsize=(15,4))
line = np.linspace(-3,3,1000).reshape(-1,1)
for n_neighbors, ax in zip([1,3,9],axes):
    reg=KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train,y_train,'^',c=mglearn.cm2(0),markersize=8)
    ax.plot(X_test,y_test,'v',c=mglearn.cm2(1),markersize=8)
plt.show()
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import mglearn
from scipy.cluster.hierarchy import dendrogram, ward


X, y = make_blobs(random_state=0, n_samples=12)
dbscan = DBSCAN()
plt.scatter(X[:, 0], X[:, 1],  cmap=mglearn.cm3, s=60)

clusters = dbscan.fit_predict(X)

mglearn.plots.plot_dbscan()


plt.show()
'''
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)

plt.show()
'''
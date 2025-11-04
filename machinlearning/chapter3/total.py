from sklearn.datasets import fetch_lfw_people

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

import numpy as np
import matplotlib.pyplot as plt

X, y= fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = X[0].shape

mask = np.zeros(y.shape, dtype=bool)
for target in np.unique(y):
    mask[np.where(y == target)[0][:50]] = True
X_people = X[mask]/255
y_people = y[mask]


pca = PCA(n_components=100, whiten=True, random_state=0)
X_pca = pca.fit_transform(X_people)

##dbscan = DBSCAN(eps=15, min_samples=3)
##labels = dbscan.fit_predict(X_pca)

##noise = X_people[labels == -1]

for eps in np.linspace(1, 13, 7):
    print(f"eps = {eps : .2f}")
    dbscan = DBSCAN(eps=eps, min_samples=3)
    labels = dbscan.fit_predict(X_pca)
    print(f"Number of clusters: {len(np.unique(labels))}")
    print(f"Cluster sizes: {np.bincount(labels + 1)}")
    print(f"Noise points: {np.sum(labels == -1)}")


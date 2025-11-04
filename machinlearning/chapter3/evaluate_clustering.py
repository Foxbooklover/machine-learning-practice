from sklearn.datasets import make_moons
from sklearn.datasets import fetch_lfw_people
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
import mglearn

import numpy as np
import matplotlib.pyplot as plt

##X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
X, y = fetch_lfw_people(return_X_y=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)
km = KMeans(n_clusters=10, random_state=0)
labelss_km = km.fit_predict(X_pca)

image_shape = X[0].shape

center = km.cluster_centers_
eigenface = pca.inverse_transform(center).reshape(image_shape)

plt.imshow(eigenface, extent=(0, 2, 0, 1), cmap='gray')
plt.show()
'''
algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]

random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

fig, axes = plt.subplots(1, 4, figsize=(15,3), subplot_kw={'xticks':(), 'yticks':()})

axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)
axes[0].set_title(f"Random assignment - ARI: {adjusted_rand_score(y, random_clusters):.2f}")

for ax, algorithm in zip(axes[1:], algorithms):
    clusters = algorithm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)
    ax.set_title(f"{algorithm.__class__.__name__} - ARI: {adjusted_rand_score(y, clusters):.2f}")

plt.show()
'''
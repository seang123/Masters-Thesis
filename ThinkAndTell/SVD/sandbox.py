import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
import time

"""
Note
----
    Anything other than svd_solver="randomized" is slow af

"""

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
X = np.array([[-1, -1, -1], [-2, -1, -1], [-3, -2, -2], [1, 1, 1], [2, 1, 1], [3, 2, 2]])
print(X.shape)

X = np.random.normal(0, 1, (5000, 20000))

start = time.time()
pca = PCA(n_components=2, svd_solver = "randomized")
pca.fit(X)

print(np.sum(pca.explained_variance_ratio_))

print(time.time() - start)

start = time.time()
kpca = KernelPCA(n_components=2, kernel = "poly")
kpca.fit(X)

#print(np.sum(kpca.explained_variance_ratio_))

print(time.time() - start)

#print()
#print("exp var      ", pca.explained_variance_)
#print("exp var ratio", pca.explained_variance_ratio_)
#print("sing val     ", pca.singular_values_)

#print("mean         ", pca.mean_)
#print("components   ", pca.components_)

#Y = pca.transform(X)

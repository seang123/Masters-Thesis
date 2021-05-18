import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
import time

"""
Note
----
    Anything other than svd_solver="randomized" is slow af
    (27000,62000) rnd #'s for 2 components takes about 3.5 min
"""

#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#X = np.array([[-1, -1, -1], [-2, -1, -1], [-3, -2, -2], [1, 1, 1], [2, 1, 1], [3, 2, 2]])
#print(X.shape)

X = np.random.normal(0, 1, (27, 62))
#X = X.swapaxes(0, 1)
print(X.shape)

start = time.time()
pca = PCA(n_components=2, svd_solver = "randomized")
pca.fit(X)

print(time.time() - start, "sec")

print("sum expl var ratio:", np.sum(pca.explained_variance_ratio_))

c_ = pca.components_
print(c_.shape)

def transform_check():
    # transform using method
    Y_1 = pca.transform(X)
    print("Y_1", Y_1.shape)
    Y_1 = np.ascontiguousarray(Y_1)

    # transform using components 
    # want: (27, 2) <= (27, 620) * (620, 2)
    X1 = X - np.mean(X, 0) # .transform automatically centers data
    Y_2 = X1 @ np.transpose(c_)
    print("Y_2", Y_2.shape)

    print("Y_1 == Y_2:", np.allclose(Y_1, Y_2))


transform_check()
print("----")

Z = np.random.normal(0, 1, (3, 62))
Z_m = Z - np.mean(Z, 0)

Z1 = pca.transform(Z)
Z2 = Z_m @ np.transpose(c_)

print("z1", Z1.shape)
print("z2", Z2.shape)

print("Z1 == Z2:", np.allclose(Z1, Z2))


#print()
#print("exp var      ", pca.explained_variance_)
#print("exp var ratio", pca.explained_variance_ratio_)
#print("sing val     ", pca.singular_values_)

#print("mean         ", pca.mean_)
#print("components   ", pca.components_)

#Y = pca.transform(X)

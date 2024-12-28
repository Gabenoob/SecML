# 5.使用PCA对breast数据进行降维表示，并可视化

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

# 加载数据
data = load_breast_cancer()
X = data.data
y = data.target

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 可视化
plt.figure()
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='r', label='malignant')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='g', label='benign')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.legend()
plt.show()

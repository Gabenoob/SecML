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
pca = PCA(n_components=2) # 设置PCA降成2维
X_pca = pca.fit_transform(X) # 训练并降维

# 可视化
plt.figure()
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='g', label='benign')# 画出良性肿瘤的点
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='r', label='malignant') # 画出恶性肿瘤的点
plt.xlabel('First principal component') 
plt.ylabel('Second principal component')
plt.legend()
plt.show()

# 降成3维
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], X_pca[y == 0, 2], c='r', label='malignant')
ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], X_pca[y == 1, 2], c='g', label='benign')
ax.set_xlabel('First principal component')
ax.set_ylabel('Second principal component')
ax.set_zlabel('Third principal component')
plt.legend()
plt.show()

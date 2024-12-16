################################
#因为每几个月SKLearn、Pandas库都会更新，如发生错误时，请先查阅最新的sklearn官方文档
##################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")   # 忽略警告

'''
    sklearn中有多种非线性SVM方法，具体取决于kernel
    kernel = ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    iris数据集：四个植物特征和一个植物类别的标签，植物类别共有三类
'''

# 读取数据，并给数据每列命名
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
irisdata = pd.read_csv("./iris_data.csv", names=colnames)

# irisdata.info() 查看数据类型
X = irisdata.drop('Class', axis=1)  # 取出特征
y = irisdata['Class']   # 取出标签

# 随机分割训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)    

# 分别使用三种SVM方法训练数据
svc_classifier1 = SVC(kernel='rbf')
svc_classifier1.fit(X_train, y_train)

svc_classifier2 = SVC(kernel='sigmoid')
svc_classifier2.fit(X_train, y_train)

svc_classifier3 = SVC(kernel='poly', degree=8)
svc_classifier3.fit(X_train, y_train)

# 分别使用训练好的模型进行预测
y_pred1 = svc_classifier1.predict(X_test)

y_pred2 = svc_classifier2.predict(X_test)

y_pred3 = svc_classifier3.predict(X_test)

# 评估模型预测结果并输出
print('kernel = ''rbf'':')
print('混淆矩阵：\n', confusion_matrix(y_test, y_pred1))
print('评估结果报告：\n', classification_report(y_test, y_pred1))

print('kernel = ''sigmoid'':')
print('混淆矩阵：\n', confusion_matrix(y_test, y_pred2))
print('评估结果报告：\n', classification_report(y_test, y_pred2))

print('kernel = ''poly'':')
print('混淆矩阵：\n', confusion_matrix(y_test, y_pred3))
print('评估结果报告：\n', classification_report(y_test, y_pred3))

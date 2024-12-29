import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris      # 数据集
from sklearn.tree import DecisionTreeClassifier     # 决策树模型
from sklearn.tree import export_graphviz    # 绘制树模型
from sklearn.utils import Bunch     # Bunch方便存储读取数据
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder 

if __name__ == "__main__":
    # 加载数据集
    dataset = Bunch()       # 定义dataset数据类型
    df = pd.read_csv("./xiguadata.csv", encoding="utf-8")      # 读取数据
    data = df.values        # 将数据转换为数组
    # 生成特征编码
    oe = OrdinalEncoder(dtype=np.int)
    classes_list = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    #encode the feature to integer
    le = LabelEncoder()
    tar = le.fit_transform(data[:, -1])
    data = oe.fit_transform(data[:, :-1])


    dataset = Bunch(
            data=data,      # 分开特征与标签
            target=tar,
            feature_names=["{}".format(i) for i in classes_list],       # 特征名字
            target_names=["否","是"],        # 标签名字
        )

    X, y = dataset.data, dataset.target     # 令Bunch中的data, target分别为X, y


    # 创建模型并训练
    # max_depth=1 令树的最大深度为1，可自行修改，比较异同
    clf = DecisionTreeClassifier(criterion="gini", max_depth=5)
    clf.fit(X, y)

    # 使用模型预测
    dataTest = pd.DataFrame(['青绿','蜷缩','沉闷','清晰','凹陷','硬滑'], index=classes_list).T
    dataTest = dataTest.values
    dataTest = oe.transform(dataTest)
    test_example = dataTest     # 测试样例
    pred = clf.predict(test_example)[0]       # 预测，并将结果存储于pred中
    print("test_example: {}".format(test_example))
    print("Prediction: " + dataset.target_names[pred])

    # export_graphviz() 可视化决策树
    export_graphviz(
        clf,
        out_file="tree.dot",
        feature_names=dataset.feature_names,
        class_names=dataset.target_names,
        rounded=True,
        filled=True,
    )
    print("Done. Open dot file with Pycharm to view.")

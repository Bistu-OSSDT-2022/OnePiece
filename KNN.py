import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier as KNN
import joblib
import matplotlib.pyplot as plt             # 用于绘制图表
from written.dataload import *


# 模型训练（对不同K值的拟合效果做出测定，并选出最合适的K值）
knn_acc = []
m=0             #数据记录
for i in range(1, 40):
    neigh = KNN(n_neighbors=i, algorithm='auto', weights='distance')   # k-邻近算法
    # n_neighbors:  邻居的数量
    # weights:      更近的邻居对于所预测的点的影响更大
    # algorithm:    自动选择最合适的算法

    neigh.fit(train_images, train_labels)                   # train_images训练数据，train_labels目标值
    y_pred = neigh.predict(test_images)                     # 给test_images预测相应的类别标签
    print(i, '\t', metrics.adjusted_rand_score(test_labels, y_pred))         # 随机兰德调整指数
    if metrics.adjusted_rand_score(test_labels, y_pred)>m:
        m=metrics.adjusted_rand_score(test_labels, y_pred)
        model_path = './neigh_model'
        joblib.dump(neigh, model_path)                      # 模型持久化
    knn_acc.append(metrics.accuracy_score(test_labels, y_pred))              # 所有分类正确的百分比

print(knn_acc)

#使用直方图表展示不同K值训练处的模型的评估效果
def plot_graphs_knn(knn_acc):
    plt.bar(list(range(1,40)), knn_acc)
    plt.xticks(np.arange(1, 40, 2))           # x坐标取值范围
    plt.ylim(0.6, 1.00)                         # y坐标取值范围
    plt.xlabel('K')
    plt.ylabel('acc')
    plt.show()
plot_graphs_knn(knn_acc)

#进行手写汉字识别
print('原数据标签：\n'+str(test_labels[:20])+"\n预测数据标签：")     # 随机取20个样本进行识别
neigh_model = joblib.load('neigh_model')

print(list(neigh_model.predict(test_images[:20])))              # 打印识别结果

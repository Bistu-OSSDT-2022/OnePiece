import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
import joblib
import matplotlib.pyplot as plt
# 获取所有数据
from written.dataload import dataload


DATA = dataload()

# 数据分类
train_images = DATA['train']
test_images = DATA['test']
train_labels = DATA['trainLabels']
test_labels = DATA['testLabels']
kmeans = KMeans(n_clusters=7, n_init=5, init='random', tol=0.00001)
    # n_clusters: 7个字对应生成七个聚类数
    # n_init： 使用不同的质心种子运行的时间
    # init: 从初始质心的数据中随机选择观测值
    # tol: 可容忍的惯性变化，调低以扫描整个空间

kmeans.fit(train_images)        # 对数据进行聚类

def find_label(kmeans):

    tdict = {'jia': 0, 'mu': 1, 'ri': 2, 'shen': 3, 'tian': 4, 'you': 5, 'yue': 6}
    t0 = []
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    t5 = []
    t6 = []
    for i in range(7):
        t0.append(0)
        t1.append(0)
        t2.append(0)
        t3.append(0)
        t4.append(0)
        t5.append(0)
        t6.append(0)

    for i in range(2800):
        if kmeans.labels_[i] == 0:              # 使用.labels_ 查看聚好的类别
            t0[tdict[train_labels[i]]] += 1
        elif kmeans.labels_[i] == 1:
            t1[tdict[train_labels[i]]] += 1
        elif kmeans.labels_[i] == 2:
            t2[tdict[train_labels[i]]] += 1
        elif kmeans.labels_[i] == 3:
            t3[tdict[train_labels[i]]] += 1
        elif kmeans.labels_[i] == 4:
            t4[tdict[train_labels[i]]] += 1
        elif kmeans.labels_[i] == 5:
            t5[tdict[train_labels[i]]] += 1
        elif kmeans.labels_[i] == 6:
            t6[tdict[train_labels[i]]] += 1
        else :
            pass
  # 构造转换

    return [np.argmax(t0), np.argmax(t1), np.argmax(t2),np.argmax(t3),np.argmax(t4),np.argmax(t5),np.argmax(t6)]
# 合成代码
# 随机初始质心位置，或修正收敛条件

k_means_acc = []
m=0             #数据记录
user_name = ['jia', 'mu', 'ri', 'shen', 'tian', 'you', 'yue']
tdict = {'jia': 0, 'mu': 1, 'ri': 2, 'shen': 3, 'tian': 4, 'you': 5, 'yue': 6}
for i in range(100):
    # 模型训练
    kmeans = KMeans(n_clusters=7, n_init=5, init='random', tol=0.001)
    kmeans.fit(train_images)
    # 构造转换
    k_means_list = find_label(kmeans)
    print(k_means_list)
    # 模型预测
    y_pred = kmeans.predict(test_images)
    # 转换到对应标签上
    for j in range(700):
        test_labels[j] = tdict[test_labels[j]]
        y_pred[j] = k_means_list[y_pred[j]]
    #评估
    print(i, '\t', metrics.adjusted_rand_score(test_labels, y_pred))
    if metrics.adjusted_rand_score(test_labels, y_pred) > m:
        m = metrics.adjusted_rand_score(test_labels, y_pred)
        model_path = './kmeans_model'
        joblib.dump(kmeans, model_path)                      # 模型持久化

    k_means_acc.append(metrics.accuracy_score(test_labels, y_pred))
    # 回滚test
    for i in range(700):
        test_labels[i] = user_name[test_labels[i]]
        # test_images[i] = user_name[test_images[i]]



def plot_graphs_kmeans(k_means_acc):
    plt.plot(k_means_acc)
    plt.xticks(np.arange(1, 100, 1))
    plt.xlabel('Random_times')
    plt.ylabel('acc')
    plt.show()
plot_graphs_kmeans(k_means_acc)
# 制图

#进行手写汉字识别
print('原数据标签：\n'+str(test_labels[:20])+"\n预测数据标签：")     # 随机取20个样本进行识别
kmeans_model = joblib.load('kmeans_model')                        # 模型从本地调回
# print(kmeans_model.predict(test_images[:20]))
# print(user_name[list(kmeans_model.predict(test_images[:20]))[1])
print(list(kmeans_model.predict(test_images[:20])))
for i in range(20):
    print(user_name[list(kmeans_model.predict(test_images[:20]))[i]], end=', ')         # 打印识别结果


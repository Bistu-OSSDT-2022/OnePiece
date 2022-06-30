import os
import numpy as np
from PIL import Image
import random
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier as KNN
import shutil
import joblib
import matplotlib.pyplot as plt

mylist = []
path = 'data_set'
mydict = {}
for file in os.listdir(path):
    name = file.split('_')[0]
    if not mydict.get(name):
        mydict.update({name:1})
        mylist.append(name)
print("标签:"+str(mylist))

user_name = ['jia','mu','ri','shen','tian','you','yue']
path = 'data_set/'
newdict = {}
for file in os.listdir(path):
    name = file.split('_')[0]
    if name in user_name:
        im = Image.open(path+file)
        if im.size[0] == 100 and im.size[1] == 100:
            if not newdict.get(name):
                newdict.update({name:1})
                print(name)
            else:
                newdict.update({name:newdict.get(name)+1})
for name in user_name:
    print(name + ' has:' + str(newdict.get(name)))

path = 'data_set/'
user_name = ['jia','mu','ri','shen','tian','you','yue']

def img2vector(im):
    im = im.convert("RGB")
    img = np.array(im).flatten()
    return img/255

def dataload():
    newdict = {}
    trainData = np.zeros((80*7,30000))
    testData = np.zeros((20*7,30000))
    trainLabels = []
    testLabels = []
    train_num = 0
    test_num = 0
    filepath = os.listdir(path)
    random.shuffle(filepath)
    for file in filepath:
        name = file.split('_')[0]
        if name in user_name:
            im = Image.open(path+file)
            if im.size[0] == 100 and im.size[1] == 100:
                if not newdict.get(name):
                    newdict.update({name:1})
                else:
                    newdict.update({name:newdict.get(name)+1})
                if newdict.get(name) <= 80:
                    trainData[train_num,:] = img2vector(im)
                    train_num +=1
                    trainLabels.append(name)
                elif newdict.get(name) <= 100:
                    testData[test_num,:] = img2vector(im)
                    test_num += 1
                    testLabels.append(name)
                else:
                    pass
    return {'train':trainData,'trainLabels':trainLabels,'test':testData,'testLabels':testLabels}

DATA = dataload()
train_images = DATA['train']
test_images = DATA['test']
train_labels = DATA['trainLabels']
test_labels = DATA['testLabels']
knn_acc = []
m = 0
for i in range(1,20):
    neigh = KNN(n_neighbors=i,algorithm='auto',weights='distance')
    neigh.fit(train_images,train_labels)
    y_pred = neigh.predict(test_images)
    print(metrics.adjusted_rand_score(test_labels,y_pred))
    if metrics.adjusted_rand_score(test_labels,y_pred) > m:
        m = metrics.adjusted_rand_score(test_labels,y_pred)
        model_path = "./neigh_model"
        joblib.dump(neigh,model_path)
    knn_acc.append(metrics.accuracy_score(test_labels,y_pred))

def plot_graphs_knn(knn_acc):
    plt.bar(list(range(1,20)),knn_acc)
    plt.xticks(np.arange(1,20,2))
    plt.ylim(0.5,1.00)
    plt.xlabel('k')
    plt.ylabel('acc')
    plt.show()
plot_graphs_knn(knn_acc)
print("原数据标签："+ str(test_labels[:13])+'\n预测数据标签：'+str(train_labels[:13]))
neigh_model = joblib.load('neigh_model')
neigh_model.predict(test_images[1:13])
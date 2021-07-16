import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
import skimage.io as io
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
import os
import glob

path = 'E:/sklearn Datasets/data/data_1/'

images = []
labels = []

for file in os.listdir(path):
    tep = io.imread(os.path.join(path, file), as_gray=True).reshape(1,-1)
    images.append(tep)
    if ((file.split("_")[0]) == 'cloud'):
        labels.append(1)
    else:
        labels.append(0)
        
x = np.concatenate(images, axis=0)

print(labels[-1])
plt.imshow(x[-1].reshape(64,64))

x, y = shuffle(x,labels, random_state=10)

x_train, x_test, y_train, y_test = train_test_split(x,y , random_state=0)

clf = make_pipeline(StandardScaler(),
                   linear_model.Perceptron(tol = 1e-3, random_state=12))
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

acu = metrics.accuracy_score(y_test, y_pred)
acu

metrics.plot_confusion_matrix(clf, x_test, y_test, cmap=plt.cm.Blues)

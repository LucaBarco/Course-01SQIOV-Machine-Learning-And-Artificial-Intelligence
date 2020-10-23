

from itertools import product
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets,utils
from sklearn.model_selection import cross_val_score

n_neighbors=1
#load data
data = load_wine()
print(data.feature_names)
#select 2 features
dataSelection=data.data[:,[0,1]]
#print(dataSelection)
size=dataSelection.shape[0]
#split data

dataSelection,validation=utils.shuffle(dataSelection,data.target,random_state=10)
nTraining=int(size*0.5);
nTest=int(size*0.3);
nValidation=int(size*0.2);
#print("T{} T{} V{}".format(nTraining,nTest,nValidation))
trainingData=dataSelection[:nTraining]
testData=dataSelection[nTraining:nTraining+nTest]
validationData=dataSelection[nTraining+nTest:]

targetValues=validation[:nTraining]
np.append(trainingData,validationData)
np.append(targetValues,validation[nTraining:nTraining+nTest])
X=trainingData
y=targetValues
# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

for i in range (1,3):
    f, axarr = plt.subplots(4, 4, sharex='col', sharey='row', figsize=(10, 8))
    for idx,par in zip(product([0,1,2,3], [0,1,2,3]),product([0.001*(10**((i-1)*4)),0.01*(10**((i-1)*4)),0.1*(10**((i-1)*4)),1*(10**((i-1)*4))],[0.001*(10**((i-1)*4)),0.01*(10**((i-1)*4)),0.1*(10**((i-1)*4)),1*(10**((i-1)*4))])) :

        clf = SVC(C=par[0], kernel='linear', gamma=par[1], probability=True)
        clf.fit(trainingData, validation[:nTraining])

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
        axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                      s=20, edgecolor='k')
        axarr[idx[0], idx[1]].set_title("Kernel SVM C={} Gamma={}".format(par[0],par[1]))
        Z_V = clf.predict(validationData)
        countV = 0
        for i in range(0, Z_V.size):
           # print ("Z:{} T:{}".format(Z_V[i], validation[i + nTraining + nTest]))
            if (Z_V[i] == validation[i + nTraining + nTest]):
                countV = countV + 1
        accuracy = countV / nValidation
        print("Accuracy of model ", "Kernel SVM C={} Gamma={}".format(par[0],par[1]), " : ", accuracy)
        scores = cross_val_score(clf, X, y, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    plt.show()

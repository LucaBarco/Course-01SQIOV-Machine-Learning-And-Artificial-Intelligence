'''WHAT YOU SHOULD DO:
1. Keep the same data you used before (same features, same split)
2. Repeat the same steps you did before, this time use a SVM with an RBF kernel:
    a. for this first step, keep gamma fixed to its default value, vary only the C parameter (choose the values you
    think are the most suitable)
    b. are the decision boundaries different? why?
3. Perform a grid search over both gamma and C at the same time:
    a. for each of them, select an appropriate range
    b. plot decision boundaries
    c. choose the best parameter according to the performances on the evaluation set
    d. evaluate the model on the test set
4. Inspect the performance scores and the decision boundaries: what is the effect of gamma?
5. Does this model perform better than the previous one? Why?'''


from itertools import product
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets,utils

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
#print(validation)
'''
clf1 = SVC(C=0.001, kernel='rbf', probability=True, decision_function_shape='ovr')
clf1.fit(trainingData, validation[:nTraining])

clf2 = SVC(C= 0.01,  kernel='rbf', probability=True, decision_function_shape='ovr')
clf2.fit(trainingData, validation[:nTraining])

clf3 = SVC(C=0.1, kernel='rbf', probability=True, decision_function_shape='ovr')
clf3.fit(trainingData, validation[:nTraining])

clf4 = SVC(C= 1, kernel='rbf', probability=True, decision_function_shape='ovr')
clf4.fit(trainingData, validation[:nTraining])

clf5 = SVC(C= 10,  kernel='rbf', probability=True, decision_function_shape='ovr')
clf5.fit(trainingData, validation[:nTraining])

clf6 = SVC(C=100, kernel='rbf', probability=True, decision_function_shape='ovr')
clf6.fit(trainingData, validation[:nTraining])

clf7 = SVC(C=1000, kernel='rbf', probability=True, decision_function_shape='ovr')
clf7.fit(trainingData, validation[:nTraining])

X=trainingData
y=validation[:nTraining]
# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0,1,2, 3], [0, 1]),
                        [clf1,clf2,clf3,clf4,clf5,clf6,clf7],
                        ['Kernel SVM rbf ovr gamma std with C=0.001',
                         'Kernel SVM rbf ovr gamma std with C=0.01',
                         'Kernel SVM rbf ovr gamma std with C=0.1',
                         'Kernel SVM rbf ovr gamma std with C=1',
                         'Kernel SVM rbf ovr gamma std with C=10',
                         'Kernel SVM rbf ovr gamma std with C=100',
                         'Kernel SVM rbf ovr gamma std with C=1000']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)
    Z_V = clf.predict(validationData)
    countV = 0
    for i in range(0, Z_V.size):
       # print("Z:{} T:{}".format(Z_V[i], validation[i + nTraining + nTest]))
        if (Z_V[i] == validation[i + nTraining + nTest]):
            countV = countV + 1
    accuracy = countV / nValidation
    print("Accuracy of model ",tt," : ",accuracy)
plt.show()

clf1 = SVC(C=0.001, kernel='linear', probability=True, decision_function_shape='ovo')
clf1.fit(trainingData, validation[:nTraining])

clf2 = SVC(C= 0.01,  kernel='linear', probability=True, decision_function_shape='ovo')
clf2.fit(trainingData, validation[:nTraining])

clf3 = SVC(C=0.1, kernel='linear', probability=True, decision_function_shape='ovo')
clf3.fit(trainingData, validation[:nTraining])

clf4 = SVC(C= 1, kernel='linear', probability=True, decision_function_shape='ovo')
clf4.fit(trainingData, validation[:nTraining])

clf5 = SVC(C= 10,  kernel='linear', probability=True, decision_function_shape='ovo')
clf5.fit(trainingData, validation[:nTraining])

clf6 = SVC(C=100, kernel='linear', probability=True, decision_function_shape='ovo')
clf6.fit(trainingData, validation[:nTraining])

clf7 = SVC(C=1000, kernel='linear', probability=True, decision_function_shape='ovo')
clf7.fit(trainingData, validation[:nTraining])

X=trainingData
y=validation[:nTraining]
# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0,1,2, 3], [0, 1]),
                        [clf1,clf2,clf3,clf4,clf5,clf6,clf7],
                        ['Kernel SVM rbf ovo gamma std with C=0.001',
                         'Kernel SVM rbf ovo gamma std with C=0.01',
                         'Kernel SVM rbf ovo gamma std with C=0.1',
                         'Kernel SVM rbf ovo gamma std with C=1',
                         'Kernel SVM rbf ovo gamma std with C=10',
                         'Kernel SVM rbf ovo gamma std with C=100',
                         'Kernel SVM rbf ovo gamma std with C=1000']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)
    Z_V = clf.predict(validationData)
    countV = 0
    for i in range(0, Z_V.size):
        #print("Z:{} T:{}".format(Z_V[i], validation[i + nTraining + nTest]))
        if (Z_V[i] == validation[i + nTraining + nTest]):
            countV = countV + 1
    accuracy = countV / nValidation
    print("Accuracy of model ", tt, " : ", accuracy)
plt.show()


clf1 = SVC(C=0.001, kernel='rbf', gamma='auto', probability=True, decision_function_shape='ovr')
clf1.fit(trainingData, validation[:nTraining])

clf2 = SVC(C= 0.01,  kernel='rbf', gamma='auto', probability=True, decision_function_shape='ovr')
clf2.fit(trainingData, validation[:nTraining])

clf3 = SVC(C=0.1, kernel='rbf', gamma='auto', probability=True, decision_function_shape='ovr')
clf3.fit(trainingData, validation[:nTraining])

clf4 = SVC(C= 1, kernel='rbf',  gamma='auto',probability=True, decision_function_shape='ovr')
clf4.fit(trainingData, validation[:nTraining])

clf5 = SVC(C= 10,  kernel='rbf', gamma='auto', probability=True, decision_function_shape='ovr')
clf5.fit(trainingData, validation[:nTraining])

clf6 = SVC(C=100, kernel='rbf',  gamma='auto',probability=True, decision_function_shape='ovr')
clf6.fit(trainingData, validation[:nTraining])

clf7 = SVC(C=1000, kernel='rbf',  gamma='auto',probability=True, decision_function_shape='ovr')
clf7.fit(trainingData, validation[:nTraining])

X=trainingData
y=validation[:nTraining]
# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0,1,2, 3], [0, 1]),
                        [clf1,clf2,clf3,clf4,clf5,clf6,clf7],
                        ['Kernel SVM rbf ovr gamma auto with C=0.001',
                         'Kernel SVM rbf ovr gamma auto with C=0.01',
                         'Kernel SVM rbf ovr gamma auto with C=0.1',
                         'Kernel SVM rbf ovr gamma auto with C=1',
                         'Kernel SVM rbf ovr gamma auto with C=10',
                         'Kernel SVM rbf ovr gamma auto with C=100',
                         'Kernel SVM rbf ovr gamma auto with C=1000']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)
    Z_V = clf.predict(validationData)
    countV = 0
    for i in range(0, Z_V.size):
        #("Z:{} T:{}".format(Z_V[i], validation[i + nTraining + nTest]))
        if (Z_V[i] == validation[i + nTraining + nTest]):
            countV = countV + 1
    accuracy = countV / nValidation
    print("Accuracy of model ", tt, " : ", accuracy)
plt.show()

clf1 = SVC(C=0.001, kernel='linear', gamma='auto', probability=True, decision_function_shape='ovo')
clf1.fit(trainingData, validation[:nTraining])

clf2 = SVC(C= 0.01,  kernel='linear', gamma='auto', probability=True, decision_function_shape='ovo')
clf2.fit(trainingData, validation[:nTraining])

clf3 = SVC(C=0.1, kernel='linear', gamma='auto', probability=True, decision_function_shape='ovo')
clf3.fit(trainingData, validation[:nTraining])

clf4 = SVC(C= 1, kernel='linear', gamma='auto', probability=True, decision_function_shape='ovo')
clf4.fit(trainingData, validation[:nTraining])

clf5 = SVC(C= 10,  kernel='linear', gamma='auto', probability=True, decision_function_shape='ovo')
clf5.fit(trainingData, validation[:nTraining])

clf6 = SVC(C=100, kernel='linear', gamma='auto', probability=True, decision_function_shape='ovo')
clf6.fit(trainingData, validation[:nTraining])

clf7 = SVC(C=1000, kernel='linear', gamma='auto', probability=True, decision_function_shape='ovo')
clf7.fit(trainingData, validation[:nTraining])

X=trainingData
y=validation[:nTraining]
# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0,1,2, 3], [0, 1]),
                        [clf1,clf2,clf3,clf4,clf5,clf6,clf7],
                        ['Kernel SVM rbf ovo gamma auto with C=0.001',
                         'Kernel SVM rbf ovo gamma auto with C=0.01',
                         'Kernel SVM rbf ovo gamma auto with C=0.1',
                         'Kernel SVM rbf ovo gamma auto with C=1',
                         'Kernel SVM rbf ovo gamma auto with C=10',
                         'Kernel SVM rbf ovo gamma auto with C=100',
                         'Kernel SVM rbf ovo gamma auto with C=1000']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)
    Z_V = clf.predict(validationData)
    countV = 0
    for i in range(0, Z_V.size):
       # print ("Z:{} T:{}".format(Z_V[i], validation[i + nTraining + nTest]))
        if (Z_V[i] == validation[i + nTraining + nTest]):
            countV = countV + 1
    accuracy = countV / nValidation
    print("Accuracy of model ", tt, " : ", accuracy)
plt.show()

#choose it clf7 = SVC(C=1000, kernel='rbf',  gamma='auto',probability=True, decision_function_shape='ovr')

clf7 = SVC(C=1000, kernel='rbf',  gamma='auto',probability=True, decision_function_shape='ovr')
clf7.fit(trainingData, validation[:nTraining])

Z_V=clf7.predict(validationData)
countV = 0
for i in range(0,Z_V.size):
   # print("Z:{} T:{}".format(Z_V[i],validation[i+nTraining+nTest]))
    if(Z_V[i]==validation[i+nTraining+nTest]):
        countV=countV+1

print(countV)
accuracy=countV/nValidation
print(accuracy)
'''
#basing on accuracy the best is rbf ovr gamma std C=100

#Using float numbers:


X=trainingData
y=validation[:nTraining]
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
    plt.show()

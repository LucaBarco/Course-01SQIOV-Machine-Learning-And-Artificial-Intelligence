'''WHAT YOU SHOULD DO:
1. Keep the same data you used before (same features, same split)
2. Repeat the same steps you did before, this time varying the penalty parameter C of the SVM with
linear kernel:
a. example values: C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
3. Carefully inspect the decision boundaries while varying C, keeping in mind the idea of soft-margin:
    a. how does the value of C affects the boundaries?
    b. what happens when C is very low? What about when it is very high?
4. Inspect the decision_function_shape parameter
a. what is its default value? Is it consistent to the results you have obtained?
b. Try also with the one-versus-one policy: what happens “behind the scenes”? Are the results different? Why?'''

'''decision_function_shape{‘ovo’, ‘ovr’}, default=’ovr’
Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) 
as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape 
(n_samples, n_classes * (n_classes - 1) / 2). 
However, one-vs-one (‘ovo’) is always used as multi-class strategy. 
The parameter is ignored for binary classification.



-> the results are the same because we have 3 classes
ovr -> nsaples x nclasses -> nsamples x 3
ovo -> nsamples x nclasses(nclasses-1)/2 -> nsamples x 3
'''

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
print(dataSelection)
size=dataSelection.shape[0]
#split data

dataSelection,validation=utils.shuffle(dataSelection,data.target,random_state=10)
nTraining=int(size*0.5);
nTest=int(size*0.3);
nValidation=int(size*0.2);
print("T{} T{} V{}".format(nTraining,nTest,nValidation))
trainingData=dataSelection[:nTraining]
testData=dataSelection[nTraining:nTraining+nTest]
validationData=dataSelection[nTraining+nTest:]
print(validation)

clf1 = SVC(C=0.001, kernel='linear', probability=True, decision_function_shape='ovr')
clf1.fit(trainingData, validation[:nTraining])

clf2 = SVC(C= 0.01,  kernel='linear', probability=True, decision_function_shape='ovr')
clf2.fit(trainingData, validation[:nTraining])

clf3 = SVC(C=0.1, kernel='linear', probability=True, decision_function_shape='ovr')
clf3.fit(trainingData, validation[:nTraining])

clf4 = SVC(C= 1, kernel='linear', probability=True, decision_function_shape='ovr')
clf4.fit(trainingData, validation[:nTraining])

clf5 = SVC(C= 10,  kernel='linear', probability=True, decision_function_shape='ovr')
clf5.fit(trainingData, validation[:nTraining])

clf6 = SVC(C=100, kernel='linear', probability=True, decision_function_shape='ovr')
clf6.fit(trainingData, validation[:nTraining])

clf7 = SVC(C=1000, kernel='linear', probability=True, decision_function_shape='ovr')
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
                        ['Kernel SVM with C=0.001',
                         'Kernel SVM with C=0.01',
                         'Kernel SVM with C=0.1',
                         'Kernel SVM with C=1',
                         'Kernel SVM with C=10',
                         'Kernel SVM with C=100',
                         'Kernel SVM with C=1000']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

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
                        ['Kernel SVM with C=0.001',
                         'Kernel SVM with C=0.01',
                         'Kernel SVM with C=0.1',
                         'Kernel SVM with C=1',
                         'Kernel SVM with C=10',
                         'Kernel SVM with C=100',
                         'Kernel SVM with C=1000']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()
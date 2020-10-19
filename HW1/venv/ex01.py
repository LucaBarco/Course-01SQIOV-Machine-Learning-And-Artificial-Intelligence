'''WHAT YOU SHOULD DO:
1. Load Wine dataset (scikit library)
2. Select ONLY 2 attributes (the first 2, for example, but feel free to try with different pairs)
        a. extra: understand, by looking at the distribution of the data in the chosen 2D, which classification method
        could have good performances and why.
3. Split into train, validation and test sets (suggested proportion 5:2:3)
4. For different values of K (example: [1,3,5,7]):
a. apply K-NN
b. plot data and decision boundaries
c. evaluate on validation set
5. Inspect the results:
a. plot a graph showing how the accuracy varies for different value of K
b. plot the boundaries for each value of K. How do they change and why?
6. Use the best value of K on the test set and evaluate the accuracy.'''

from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import numpy as np
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

#KNN

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])
# we create an instance of Neighbours Classifier and fit the data.
K=[1,2,3,4,5,6,7]
accuracy=[0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0]

for k in range(0,7):
    for weights in ['uniform', 'distance']:

        clf = neighbors.KNeighborsClassifier(K[k], weights=weights)
        clf.fit(trainingData, validation[:nTraining])

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = trainingData[:, 0].min() - 1, trainingData[:, 0].max() + 1
        y_min, y_max = trainingData[:, 1].min() - 1, trainingData[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(trainingData[:, 0], trainingData[:, 1], c=validation[:nTraining], cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (K[k], weights))

        Z_V=clf.predict(validationData)
        Z_T=clf.predict(testData)
        countV = 0
        for i in range(0,Z_V.size):
            print("Z:{} T:{}".format(Z_V[i],validation[i+nTraining+nTest]))
            if(Z_V[i]==validation[i+nTraining+nTest]):
                countV=countV+1

        print(countV)
        accuracy[k]=countV/nValidation
plt.show()

#validation
print([accuracy])
plt.plot(K,accuracy)
plt.show()
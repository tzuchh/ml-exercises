from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import datetime

from sklearn.metrics import accuracy_score

# Load the dataset
iris = datasets.load_iris()

# K-fold Cross Validation
kfold = 3
n_validation = 20
neighbors = list(range(1, n_validation+1, 1))
mse = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(iris.data, iris.target)
    scores = cross_val_score(knn, iris.data, iris.target, cv=kfold, scoring='accuracy')
    mse.append(1 - scores.mean())

plt.plot(neighbors, mse)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Misclassification Error")
plt.xticks(np.arange(0, n_validation+1, 1))
plt.grid()
plt.show()

optimal_k = neighbors[mse.index(min(mse))]
print("\nThe optimal K: {}".format(optimal_k))

# Prepare data
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target,
    test_size=1.0/kfold, shuffle = True, random_state = 42)
print("\nx_train shape: {}".format(x_train.shape))
print("x_test shape: {}".format(x_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))

# Training
print("\nTraining...")
start_time = datetime.datetime.now()
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(x_train, y_train)
end_time = datetime.datetime.now()
print('Start time: {} (UTC)'.format(start_time))
print('  End time: {} (UTC)'.format(end_time))

# Testing
print("\nAccuracy = {0}%".format(100*accuracy_score(knn.predict(x_test), y_test)))
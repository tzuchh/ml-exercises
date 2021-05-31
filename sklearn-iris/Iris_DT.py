from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Load the dataset
iris = datasets.load_iris()

# K-fold Cross Validation
kfold = 3
mse = []
dt = DecisionTreeClassifier(min_samples_split=2)
dt.fit(iris.data, iris.target)
scores = cross_val_score(dt, iris.data, iris.target, cv=kfold, scoring='accuracy')
mse = 1 - scores.mean()
print(scores)
print("\nMSE: {}".format(mse))
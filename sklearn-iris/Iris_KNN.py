from sklearn import datasets
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import datetime
from sklearn.model_selection import cross_val_score

# Load the dataset
iris = datasets.load_iris()

# Data exploration
df = pd.DataFrame(iris.data)
df.columns = [iris.feature_names[0], iris.feature_names[1], iris.feature_names[2], iris.feature_names[3]]
df['class'] = iris.target
sns.pairplot(df, hue='class', plot_kws=dict(marker="+", linewidth=1),
    palette=['blue', 'red' ,'green'], corner=True) 
plt.show()

# Prepare data
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target,
    test_size=0.2, shuffle = True, random_state = 0)
print("x_train shape: {}".format(x_train.shape))
print("x_test shape: {}".format(x_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))

# Training
print("Training...")
start_time = datetime.datetime.now()
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
end_time = datetime.datetime.now()
print('Start time: {} (UTC)'.format(start_time))
print('  End time: {} (UTC)'.format(end_time))

# Testing
print("Accuracy = {0}%".format(100*np.sum(knn.predict(x_test) == y_test)/len(y_test)))
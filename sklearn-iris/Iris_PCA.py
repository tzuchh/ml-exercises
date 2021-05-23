import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# %matplotlib inline

# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
url = "./data/iris.csv"

# loading dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length', 'sepal width',
                             'petal length', 'petal width', 'target'])
print(df.head())


# Standardize the Data
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = df.loc[:, features].values
y = df.loc[:, ['target']].values
x = StandardScaler().fit_transform(x)
print(pd.DataFrame(data=x, columns=features).head())


# PCA Projection to 2D
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=[
                           'principal component 1', 'principal component 2'])
# print(principalDf.head(5))
# print(df[['target']].head())
finalDf = pd.concat([principalDf, df[['target']]], axis=1)
print(finalDf.head(5))

# The explained variance tells us how much information (variance) can be attributed to each of the principal components.
# The first principal component contains 72.77% of the variance and the second principal component contains 23.03% of the variance.
# The third and fourth principal component contained the rest of the variance of the dataset.
print(pca.explained_variance_ratio_)


# Visualize 2D Projection
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
ax.legend(targets)
ax.grid()
plt.show()

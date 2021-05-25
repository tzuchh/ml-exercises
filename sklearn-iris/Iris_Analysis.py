from sklearn import datasets
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Load the dataset
iris = datasets.load_iris()

# Data exploration
df = pd.DataFrame(iris.data)
df.columns = [iris.feature_names[0], iris.feature_names[1], iris.feature_names[2], iris.feature_names[3]]
df['class'] = iris.target
sns.pairplot(df, hue='class', plot_kws=dict(marker="+", linewidth=1),
    palette=['blue', 'red' ,'green'], corner=True) 
plt.show()
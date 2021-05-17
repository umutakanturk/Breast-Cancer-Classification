import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("cancer.csv")
data.drop(['Unnamed: 32','id'], inplace = True, axis = 1)

data = data.rename(columns = {"diagnosis":"target"})

sns.countplot(data["target"])
print(data.target.value_counts())

data["target"] = [1 if i.strip() == "M" else 0 for i in data.target]

print(len(data))

print(data.head)

print("Data shape ", data.shape)

data.info()

describe = data.describe()

"""
standardization
missing value: none
"""

# %% EDA

# Correlation
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation Between Features")
plt.show

#
threshold = 0.50
filtre = np.abs(corr_matrix["target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation Between Features w Corr Threshold 0.75")

"""
there some correlated features
"""

#box plot
data_melted = pd.melt(data, id_vars="target",
                      var_name="features",
                      value_name="value")
plt.figure()
sns.boxplot(x="features", y="value", hue="target", data = data_melted)
plt.xticks(rotation=90)
plt.show()

"""
standardization
"""

#pair plot
sns.pairplot(data[corr_features],diag_kind="kde",markers="+", hue = "target")
plt.show()

"""
skewness
"""

# %% outlier
y = data.target
x = data.drop(["target"], axis = 1)
columns = x.columns.tolist()

clf = LocalOutlierFactor()
y_pred = clf.fit_predict(x)
X_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()
outlier_score["score"] = X_score

#threshold
threshold = -2.5
filtre = outlier_score["score"] < threshold
outlier_index = outlier_score[filtre].index.tolist()


plt.figure()
plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1], s=50, label = "Outliers")
plt.scatter(x.iloc[:,0], x.iloc[:,1], color = "k", s=3,label = "Data Points" )

radius = (X_score.max() - X_score)/(X_score.max() - X_score.min())
outlier_score["radius"] = radius
plt.scatter(x.iloc[:,0],x.iloc[:,1], s=1000*radius,edgecolors="r", facecolor ="none", label="Outlier Scores")
plt.legend()
plt.show()

#drop outliers
x = x.drop(outlier_index)
y = y.drop(outlier_index).values

# %% Train test split
test_size = 0.3
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size = test_size,random_state = 42)

# %%

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.ransform(X_test)


































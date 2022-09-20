# Link: https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b


#importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#import statsmodels.api as sm
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
#Loading the dataset
#x = load_boston()
#df = pd.DataFrame(x.data, columns = x.feature_names)
#df["MEDV"] = x.target
#X = df.drop("MEDV",1)   #Feature Matrix
#y = df["MEDV"]          #Target Variable
#df.head()

# read csv input file
input_data = pd.read_csv("All_Participents_DataFrame.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_DataFrame_Filtered_BandPower.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_Mean_DataFrame.csv", sep=";", decimal=',')

#Using Pearson Correlation
plt.figure(figsize=(12,10))
correlation = input_data.corr()
sns.heatmap(correlation, annot=True)#, cmap=plt.cm.Reds)
plt.show()

#Correlation with output variable
cor_target = abs(correlation["pId"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
print(relevant_features)

# highly correlated with each other -> keep only one variable and drop the other
#print(input_data[["HeartRate","int"]].corr())
#print(input_data[["str","HeartRate"]].corr())


f = plt.figure(figsize=(28, 32))
plt.matshow(input_data.corr(), fignum=f.number)
plt.xticks(range(input_data.select_dtypes(['number']).shape[1]), input_data.select_dtypes(['number']).columns, fontsize=8, rotation=45)
plt.yticks(range(input_data.select_dtypes(['number']).shape[1]), input_data.select_dtypes(['number']).columns, fontsize=8)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=8)
plt.title('Correlation Matrix', fontsize=16);
plt.show()

from pandas.plotting import radviz
plt.figure();
radviz(input_data, "pId");
plt.show()

from pandas.plotting import scatter_matrix
scatter_matrix(input_data, alpha=0.2, figsize=(6, 6), diagonal="kde");
plt.show()
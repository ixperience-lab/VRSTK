# Source-Link: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#------ load dataset into Pandas DataFrame
#input_data = pd.read_csv("All_Participents_DataFrame.csv", sep=";", decimal=',')
input_data = pd.read_csv("All_Participents_WaveSum_DataFrame.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_DataFrame_Filtered_BandPower.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_DataFrame_Filtered_PerformanceMetric.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_Mean_DataFrame.csv", sep=";", decimal=',')

#colnames = list (input_data.columns)
#input_data.reset_index().plot(x="index", y=colnames[1:], kind = 'line', legend=False, 
#                 subplots = True, sharex = True, figsize = (5.5,4), ls="none", marker="o")
#plt.show()


#------ Normalizing
#features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = input_data.loc[:, :].values
#print(x) # Debug only
# Separating out the target
y = input_data.loc[:,['pId']].values
#print(y) # Debug only
# Standardizing the features
x = StandardScaler().fit_transform(x)
#print(x) # Debug only

#------ Principal Component Analysis n_components=2
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(x)
print(pca.explained_variance_ratio_)  # Debug only
#print(principalComponents) # Debug only
principalDataFrame = pd.DataFrame(data = principalComponents)#, columns = ['principal component 1', 'principal component 2'])#, 'principal component 3', 'principal component 4'])
#print(principalDataFrame) # Debug only

colnames = list (principalDataFrame.columns)
principalDataFrame.reset_index().plot(x="index", y=colnames[0:], kind = 'line', legend=False, 
                 subplots = True, sharex = True, figsize = (5.5,4), ls="none", marker="o")

plt.show()

#------ Principal Component Analysis n_components=1
""" pca = PCA(n_components=1)
principalComponents = pca.fit_transform(x)
print(principalComponents) # Debug only
principalDataFrame = pd.DataFrame(data = principalComponents, columns = ['principal component 1'])
#print(principalDataFrame) # Debug only """

resultDataFrame = pd.concat([principalDataFrame, input_data[['pId']]], axis = 1)
print(resultDataFrame) # Debug only

""" fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['1', '2', '13']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = resultDataFrame['pId'] == target
    ax.scatter(resultDataFrame.loc[indicesToKeep, 'principal component 1'], resultDataFrame.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
ax.legend(targets)
ax.grid() """

#ax2 = resultDataFrame.plot.scatter(x='principal component 1', y='principal component 2', c='pId', colormap='viridis')
# show the plot
#plt.show()

#copyDataFrame = resultDataFrame
#copyDataFrame['Y_Val'] = np.zeros_like(resultDataFrame['principal component 1'])
#ax2 = copyDataFrame.plot.scatter(x='principal component 1', y='Y_Val' , c='pId', colormap='viridis')
#plt.show()

#ax2 = copyDataFrame.plot.scatter(x='principal component 2', y='Y_Val' , c='pId', colormap='viridis')
#plt.show()


gaussianDataFrame = resultDataFrame

# GaussianMixture
# define the model
model = GaussianMixture(n_components=2)

gaussianDataFrame["Cluster"] = model.fit_predict(gaussianDataFrame)
gaussianDataFrame["Cluster"] = gaussianDataFrame["Cluster"].astype("int")
print(gaussianDataFrame.head()) 

ax2 = gaussianDataFrame.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# show the plot
plt.show()

kMeansDataFrame = resultDataFrame

# define the model
model = MiniBatchKMeans(n_clusters=2)

kMeansDataFrame["Cluster"] = model.fit_predict(kMeansDataFrame)
kMeansDataFrame["Cluster"] = kMeansDataFrame["Cluster"].astype("int")

ax2 = kMeansDataFrame.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# show the plot
plt.show()
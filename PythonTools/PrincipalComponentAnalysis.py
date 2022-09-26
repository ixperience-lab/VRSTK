# Source-Link: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#------ load dataset into Pandas DataFrame
#input_data = pd.read_csv("All_Participents_Stage1_DataFrame.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_WaveSum_DataFrame.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_WaveSum_Mean_DataFrame.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_DataFrame_Filtered_BandPower.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_DataFrame_Filtered_PerformanceMetric.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_Mean_DataFrame.csv", sep=";", decimal=',')
#input_data = pd.read_csv("All_Participents_WaveSum_Mean_DataFrame.csv", sep=";", decimal=',')
input_data = pd.read_csv("All_Participents_Mean_Diff_Of_Stages_DataFrame.csv", sep=";", decimal=',')

#------ Normalizing
# Separating out the features
x = input_data.loc[:, :].values
# Separating out the target
y = input_data.loc[:,['pId']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

#------ Principal Component Analysis n_components=2
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(x)
print(pca.explained_variance_ratio_)  # Debug only
#print(principalComponents) # Debug only
principalDataFrame = pd.DataFrame(data = principalComponents)#, columns = ['principal component 1', 'principal component 2'])#, 'principal component 3', 'principal component 4'])
#print(principalDataFrame) # Debug only
#------ correlation matrix
f = plt.figure(figsize=(28, 32))
plt.matshow(principalDataFrame.corr(), fignum=f.number)
plt.xticks(range(principalDataFrame.select_dtypes(['number']).shape[1]), principalDataFrame.select_dtypes(['number']).columns, fontsize=8, rotation=45)
plt.yticks(range(principalDataFrame.select_dtypes(['number']).shape[1]), principalDataFrame.select_dtypes(['number']).columns, fontsize=8)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=8)
plt.title('Correlation Matrix', fontsize=16);
plt.show()

#colnames = list (principalDataFrame.columns)
#principalDataFrame.reset_index().plot(x="index", y=colnames[0:], kind = 'line', legend=False, 
#                 subplots = True, sharex = True, figsize = (5.5,4), ls="none", marker="o")

#plt.show()

resultDataFrame = pd.concat([principalDataFrame, input_data[['pId']]], axis = 1)
#print(resultDataFrame) # Debug only

#ax2 = resultDataFrame.plot.scatter(x='principal component 1', y='principal component 2', c='pId', colormap='viridis')
# show the plot
#plt.show()

gaussianDataFrame = input_data

# GaussianMixture
# define the model
#model = GaussianMixture(n_components=2)

#gaussianDataFrame["Cluster"] = model.fit_predict(gaussianDataFrame)
#gaussianDataFrame["Cluster"] = gaussianDataFrame["Cluster"].astype("int")
#print(gaussianDataFrame.head()) 

#print("=================================================== gaussianDataFrame normal plot")

#ax2 = gaussianDataFrame.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# show the plot
#plt.show()

#--------------------

#_Ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,34]
#for id in _Ids:
#   temp = gaussianDataFrame.loc[gaussianDataFrame["pId"] == id]
#    first =  temp[temp.Cluster == 0].shape[0]
#    second =  temp[temp.Cluster == 1].shape[0]
#    print(first)
#    print(second)
#    print ("test")
#    if first > second: 
#        gaussianDataFrame.Cluster[gaussianDataFrame.pId == id] = 0
#    if first < second: 
#        gaussianDataFrame.Cluster[gaussianDataFrame.pId == id] = 1

#print("=================================================== gaussianDataFrame ids filter plot")

#ax2 = gaussianDataFrame.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# show the plot
#plt.show()

#--------------------

gaussianDataFrame = resultDataFrame

# GaussianMixture
# define the model
model = GaussianMixture(n_components=2)

gaussianDataFrame["Cluster"] = model.fit_predict(gaussianDataFrame)
gaussianDataFrame["Cluster"] = gaussianDataFrame["Cluster"].astype("int")
#print(gaussianDataFrame.head()) 

print("=================================================== gaussianDataFrame principal component plot")

ax2 = gaussianDataFrame.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# show the plot
plt.show()

_Ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,34]
for id in _Ids:
    temp = gaussianDataFrame.loc[gaussianDataFrame["pId"] == id]
    first =  temp[temp.Cluster == 0].shape[0]
    second =  temp[temp.Cluster == 1].shape[0]
    print(first)
    print(second)
    print ("test")
    if first > second: 
        gaussianDataFrame.Cluster[gaussianDataFrame.pId == id] = 0
    if first < second: 
        gaussianDataFrame.Cluster[gaussianDataFrame.pId == id] = 1

print("=================================================== gaussianDataFrame principal component ids filter plot")

ax2 = gaussianDataFrame.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# show the plot
plt.show()


#======================================================= K-MEANS

kMeansDataFrame = input_data

# define the model
model = MiniBatchKMeans(n_clusters=2)

kMeansDataFrame["Cluster"] = model.fit_predict(kMeansDataFrame)
kMeansDataFrame["Cluster"] = kMeansDataFrame["Cluster"].astype("int")

print("=================================================== kMeansDataFrame normal plot")

ax2 = kMeansDataFrame.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# show the plot
plt.show()

#--------------------

_Ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,34]
for id in _Ids:
    temp = kMeansDataFrame.loc[kMeansDataFrame["pId"] == id]
    first =  temp[temp.Cluster == 0].shape[0]
    second =  temp[temp.Cluster == 1].shape[0]
    print(first)
    print(second)
    print ("test")
    if first > second: 
        kMeansDataFrame.Cluster[kMeansDataFrame.pId == id] = 0
    if first < second: 
        kMeansDataFrame.Cluster[kMeansDataFrame.pId == id] = 1

print("=================================================== kMeansDataFrame ids filter plot")

ax2 = kMeansDataFrame.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# show the plot
plt.show()

#--------------------

kMeansDataFrame = resultDataFrame

# define the model
model = MiniBatchKMeans(n_clusters=2)

kMeansDataFrame["Cluster"] = model.fit_predict(kMeansDataFrame)
kMeansDataFrame["Cluster"] = kMeansDataFrame["Cluster"].astype("int")

print("=================================================== kMeansDataFrame principal component plot")

ax2 = kMeansDataFrame.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# show the plot
plt.show()
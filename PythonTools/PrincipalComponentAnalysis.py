# Source-Link: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
import numpy.matlib
import sys

# Source-Link: https://towardsdatascience.com/confidence-in-k-means-d7d3a13ca856
# Source-Link: https://github.com/drmattcrooks/Medium-SoftClusteringWeights
def soft_clustering_weights(data, cluster_centres, **kwargs):
    
    """
    Function to calculate the weights from soft k-means
    data: Array of data. Features arranged across the columns with each row being a different data point
    cluster_centres: array of cluster centres. Input kmeans.cluster_centres_ directly.
    param: m - keyword argument, fuzziness of the clustering. Default 2
    """
    
    # Fuzziness parameter m>=1. Where m=1 => hard segmentation
    m = 2
    if 'm' in kwargs:
        m = kwargs['m']
    
    Nclusters = cluster_centres.shape[0]
    Ndp = data.shape[0]
    Nfeatures = data.shape[1]

    # Get distances from the cluster centres for each data point and each cluster
    EuclidDist = np.zeros((Ndp, Nclusters))
    for i in range(Nclusters):
        EuclidDist[:,i] = np.sum((data-np.matlib.repmat(cluster_centres[i], Ndp, 1))**2,axis=1)
    

    
    # Denominator of the weight from wikipedia:
    invWeight = EuclidDist**(2/(m-1))*np.matlib.repmat(np.sum((1./EuclidDist)**(2/(m-1)),axis=1).reshape(-1,1),1,Nclusters)
    Weight = 1./invWeight
    
    return Weight

# read csv train data as pandas data frame
input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')			# plan of sensors weighting:
#input_data = pd.read_csv("All_Participents_ECG_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/9
#input_data = pd.read_csv("All_Participents_EDA_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/9
#input_data = pd.read_csv("All_Participents_EEG_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/9
#input_data = pd.read_csv("All_Participents_EYE_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/9
#input_data = pd.read_csv("All_Participents_PAGES_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 	# weight with 3/9

# read cvs test data
load_test_data = pd.read_csv("All_Participents_Condition-C_WaveSum_DataFrame.csv", sep=";", decimal=',')			# plan of sensors weighting:
#load_test_data = pd.read_csv("All_Participents_Condition-C_ECG_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/9
#load_test_data = pd.read_csv("All_Participents_Condition-C_EDA_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/9
#load_test_data = pd.read_csv("All_Participents_Condition-C_EEG_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/9
#load_test_data = pd.read_csv("All_Participents_Condition-C_EYE_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/9
#load_test_data = pd.read_csv("All_Participents_Condition-C_PAGES_WaveSum_DataFrame.csv", sep=";", decimal=',') 	# weight with 3/9

# ------- fitler columns of train data
train_data = input_data.drop(columns=['Conscientious', 'time', 'pId'])
# count rows and columns
#r_num = train_data.shape[0]
#print(r_num)
c_num = train_data.shape[1]
print(c_num)

# -------  filter columns of test data 
test_data = load_test_data.drop(columns=['time', 'pId'])
r_num_test_data = test_data.shape[0]
test_x = test_data.iloc[:, :].values

# ------ Normalizing
# Separating out the features
x = train_data.loc[:, :].values
# Separating out the target
y = np.array(input_data[["Conscientious"]].values.flatten())
# Standardizing the features of train data
x = StandardScaler().fit_transform(x)
# Standardizing the features of Test data
transformed_test_x = StandardScaler().fit_transform(test_x)

print("------ Principal Component Analysis n_components=2 of train data")
# ------ Principal Component Analysis n_components=2 of train data
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
print(pca.score(x)) # Debug only
print(pca.explained_variance_ratio_)  # Debug only
#print(principalComponents.shape) # df.loc[:, ["City", "Salary"]] .iloc[:, [0, 1]] .loc[df['favorite_color'] == 'yellow'] indices = np.where(input_data['Conscientious'] == 0)
#print(input_data.index[input_data['Conscientious'] == 0].tolist())
principalDataFrame = pd.DataFrame(data = principalComponents)
conscientious_indeces = input_data.index[input_data['Conscientious'] == 0]
none_conscientious_indeces = input_data.index[input_data['Conscientious'] == 1]
plt.scatter(principalComponents[conscientious_indeces.tolist(),0], principalComponents[conscientious_indeces.tolist(),1], c="b")
plt.scatter(principalComponents[none_conscientious_indeces.tolist(),0], principalComponents[none_conscientious_indeces.tolist(),1], c="r")
plt.title('Principal Component Analysis train data n_components=2 plot', fontsize=16)
plt.show()

sys.exit()
#------ correlation matrix of train data
#f = plt.figure(figsize=(28, 32))
#plt.matshow(principalDataFrame.corr(), fignum=f.number)
#plt.xticks(range(principalDataFrame.select_dtypes(['number']).shape[1]), principalDataFrame.select_dtypes(['number']).columns, fontsize=8, rotation=45)
#plt.yticks(range(principalDataFrame.select_dtypes(['number']).shape[1]), principalDataFrame.select_dtypes(['number']).columns, fontsize=8)
#cb = plt.colorbar()
#cb.ax.tick_params(labelsize=8)
#plt.title('Correlation Matrix of train data principal components', fontsize=16)
#plt.show()

#------ Principal Component Analysis n_components=2 of test data
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(transformed_test_x)
print(pca.score(transformed_test_x)) # Debug only
print(pca.explained_variance_ratio_)  # Debug only
principalTestDataFrame = pd.DataFrame(data = principalComponents)#, columns = ['principal component 1', 'principal component 2'])#, 'principal component 3', 'principal component 4'])

#------ correlation matrix of test data
#f = plt.figure(figsize=(28, 32))
#plt.matshow(principalTestDataFrame.corr(), fignum=f.number)
#plt.xticks(range(principalTestDataFrame.select_dtypes(['number']).shape[1]), principalTestDataFrame.select_dtypes(['number']).columns, fontsize=8, rotation=45)
#plt.yticks(range(principalTestDataFrame.select_dtypes(['number']).shape[1]), principalTestDataFrame.select_dtypes(['number']).columns, fontsize=8)
#cb = plt.colorbar()
#cb.ax.tick_params(labelsize=8)
#plt.title('Correlation Matrix of test data principal components', fontsize=16)
#plt.show()

print("------- Mini-Batch-K-Means Model")
# ------- Mini-Batch-K-Means Model
miniBatchKMeans = MiniBatchKMeans(init="k-means++", n_clusters=2).fit(principalDataFrame) #miniBatchKMeans = MiniBatchKMeans(n_clusters=2).fit(input_x)
input_score = miniBatchKMeans.score(principalDataFrame) #input_score = miniBatchKMeans.score(np.array(input_x)[0:1])
input_cluster_centers_ = miniBatchKMeans.cluster_centers_
print(input_score)
print(input_cluster_centers_)

train_data["Conscientious"] = miniBatchKMeans.predict(principalDataFrame) #input_x["Cluster"] = miniBatchKMeans.predict(input_x)
train_data["Conscientious"] = train_data["Conscientious"].astype("int")
train_data["pId"] = input_data["pId"]

df = DataFrame()
for i in range(2):
    df['p' + str(i)] = 0

df[['p0', 'p1']] = soft_clustering_weights(principalDataFrame, input_cluster_centers_)
df['confidence'] = np.max(df[['p0', 'p1']].values, axis = 1)
train_data["Confidence"] = df['confidence']

plt.figure(figsize=(15,7))
plt.title('Mini-Batch-K-Means-Model PCA Confidence-Histogram plot', fontsize=16)
plt.hist(df['confidence'][train_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
plt.hist(df['confidence'][train_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 

print(miniBatchKMeans.get_params(deep=True))
print(miniBatchKMeans.labels_)

# get probability score of each sample
loss = log_loss(input_data['Conscientious'], train_data['Conscientious'])
print(loss)

input_means_labels = pairwise_distances_argmin(principalDataFrame[:].values.copy(order='C'), input_cluster_centers_.copy(order='C'))
print(input_means_labels)

# ----------- miniBatchKMeans Cluster plot of principalDataFrame
p_num = principalDataFrame.shape[1]
print(p_num)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
for i in range(p_num - 1):
	ax.plot(principalDataFrame[:].values[input_means_labels == 0, i], principalDataFrame[:].values[input_means_labels == 0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
	ax.plot(principalDataFrame[:].values[input_means_labels == 1, i], principalDataFrame[:].values[input_means_labels == 1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)
cluster_center = input_cluster_centers_[0]
ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='b', markeredgecolor="k", markersize=6, zorder=2)
cluster_center = input_cluster_centers_[1]
ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='r', markeredgecolor="k", markersize=6, zorder=2)
ax.set_title('Mini-Batch-K-Means-Model PCA features plot', fontsize=16)
ax.set_xticks(())
ax.set_yticks(())
plt.show()

colors = {0:'b', 1:'r'}
plt.scatter(x=train_data['Conscientious'], y=train_data['pId'], alpha=0.5, c=train_data['Conscientious'].map(colors))
plt.title('Mini-Batch-K-Means-Model PCA Conscientious-pId plot', fontsize=16)
plt.show()

ax2 = train_data.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=train_data['Conscientious'].map(colors))
ax2.set_title("Mini-Batch-K-Means-Model PCA pId-Confidence plot", fontsize=16)
plt.show()

# ----------- miniBatchKMeans Cluster IDs plot with heighest confidence
_ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,31,34]
for id in _ids:
	temp = train_data.loc[train_data["pId"] == id]
	max_confi = temp['Confidence'].max()
	highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
	temp['Conscientious'] = temp.at[highest_confidet_index, 'Conscientious']
	train_data.Conscientious[train_data.pId == id] = temp['Conscientious'].values[0]

ax2 = train_data.plot.scatter(x='Conscientious',  y='pId', c=train_data['Conscientious'].map(colors))
ax2.set_title("Mini-Batch-K-Means-Model PCA Conscientious-pId (with heighest confidence) plot", fontsize=16)
plt.show()

print("------- Gaussian Mixtures Model")
# ------- Gaussian Mixtures Model
gaussianDataFrame = principalDataFrame
gaussianMixture = GaussianMixture(n_components=2, init_params='k-means++').fit(gaussianDataFrame)
input_score = gaussianMixture.score(gaussianDataFrame) #
input_score_sampels = gaussianMixture.score_samples(gaussianDataFrame)
input_mean = gaussianMixture.means_
print(input_score)
print(input_score_sampels)
print(input_mean)

train_data["Conscientious"] = gaussianMixture.predict(gaussianDataFrame)
train_data["Conscientious"] = train_data["Conscientious"].astype("int")

prediction=gaussianMixture.predict_proba(gaussianDataFrame)#[:,1]
print(prediction)
train_data["Confidence"] = np.max(prediction, axis = 1)

plt.figure(figsize=(15,7))
plt.title('Gaussian-Mixtures-Model PCA Confidence-Histogram plot', fontsize=16)
plt.hist(train_data['Confidence'][train_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
plt.hist(train_data['Confidence'][train_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 

print(gaussianMixture.get_params(deep=True))

# get probability score of each sample
loss = log_loss(input_data['Conscientious'], train_data['Conscientious'])
print(loss)

input_means_labels = pairwise_distances_argmin(gaussianDataFrame[:].values.copy(order='C'), input_mean.copy(order='C'))
print(input_means_labels)

# GaussianMixture
p_num = gaussianDataFrame.shape[1]
print(p_num)
fig = plt.figure(figsize=(10, 10))
colors = ["#4EACC5", "#FF9C34"]
ax = fig.add_subplot(1, 1, 1)
for i in range(p_num - 1):
	ax.plot(gaussianDataFrame[:].values[input_means_labels == 0, i], gaussianDataFrame[:].values[input_means_labels == 0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
	ax.plot(gaussianDataFrame[:].values[input_means_labels == 1, i], gaussianDataFrame[:].values[input_means_labels == 1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)
cluster_center = input_mean[0]
ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='b', markeredgecolor="k", markersize=6, zorder=2)
cluster_center = input_mean[1]
ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='r', markeredgecolor="k", markersize=6, zorder=2)
ax.set_title("Gaussian-Mixtures-Cluster PCA features plot")
ax.set_xticks(())
ax.set_yticks(())
plt.show()

train_data["pId"] = input_data["pId"]
colors = {0:'b', 1:'r'}

plt.scatter(x=train_data['Conscientious'], y=train_data['pId'], alpha=0.5, c=train_data['Conscientious'].map(colors))
plt.title('Gaussian-Mixtures-Model PCA Conscientious-pId plot', fontsize=16)
plt.show()

ax2 = train_data.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=train_data['Conscientious'].map(colors))
ax2.set_title("Gaussian-Mixtures-Model PCA pId-Confidence plot", fontsize=16)
plt.show()

# ----------- gaussianMixture Cluster IDs plot with heighest confidence
_ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,31,34]
for id in _ids:
    temp = train_data.loc[train_data["pId"] == id]
    max_confi = temp['Confidence'].max()
    highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
    temp['Conscientious'] = temp.at[highest_confidet_index, 'Conscientious']
    train_data.Conscientious[train_data.pId == id] = temp['Conscientious'].values[0]

ax2 = train_data.plot.scatter(x='Conscientious',  y='pId', c=train_data['Conscientious'].map(colors))
ax2.set_title("Gaussian-Mixtures-Model PCA Conscientious-pId (with heighest confidence) plot", fontsize=16)
plt.show()

# ------- Linear Discriminant Analysis Model
print("------- Linear Discriminant Analysis Model")
linearDiscriminantAnalysis = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
linearDiscriminantAnalysis.fit(principalDataFrame, y)

df11=pd.DataFrame(linearDiscriminantAnalysis.coef_[0].reshape(-1,1), principalDataFrame.columns, columns=["Weight"])
df12=pd.DataFrame(linearDiscriminantAnalysis.intercept_[0].reshape(-1,1), ["Bias"], columns=["Weight"])
resulty = pd.concat([df12, df11], axis=0)
print("====================== fit informations")
print(resulty)

result_array = linearDiscriminantAnalysis.predict(principalDataFrame)
print(result_array)

#sys.exit()

train_data["Conscientious"] = result_array
train_data["Conscientious"] = train_data["Conscientious"].astype("int")

prediction = linearDiscriminantAnalysis.predict_proba(principalDataFrame)
train_data["Confidence"] = np.max(prediction, axis = 1)

plt.figure(figsize=(15,7))
plt.title('Linear-Discriminant-Analysis-Model PCA training data Confidence-Histogram plot', fontsize=16)
plt.hist(train_data['Confidence'][train_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.7, color='b')
plt.hist(train_data['Confidence'][train_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.7, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 

print(linearDiscriminantAnalysis.get_params(deep=True))

# Linear Discriminant Analysis
print("Linear Discriminant Analysis Training Data Plot")
p_num = principalDataFrame.shape[1]
print(p_num)
fig = plt.figure(figsize=(10, 10))
colors = ["#4EACC5", "#FF9C34"]
ax = fig.add_subplot(1, 1, 1)
for i in range(p_num - 1):
	ax.plot(principalDataFrame[:].values[0, i], principalDataFrame[:].values[0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
	ax.plot(principalDataFrame[:].values[1, i], principalDataFrame[:].values[1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)
ax.set_title("Linear-Discriminant-Analysis-Model training data features Plot", fontsize=16)
ax.set_xticks(())
ax.set_yticks(())
plt.show()

print("Linear Discriminant Analysis training data Conscientious-pId Plot")
train_data["pId"] = input_data["pId"]
colors = {0:'b', 1:'r'}
print(train_data['Conscientious'])

plt.scatter(x=train_data['Conscientious'], y=train_data['pId'], alpha=0.5, c=train_data['Conscientious'].map(colors))
plt.title('Linear-Discriminant-Analysis-Model PCA training data Conscientious-pId plot', fontsize=16)
plt.show()

print("Linear Discriminant Analysis Training Data pId-Confidence Plot")
ax2 = train_data.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=train_data['Conscientious'].map(colors))
ax2.set_title("Linear-Discriminant-Analysis-Model PCA training data pId-Confidence plot", fontsize=16)
plt.show()

_ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,31,34]
for id in _ids:
	temp = train_data.loc[train_data["pId"] == id]
	max_confi = temp['Confidence'].max()
	highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
	temp['Conscientious'] = temp.at[highest_confidet_index, 'Conscientious']
	train_data.Conscientious[train_data.pId == id] = temp['Conscientious'].values[0]

ax2 = train_data.plot.scatter(x='Conscientious',  y='pId', c=train_data['Conscientious'].map(colors))
ax2.set_title("Linear-Discriminant-Analysis-Model PCA training data Conscientious-pId (with heighest confidence) plot", fontsize=16)
plt.show()

# ================ transformend test validation input predictions informations
print("================ transformend test validation input predictions informations")
true_value_test_data = []
test_data['pId'] = load_test_data['pId']
#ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
# set real Conscientious values
for i in range(r_num_test_data):
    true_value_test_data.append([0])
    if test_data['pId'].values[i] == 24 or test_data['pId'].values[i] == 25 or test_data['pId'].values[i] == 29:
        true_value_test_data[i] = [1]

result_array = linearDiscriminantAnalysis.predict(principalTestDataFrame)
test_data["Conscientious"] = result_array
test_data["Conscientious"] = test_data["Conscientious"].astype("int")

prediction = linearDiscriminantAnalysis.predict_proba(principalTestDataFrame)
test_data["Confidence"] = np.max(prediction, axis = 1)

plt.figure(figsize=(15,7))
plt.title('Linear-Discriminant-Analysis-Model PCA test data Confidence-Histogram plot', fontsize=16)	
plt.hist(test_data['Confidence'][test_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.7, color='b')
plt.hist(test_data['Confidence'][test_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.7, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 

p_num = principalTestDataFrame.shape[1]
print(p_num)
fig = plt.figure(figsize=(10, 10))
colors = ["#4EACC5", "#FF9C34"]
ax = fig.add_subplot(1, 1, 1)
for i in range(p_num - 1):
	ax.plot(principalTestDataFrame[:].values[0, i], principalTestDataFrame[:].values[0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
	ax.plot(principalTestDataFrame[:].values[1, i], principalTestDataFrame[:].values[1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)
ax.set_title("Linear Discriminant Analysis test data plot", fontsize=16)
ax.set_xticks(())
ax.set_yticks(())
plt.show()

colors = {0:'b', 1:'r'}
plt.scatter(x=test_data['Conscientious'], y=test_data['pId'], alpha=0.5, c=test_data['Conscientious'].map(colors))
plt.title('Linear-Discriminant-Analysis-Model PCA test data Conscientious-pId plot', fontsize=16)	
plt.show()

ax2 = test_data.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=test_data['Conscientious'].map(colors))
ax2.set_title('Linear-Discriminant-Analysis-Model PCA test data pId-Confidence plot', fontsize=16)
plt.show()

# ----------- linearDiscriminantAnalysis Cluster IDs plot with heighest confidence
_ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
for id in _ids:
	temp = test_data.loc[test_data["pId"] == id]
	max_confi = temp['Confidence'].max()
	highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
	temp['Conscientious'] = temp.at[highest_confidet_index, 'Conscientious']
	test_data.Conscientious[test_data.pId == id] = temp['Conscientious'].values[0]
	
ax2 = test_data.plot.scatter(x='Conscientious',  y='pId', c=test_data['Conscientious'].map(colors))
ax2.set_title('Linear-Discriminant-Analysis-Model PCA test data Conscientious-pId (with heighest confidence) plot', fontsize=16)
plt.show()

# ------- display linearDiscriminantAnalysis roc_auc curve
lda_roc_auc = roc_auc_score(true_value_test_data, linearDiscriminantAnalysis.predict(principalTestDataFrame))
fpr, tpr, thresholds = roc_curve(true_value_test_data, linearDiscriminantAnalysis.predict_proba(principalTestDataFrame)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Linear-Discriminant-Analysis-Model (area = %0.2f)' % lda_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Linear-Discriminant-Analysis-Model ROC curve')
plt.show()

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

# gaussianDataFrame = resultDataFrame
# # GaussianMixture
# # define the model
# model = GaussianMixture(n_components=2)

# gaussianDataFrame["Cluster"] = model.fit_predict(gaussianDataFrame)
# gaussianDataFrame["Cluster"] = gaussianDataFrame["Cluster"].astype("int")
# #print(gaussianDataFrame.head()) 

# print("=================================================== gaussianDataFrame principal component plot")

# ax2 = gaussianDataFrame.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# # show the plot
# plt.show()

# _Ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,34]
# for id in _Ids:
#     temp = gaussianDataFrame.loc[gaussianDataFrame["pId"] == id]
#     first =  temp[temp.Cluster == 0].shape[0]
#     second =  temp[temp.Cluster == 1].shape[0]
#     print(first)
#     print(second)
#     print ("test")
#     if first > second: 
#         gaussianDataFrame.Cluster[gaussianDataFrame.pId == id] = 0
#     if first < second: 
#         gaussianDataFrame.Cluster[gaussianDataFrame.pId == id] = 1

# print("=================================================== gaussianDataFrame principal component ids filter plot")

# ax2 = gaussianDataFrame.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# # show the plot
# plt.show()


# #======================================================= K-MEANS

# kMeansDataFrame = input_data

# # define the model
# model = MiniBatchKMeans(n_clusters=2)

# kMeansDataFrame["Cluster"] = model.fit_predict(kMeansDataFrame)
# kMeansDataFrame["Cluster"] = kMeansDataFrame["Cluster"].astype("int")

# print("=================================================== kMeansDataFrame normal plot")

# ax2 = kMeansDataFrame.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# # show the plot
# plt.show()

# #--------------------

# _Ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,34]
# for id in _Ids:
#     temp = kMeansDataFrame.loc[kMeansDataFrame["pId"] == id]
#     first =  temp[temp.Cluster == 0].shape[0]
#     second =  temp[temp.Cluster == 1].shape[0]
#     print(first)
#     print(second)
#     print ("test")
#     if first > second: 
#         kMeansDataFrame.Cluster[kMeansDataFrame.pId == id] = 0
#     if first < second: 
#         kMeansDataFrame.Cluster[kMeansDataFrame.pId == id] = 1

# print("=================================================== kMeansDataFrame ids filter plot")

# ax2 = kMeansDataFrame.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# # show the plot
# plt.show()

# #--------------------

# kMeansDataFrame = resultDataFrame

# # define the model
# model = MiniBatchKMeans(n_clusters=2)

# kMeansDataFrame["Cluster"] = model.fit_predict(kMeansDataFrame)
# kMeansDataFrame["Cluster"] = kMeansDataFrame["Cluster"].astype("int")

# print("=================================================== kMeansDataFrame principal component plot")

# ax2 = kMeansDataFrame.plot.scatter(x='Cluster', y='pId', c='Cluster', colormap='viridis')
# # show the plot
# plt.show()
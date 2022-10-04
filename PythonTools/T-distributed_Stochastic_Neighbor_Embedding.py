import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
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
y_result_output = np.array(input_data[["Conscientious"]].values.flatten())
print(y_result_output)
# Standardizing the features of train data
x = StandardScaler().fit_transform(x)
# Standardizing the features of Test data
transformed_test_x = StandardScaler().fit_transform(test_x)

print("------ T-Distributed Stochastic Neighbor Embedding n_components=2 of train data")
# ------ T-Distributed Stochastic Neighbor Embedding n_components=2 of train data
X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(x)
print(X_embedded.shape)
plt.scatter(X_embedded[:,0], X_embedded[:,1])
plt.title('T-Distributed Stochastic Neighbor Embedding train data n_components=2 plot', fontsize=16)
plt.show()

print("------ T-Distributed Stochastic Neighbor Embedding n_components=2 of test data")
# ------ T-Distributed Stochastic Neighbor Embedding n_components=2 of test data
test_x_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(transformed_test_x)
print(test_x_embedded.shape)
plt.scatter(test_x_embedded[:,0], test_x_embedded[:,1])
plt.title('T-Distributed Stochastic Neighbor Embedding test data n_components=2 plot', fontsize=16)
plt.show()

print("------- Mini-Batch-K-Means Model")
# ------- Mini-Batch-K-Means Model
mbkm_x_embedded_data_frame = pd.DataFrame(data = X_embedded)
mbkm_train_data = train_data.copy()
miniBatchKMeans = MiniBatchKMeans(init="k-means++", n_clusters=2).fit(mbkm_x_embedded_data_frame) #miniBatchKMeans = MiniBatchKMeans(n_clusters=2).fit(input_x)
input_score = miniBatchKMeans.score(mbkm_x_embedded_data_frame) #input_score = miniBatchKMeans.score(np.array(input_x)[0:1])
input_cluster_centers_ = miniBatchKMeans.cluster_centers_
print(input_score)
print(input_cluster_centers_)

mbkm_train_data["Conscientious"] = miniBatchKMeans.predict(mbkm_x_embedded_data_frame) #input_x["Cluster"] = miniBatchKMeans.predict(input_x)
mbkm_train_data["Conscientious"] = mbkm_train_data["Conscientious"].astype("int")
mbkm_train_data["pId"] = input_data["pId"]

df = DataFrame()
for i in range(2):
    df['p' + str(i)] = 0

df[['p0', 'p1']] = soft_clustering_weights(mbkm_x_embedded_data_frame, input_cluster_centers_)
df['confidence'] = np.max(df[['p0', 'p1']].values, axis = 1)
mbkm_train_data["Confidence"] = df['confidence']

plt.figure(figsize=(15,7))
plt.title('Mini-Batch-K-Means-Model PCA Confidence-Histogram plot', fontsize=16)
plt.hist(df['confidence'][mbkm_train_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
plt.hist(df['confidence'][mbkm_train_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 

print(miniBatchKMeans.get_params(deep=True))
print(miniBatchKMeans.labels_)

# get probability score of each sample
loss = log_loss(y_result_output, mbkm_train_data['Conscientious'])
print(loss)

input_means_labels = pairwise_distances_argmin(mbkm_x_embedded_data_frame[:].values.copy(order='C'), input_cluster_centers_.copy(order='C'))
print(input_means_labels)

# ----------- miniBatchKMeans Cluster plot of principalDataFrame
# p_num = mbkm_x_embedded_data_frame.shape[1]
# print(p_num)
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1)
# for i in range(p_num - 1):
# 	ax.plot(mbkm_x_embedded_data_frame[:].values[input_means_labels == 0, i], mbkm_x_embedded_data_frame[:].values[input_means_labels == 0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
# 	ax.plot(mbkm_x_embedded_data_frame[:].values[input_means_labels == 1, i], mbkm_x_embedded_data_frame[:].values[input_means_labels == 1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)
# cluster_center = input_cluster_centers_[0]
# ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='b', markeredgecolor="k", markersize=6, zorder=2)
# cluster_center = input_cluster_centers_[1]
# ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='r', markeredgecolor="k", markersize=6, zorder=2)
# ax.set_title('Mini-Batch-K-Means-Model PCA features plot', fontsize=16)
# ax.set_xticks(())
# ax.set_yticks(())
# plt.show()

# colors = {0:'b', 1:'r'}
# plt.scatter(x=mbkm_train_data['Conscientious'], y=mbkm_train_data['pId'], alpha=0.5, c=mbkm_train_data['Conscientious'].map(colors))
# plt.title('Mini-Batch-K-Means-Model PCA Conscientious-pId plot', fontsize=16)
# plt.show()

# ax2 = mbkm_train_data.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=mbkm_train_data['Conscientious'].map(colors))
# ax2.set_title("Mini-Batch-K-Means-Model PCA pId-Confidence plot", fontsize=16)
# plt.show()

# ----------- miniBatchKMeans Cluster IDs plot with heighest confidence
_ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,31,34]
for id in _ids:
    temp = mbkm_train_data.loc[mbkm_train_data["pId"] == id]
    max_confi = temp['Confidence'].max()
    highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
    highest_confidet = temp.at[highest_confidet_index, 'Conscientious']
    mbkm_train_data.loc[mbkm_train_data.pId == id, 'Conscientious'] = highest_confidet

# ax2 = mbkm_train_data.plot.scatter(x='Conscientious',  y='pId', c=mbkm_train_data['Conscientious'].map(colors))
# ax2.set_title("Mini-Batch-K-Means-Model PCA Conscientious-pId (with heighest confidence) plot", fontsize=16)
# plt.show()

# # ------- display roc_auc curve
model_roc_auc = roc_auc_score(y_result_output, miniBatchKMeans.predict(mbkm_x_embedded_data_frame))
fpr, tpr, thresholds = roc_curve(y_result_output, mbkm_train_data["Confidence"])
plt.figure()
plt.plot(fpr, tpr, label='Mini-Batch-K-Means-Model (area = %0.2f)' % model_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Mini-Batch-K-Means-Model ROC curve')
plt.show()


print("------- Gaussian Mixtures Model")
# ------- Gaussian Mixtures Model
gaussian_x_embedded_data_frame = pd.DataFrame(data = X_embedded)
gaussian_train_data = train_data.copy()
gaussianMixture = GaussianMixture(n_components=2).fit(gaussian_x_embedded_data_frame)
input_score = gaussianMixture.score(gaussian_x_embedded_data_frame) #
input_score_sampels = gaussianMixture.score_samples(gaussian_x_embedded_data_frame)
input_mean = gaussianMixture.means_
print(input_score)
print(input_score_sampels)
print(input_mean)

gaussian_train_data["Conscientious"] = gaussianMixture.predict(gaussian_x_embedded_data_frame)
gaussian_train_data["Conscientious"] = gaussian_train_data["Conscientious"].astype("int")

prediction=gaussianMixture.predict_proba(gaussian_x_embedded_data_frame)#[:,1]
print(prediction)
gaussian_train_data["Confidence"] = np.max(prediction, axis = 1)

plt.figure(figsize=(15,7))
plt.title('Gaussian-Mixtures-Model PCA Confidence-Histogram plot', fontsize=16)
plt.hist(gaussian_train_data['Confidence'][gaussian_train_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
plt.hist(gaussian_train_data['Confidence'][gaussian_train_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 

print(gaussianMixture.get_params(deep=True))

# get probability score of each sample
loss = log_loss(y_result_output, gaussian_train_data['Conscientious'])
print(loss)

input_means_labels = pairwise_distances_argmin(gaussian_x_embedded_data_frame[:].values.copy(order='C'), input_mean.copy(order='C'))
print(input_means_labels)

# GaussianMixture
# p_num = gaussian_x_embedded_data_frame.shape[1]
# print(p_num)
# fig = plt.figure(figsize=(10, 10))
# colors = ["#4EACC5", "#FF9C34"]
# ax = fig.add_subplot(1, 1, 1)
# for i in range(p_num - 1):
# 	ax.plot(gaussian_x_embedded_data_frame[:].values[input_means_labels == 0, i], gaussian_x_embedded_data_frame[:].values[input_means_labels == 0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
# 	ax.plot(gaussian_x_embedded_data_frame[:].values[input_means_labels == 1, i], gaussian_x_embedded_data_frame[:].values[input_means_labels == 1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)
# cluster_center = input_mean[0]
# ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='b', markeredgecolor="k", markersize=6, zorder=2)
# cluster_center = input_mean[1]
# ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='r', markeredgecolor="k", markersize=6, zorder=2)
# ax.set_title("Gaussian-Mixtures-Cluster PCA features plot")
# ax.set_xticks(())
# ax.set_yticks(())
# plt.show()

gaussian_train_data["pId"] = input_data["pId"]

#colors = {0:'b', 1:'r'}
# plt.scatter(x=gaussian_train_data['Conscientious'], y=gaussian_train_data['pId'], alpha=0.5, c=gaussian_train_data['Conscientious'].map(colors))
# plt.title('Gaussian-Mixtures-Model PCA Conscientious-pId plot', fontsize=16)
# plt.show()

# ax2 = gaussian_train_data.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=gaussian_train_data['Conscientious'].map(colors))
# ax2.set_title("Gaussian-Mixtures-Model PCA pId-Confidence plot", fontsize=16)
# plt.show()

# ----------- gaussianMixture Cluster IDs plot with heighest confidence
_ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,31,34]
for id in _ids:
    temp = gaussian_train_data.loc[gaussian_train_data["pId"] == id]
    max_confi = temp['Confidence'].max()
    highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
    highest_confidet = temp.at[highest_confidet_index, 'Conscientious']
    gaussian_train_data.loc[gaussian_train_data.pId == id, 'Conscientious'] = highest_confidet
    
# ax2 = gaussian_train_data.plot.scatter(x='Conscientious',  y='pId', c=gaussian_train_data['Conscientious'].map(colors))
# ax2.set_title("Gaussian-Mixtures-Model PCA Conscientious-pId (with heighest confidence) plot", fontsize=16)
# plt.show()

# ------- display roc_auc curve
model_roc_auc = roc_auc_score(y_result_output, gaussianMixture.predict(gaussian_x_embedded_data_frame))
fpr, tpr, thresholds = roc_curve(y_result_output, gaussian_train_data["Confidence"])
plt.figure()
plt.plot(fpr, tpr, label='Gaussian-Mixtures-Model (area = %0.2f)' % model_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Gaussian-Mixtures-Model ROC curve')
plt.show()


print("------- Linear Discriminant Analysis Model")
# ------- Linear Discriminant Analysis Model
lda_x_embedded_data_frame = pd.DataFrame(data = X_embedded)
lda_train_data = train_data.copy()
linearDiscriminantAnalysis = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
linearDiscriminantAnalysis.fit(lda_x_embedded_data_frame, y_result_output)

df11=pd.DataFrame(linearDiscriminantAnalysis.coef_[0].reshape(-1,1), lda_x_embedded_data_frame.columns, columns=["Weight"])
df12=pd.DataFrame(linearDiscriminantAnalysis.intercept_[0].reshape(-1,1), ["Bias"], columns=["Weight"])
resulty = pd.concat([df12, df11], axis=0)
print("====================== fit informations")
print(resulty)

result_array = linearDiscriminantAnalysis.predict(lda_x_embedded_data_frame)
lda_train_data["Conscientious"] = result_array
lda_train_data["Conscientious"] = lda_train_data["Conscientious"].astype("int")

prediction = linearDiscriminantAnalysis.predict_proba(lda_x_embedded_data_frame)
lda_train_data["Confidence"] = np.max(prediction, axis = 1)

plt.figure(figsize=(15,7))
plt.hist(lda_train_data['Confidence'][lda_train_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.7, color='b')
plt.hist(lda_train_data['Confidence'][lda_train_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.7, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 

print(linearDiscriminantAnalysis.get_params(deep=True))

# Linear Discriminant Analysis
# p_num = lda_x_embedded_data_frame.shape[1]
# print(p_num)
# fig = plt.figure(figsize=(10, 10))
# colors = ["#4EACC5", "#FF9C34"]
# ax = fig.add_subplot(1, 1, 1)
# for i in range(p_num - 1):
# 	ax.plot(lda_x_embedded_data_frame[:].values[0, i], lda_x_embedded_data_frame[:].values[0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
# 	ax.plot(lda_x_embedded_data_frame[:].values[1, i], lda_x_embedded_data_frame[:].values[1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)
# ax.set_title("Linear Discriminant Analysis Training Data Plot")
# ax.set_xticks(())
# ax.set_yticks(())
# plt.show()

lda_train_data["pId"] = input_data["pId"]

colors = {0:'b', 1:'r'}
plt.scatter(x=lda_train_data['Conscientious'], y=lda_train_data['pId'], alpha=0.5, c=lda_train_data['Conscientious'].map(colors))
plt.show()

ax2 = lda_train_data.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=lda_train_data['Conscientious'].map(colors))
plt.show()

_ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,31,34]
for id in _ids:
    temp = lda_train_data.loc[lda_train_data["pId"] == id]
    max_confi = temp['Confidence'].max()
    highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
    highest_confidet = temp.at[highest_confidet_index, 'Conscientious']
    lda_train_data.loc[lda_train_data.pId == id, 'Conscientious'] = highest_confidet
	
ax2 = lda_train_data.plot.scatter(x='Conscientious',  y='pId', c=lda_train_data['Conscientious'].map(colors))
plt.show()

# ------- display roc_auc curve
lda_roc_auc = roc_auc_score(y_result_output, linearDiscriminantAnalysis.predict(lda_x_embedded_data_frame))
fpr, tpr, thresholds = roc_curve(y_result_output, linearDiscriminantAnalysis.predict_proba(lda_x_embedded_data_frame)[:,1])
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


print("================ transformend test validation input predictions informations")
true_value_test_data = []
test_data['pId'] = load_test_data['pId']
#ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
# set real Conscientious values
for i in range(r_num_test_data):
    true_value_test_data.append([0])
    if test_data['pId'].values[i] == 24 or test_data['pId'].values[i] == 25: #or test_data['pId'].values[i] == 29:
        true_value_test_data[i] = [1]

result_array = linearDiscriminantAnalysis.predict(test_x_embedded)
test_data["Conscientious"] = result_array
test_data["Conscientious"] = test_data["Conscientious"].astype("int")

prediction = linearDiscriminantAnalysis.predict_proba(test_x_embedded)
test_data["Confidence"] = np.max(prediction, axis = 1)

plt.figure(figsize=(15,7))
plt.hist(test_data['Confidence'][test_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.7, color='b')
plt.hist(test_data['Confidence'][test_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.7, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 

# p_num = test_x_embedded.shape[1]
# print(p_num)
# fig = plt.figure(figsize=(10, 10))
# colors = ["#4EACC5", "#FF9C34"]
# ax = fig.add_subplot(1, 1, 1)
# for i in range(p_num - 1):
# 	ax.plot(test_x_embedded[0, i], test_x_embedded[0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
# 	ax.plot(test_x_embedded[1, i], test_x_embedded[1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)
# ax.set_title("Linear Discriminant Analysis Test Data Plot")
# ax.set_xticks(())
# ax.set_yticks(())
# plt.show()

colors = {0:'b', 1:'r'}
plt.scatter(x=test_data['Conscientious'], y=test_data['pId'], alpha=0.5, c=test_data['Conscientious'].map(colors))
plt.show()

ax2 = test_data.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=test_data['Conscientious'].map(colors))
plt.show()

# ----------- linearDiscriminantAnalysis Cluster IDs plot with heighest confidence
_ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
for id in _ids:
    temp = test_data.loc[test_data["pId"] == id]
    max_confi = temp['Confidence'].max()
    highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
    highest_confidet = temp.at[highest_confidet_index, 'Conscientious']
    test_data.loc[test_data.pId == id, 'Conscientious'] = highest_confidet
	
ax2 = test_data.plot.scatter(x='Conscientious',  y='pId', c=test_data['Conscientious'].map(colors))
plt.show()

# ------- display roc_auc curve
lda_roc_auc = roc_auc_score(true_value_test_data, linearDiscriminantAnalysis.predict(test_x_embedded))
fpr, tpr, thresholds = roc_curve(true_value_test_data, linearDiscriminantAnalysis.predict_proba(test_x_embedded)[:,1])
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
# Source-Link: https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
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
import os

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

def plot_roc_curve(true_positive_rate, false_positive_rate, legend_label, title, file_name, show=False, save=False):
    plt.figure(figsize=(15,10))
    plt.plot(false_positive_rate, true_positive_rate, label=legend_label)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = 'Receiver operating characteristic {}'.format(title)
    plt.title(title)
    plt.legend(loc="lower right")
    if save:
        plt.savefig(file_name)
    if show:
        plt.show()
    plt.close()

def plot_data_cluster(data, conscientious_indeces_list, none_conscientious_indeces_list, title, file_name, show=False, save=False):
    plt.figure(figsize=(15,10))
    plt.scatter(data[conscientious_indeces_list, 0], data[conscientious_indeces_list, 1], c="b")
    plt.scatter(data[none_conscientious_indeces_list, 0], data[none_conscientious_indeces_list, 1], c="r")
    plt.title(title, fontsize=16)
    if save:
        plt.savefig(file_name)
    if show:
        plt.show()
    plt.close()

# input_data_type = { all_sensors = 0, ecg = 1, eda = 2, eeg = 3, eye = 4, pages = 5 }
input_data_type = 0

# read csv train data as pandas data frame
input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')			# plan of sensors weighting:
if input_data_type == 1: 
	input_data = pd.read_csv("All_Participents_ECG_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/10
if input_data_type == 2: 
	input_data = pd.read_csv("All_Participents_EDA_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/10
if input_data_type == 3: 
	input_data = pd.read_csv("All_Participents_EEG_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/10
if input_data_type == 4: 
	input_data = pd.read_csv("All_Participents_EYE_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/10
if input_data_type == 5: 
	input_data = pd.read_csv("All_Participents_PAGES_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 	# weight with 4/10

# read cvs test data
load_test_data = pd.read_csv("All_Participents_Condition-C_WaveSum_DataFrame.csv", sep=";", decimal=',')			# plan of sensors weighting:
if input_data_type == 1: 
	load_test_data = pd.read_csv("All_Participents_Condition-C_ECG_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/10
if input_data_type == 2: 
	load_test_data = pd.read_csv("All_Participents_Condition-C_EDA_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/10
if input_data_type == 3: 
	load_test_data = pd.read_csv("All_Participents_Condition-C_EEG_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/10
if input_data_type == 4: 
	load_test_data = pd.read_csv("All_Participents_Condition-C_EYE_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/10
if input_data_type == 5: 
	load_test_data = pd.read_csv("All_Participents_Condition-C_PAGES_WaveSum_DataFrame.csv", sep=";", decimal=',') 	# weight with 4/10

# ------- fitler columns of train data
train_data = input_data.drop(columns=['Conscientious', 'time', 'pId'])
# count rows and columns
#r_num = train_data.shape[0]
#print(r_num)
c_num = train_data.shape[1]
print(c_num)

# exc_cols = [col for col in train_data.columns if col not in ['DegTimeLowQuality', 'EvaluatedGlobalTIMERSICalc']]
# train_data.loc[train_data.DegTimeLowQuality > 0, exc_cols] *= 2.0
# train_data.loc[train_data.EvaluatedGlobalTIMERSICalc >= 1, exc_cols] *= 2.0

# -------  filter columns of test data 
test_data = load_test_data.drop(columns=['time', 'pId'])

# exc_cols = [col for col in test_data.columns if col not in ['DegTimeLowQuality', 'EvaluatedGlobalTIMERSICalc']]
# test_data.loc[test_data.DegTimeLowQuality > 0, exc_cols] *= 2.0
# test_data.loc[test_data.EvaluatedGlobalTIMERSICalc >= 1, exc_cols] *= 2.0

r_num_test_data = test_data.shape[0]
test_x = test_data.iloc[:, :].values
# ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
# set real Conscientious values
true_value_test_data = []
for i in range(r_num_test_data):
    true_value_test_data.append(0)
    if load_test_data['pId'].values[i] == 24 or load_test_data['pId'].values[i] == 25: # or load_test_data['pId'].values[i] == 28:
        true_value_test_data[i] = 1
true_value_test_data = pd.DataFrame({ "Conscientious" : true_value_test_data})      


# ------ Normalizing
# Separating out the features
x = train_data.loc[:, :].values
# Separating out the target
y_result_output = np.array(input_data[["Conscientious"]].values.flatten())
# Standardizing the features of train data
x = StandardScaler().fit_transform(x)
# Standardizing the features of Test data
transformed_test_x = StandardScaler().fit_transform(test_x)

# set sensor and validity score weights
# weight_ecg = 2/5       #train_data.loc[:,1:26]                                 -> count() = 26
# weight_eda = 3/5       #train_data.loc[:,27:31]                                -> count() = 5
# weight_eeg = 1/5       #train_data.loc[:,32:107]  , train_data.loc[:,141:145]  -> count() = 76, 5
# weight_eye = 2/5       #train_data.loc[:,108:117] , train_data.loc[:,130:137]  -> count() = 10, 8
# weight_pages = 1       #train_data.loc[:,118:129] , train_data.loc[:,138:140]  -> count() = 12, 3
# ------
weight_ecg = 1/5       #train_data.loc[:,1:26]                                 -> count() = 26
weight_eda = 2/5       #train_data.loc[:,27:31]                                -> count() = 5
weight_eeg = 1/5       #train_data.loc[:,32:107]  , train_data.loc[:,141:145]  -> count() = 76, 5
weight_eye = 1/5       #train_data.loc[:,108:117] , train_data.loc[:,130:137]  -> count() = 10, 8
weight_pages = 1       #train_data.loc[:,118:129] , train_data.loc[:,138:140]  -> count() = 12, 3

if input_data_type == 0:
	x[:,0:26]    = x[:,0:26]    * weight_ecg
	x[:,26:31]   = x[:,26:31]   * weight_eda
	x[:,31:107]  = x[:,31:107]  * weight_eeg
	x[:,140:145] = x[:,140:145] * weight_eeg
	x[:,107:117] = x[:,107:117] * weight_eye
	x[:,129:137] = x[:,129:137] * weight_eye
	x[:,117:129] = x[:,117:129] * weight_pages
	x[:,137:140] = x[:,137:140] * weight_pages

	transformed_test_x[:,0:26]    = transformed_test_x[:,0:26]    * weight_ecg
	transformed_test_x[:,26:31]   = transformed_test_x[:,26:31]   * weight_eda
	transformed_test_x[:,31:107]  = transformed_test_x[:,31:107]  * weight_eeg
	transformed_test_x[:,140:145] = transformed_test_x[:,140:145] * weight_eeg
	transformed_test_x[:,107:117] = transformed_test_x[:,107:117] * weight_eye
	transformed_test_x[:,129:137] = transformed_test_x[:,129:137] * weight_eye
	transformed_test_x[:,117:129] = transformed_test_x[:,117:129] * weight_pages
	transformed_test_x[:,137:140] = transformed_test_x[:,137:140] * weight_pages
if input_data_type == 1:
	x[:,:] = x[:,:] * weight_ecg
	transformed_test_x[:,:]  = transformed_test_x[:,:]  * weight_ecg
if input_data_type == 2:
	x[:,:] = x[:,:] * weight_eda
	transformed_test_x[:,:]  = transformed_test_x[:,:]  * weight_eda
if input_data_type == 3:
	x[:,:] = x[:,:] * weight_eeg
	transformed_test_x[:,:]  = transformed_test_x[:,:]  * weight_eeg
if input_data_type == 4:
	x[:,:] = x[:,:] * weight_eye
	transformed_test_x[:,:]  = transformed_test_x[:,:]  * weight_eye
if input_data_type == 5:
	x[:,:] = x[:,:] * weight_pages
	transformed_test_x[:,:]  = transformed_test_x[:,:]  * weight_pages

print("Create output directory")
# --- create dir
mode = 0o666
if not os.path.exists("./output"):
    os.mkdir("./output", mode)
path = "./output/Principal_Component_Analysis_{}".format(input_data_type)
if not os.path.exists(path):
    os.mkdir(path, mode)
path_knc = "{}/K-Neighbors-Classifier-Model".format(path)
if not os.path.exists(path_knc):
    os.mkdir(path_knc, mode)
path_mbkm = "{}/Mini-Batch-K-Means-Model".format(path)
if not os.path.exists(path_mbkm):
    os.mkdir(path_mbkm, mode)
path_gmm = "{}/Gaussian-Mixtures-Model".format(path)
if not os.path.exists(path_gmm):
    os.mkdir(path_gmm, mode)
path_lda = "{}/Linear-Discriminant-Analysis-Model".format(path)
if not os.path.exists(path_lda):
    os.mkdir(path_lda, mode)

print("------ Principal Component Analysis n_components=2 of train data")
# ------ Principal Component Analysis n_components=2 of train data
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
print(pca.score(x)) # Debug only
print(pca.explained_variance_ratio_)  # Debug only

conscientious_indeces = input_data.index[input_data['Conscientious'] == 0]
none_conscientious_indeces = input_data.index[input_data['Conscientious'] == 1]
plt.figure(figsize=(15,10))
plt.scatter(principalComponents[conscientious_indeces.tolist(),0], principalComponents[conscientious_indeces.tolist(),1], c="b")
plt.scatter(principalComponents[none_conscientious_indeces.tolist(),0], principalComponents[none_conscientious_indeces.tolist(),1], c="r")
plt.title('Principal Component Analysis train data n_components=2 plot', fontsize=16)
file_name = '{}/True_principal_components_train_data_plot.png'.format(path)
plt.savefig(file_name)
plt.close()

# plt.figure(figsize=(15,10))
# ax = plt.axes(projection='3d')
# ax.scatter(principalComponents[conscientious_indeces.tolist(),0], principalComponents[conscientious_indeces.tolist(),1], principalComponents[conscientious_indeces.tolist(),2], c="b", linewidth=0.5)
# ax.scatter(principalComponents[none_conscientious_indeces.tolist(),0], principalComponents[none_conscientious_indeces.tolist(),1], principalComponents[none_conscientious_indeces.tolist(),2], c="r", linewidth=0.5)
# plt.title('T-Distributed Stochastic Neighbor Embedding n_components=3 of (True) train data  plot', fontsize=16)
# plt.show()
# plt.close()

principal_components_test_x = pca.fit_transform(transformed_test_x)
print(pca.score(transformed_test_x)) # Debug only
print(pca.explained_variance_ratio_)  # Debug only

conscientious_indeces = true_value_test_data.index[true_value_test_data['Conscientious'] == 0]
none_conscientious_indeces = true_value_test_data.index[true_value_test_data['Conscientious'] == 1]
plt.figure(figsize=(15,10))
plt.scatter(principal_components_test_x[conscientious_indeces.tolist(),0], principal_components_test_x[conscientious_indeces.tolist(),1], c="b")
plt.scatter(principal_components_test_x[none_conscientious_indeces.tolist(),0], principal_components_test_x[none_conscientious_indeces.tolist(),1], c="r")
plt.title('Principal Component Analysis test data n_components=2 plot', fontsize=16)
file_name = '{}/True_principal_components_test_data_plot.png'.format(path)
plt.savefig(file_name)
plt.close()

# plt.figure(figsize=(15,10))
# ax = plt.axes(projection='3d')
# ax.scatter(principal_components_test_x[conscientious_indeces.tolist(),0], principal_components_test_x[conscientious_indeces.tolist(),1], principal_components_test_x[conscientious_indeces.tolist(),2], c="b", linewidth=0.5)
# ax.scatter(principal_components_test_x[none_conscientious_indeces.tolist(),0], principal_components_test_x[none_conscientious_indeces.tolist(),1], principal_components_test_x[none_conscientious_indeces.tolist(),2], c="r", linewidth=0.5)
# plt.title('T-Distributed Stochastic Neighbor Embedding n_components=3 of (True) train data  plot', fontsize=16)
# plt.show()
# plt.close()

print("------- K-Neighbors-Classifier-Model")
# ------- K-Neighbors-Classifier-Model
knc_x_embedded_data_frame = pd.DataFrame(data = principalComponents)
knc_train_data = train_data.copy()
# --- training (fitting)
k_neigbors_classifier = KNeighborsClassifier(n_neighbors=200, weights='uniform', algorithm='ball_tree')
k_neigbors_classifier.fit(knc_x_embedded_data_frame, y_result_output) 

# --- splitter
# splitter = ShuffleSplit(n_splits=10, train_size=0.4, test_size=0.4)
# #splitter = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=36851234)
# for _ in range(10):
#     for train_index, test_index in splitter.split(knc_x_embedded_data_frame): # , y_result_output):
#         print("TRAIN:", train_index, "TEST:", test_index)
#         X_train, X_test = knc_x_embedded_data_frame.values[train_index], knc_x_embedded_data_frame.values[test_index]
#         y_train, y_test = y_result_output[train_index], y_result_output[test_index]
#         k_neigbors_classifier.fit(X_train, y_train)

input_score = k_neigbors_classifier.score(knc_x_embedded_data_frame, y_result_output) 
print(input_score)
# --- train data predictions 
knc_train_data["Conscientious"] = k_neigbors_classifier.predict(knc_x_embedded_data_frame) 
knc_train_data["Conscientious"] = knc_train_data["Conscientious"].astype("int")
knc_train_data["pId"] = input_data["pId"]

prediction = k_neigbors_classifier.predict_proba(knc_x_embedded_data_frame)
knc_train_data["Confidence"] = np.max(prediction, axis = 1)

# plt.figure(figsize=(15,10))
# plt.title('K-Neighbors-Classifier-Model train data Confidence-Histogram plot', fontsize=16)
# plt.hist(knc_train_data['Confidence'][knc_train_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
# plt.hist(knc_train_data['Confidence'][knc_train_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
# plt.xlabel('Calculated Probability', fontsize=25)
# plt.ylabel('Number of records', fontsize=25)
# plt.legend(fontsize=15)
# plt.tick_params(axis='both', labelsize=25, pad=5)
# plt.close()

print(k_neigbors_classifier.get_params(deep=True))

# get probability score of each sample
loss = log_loss(y_result_output, knc_train_data['Conscientious'])
print(loss)
print("------ K-Neighbors-Classifier-Model n_components=2 of (predicted) train data ")
conscientious_indeces = knc_train_data.index[knc_train_data['Conscientious'] == 0]
none_conscientious_indeces = knc_train_data.index[knc_train_data['Conscientious'] == 1]
file_name = '{}/K-Neighbors-Classifier-Model_predicted_train_data_plot.png'.format(path_knc)
plot_data_cluster(principalComponents, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'K-Neighbors-Classifier-Model n_components=2 of (predicted) train data plot', file_name, show=False, save=True)

# ------- display roc_auc curve
knc_roc_auc = roc_auc_score(input_data[["Conscientious"]], k_neigbors_classifier.predict(knc_x_embedded_data_frame))
fpr, tpr, thresholds = roc_curve(input_data[["Conscientious"]], k_neigbors_classifier.predict_proba(knc_x_embedded_data_frame)[:,1])
file_name = '{}/K-Neighbors-Classifier-Model_train_data_roc-curve.png'.format(path_knc)
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'K-Neighbors-Classifier-Model train data (area = %0.2f)' % knc_roc_auc, 
               title = 'K-Neighbors-Classifier-Model train data', file_name = file_name, show=False, save=True)

# --- test data predictions 
knc_test_x_embedded_data_frame = pd.DataFrame(data = principal_components_test_x)
knc_test_data = test_data.copy()
knc_test_data["Conscientious"] = k_neigbors_classifier.predict(knc_test_x_embedded_data_frame) 
knc_test_data["Conscientious"] = knc_test_data["Conscientious"].astype("int")
knc_test_data["pId"] = load_test_data["pId"]

prediction = k_neigbors_classifier.predict_proba(knc_test_x_embedded_data_frame)
knc_test_data["Confidence"] = np.max(prediction, axis = 1)

# ----------- Cluster IDs plot with heighest confidence
colors = {0:'b', 1:'r'}
_ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
for id in _ids:
	temp = knc_test_data.loc[knc_test_data["pId"] == id]
	max_confi = temp['Confidence'].max()
	highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
	highest_confidet = temp.at[highest_confidet_index, 'Conscientious']
	knc_test_data.loc[knc_test_data.pId == id, 'Conscientious'] = highest_confidet 
	
ax2 = knc_test_data.plot.scatter(x='Conscientious',  y='pId', c=knc_test_data['Conscientious'].map(colors))
plt.show()
plt.close()

# plt.figure(figsize=(15,7))
# plt.title('K-Neighbors-Classifier-Model test data Confidence-Histogram plot', fontsize=16)
# plt.hist(knc_test_data['Confidence'][knc_test_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
# plt.hist(knc_test_data['Confidence'][knc_test_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
# plt.xlabel('Calculated Probability', fontsize=25)
# plt.ylabel('Number of records', fontsize=25)
# plt.legend(fontsize=15)
# plt.tick_params(axis='both', labelsize=25, pad=5)
# plt.close()

print(k_neigbors_classifier.get_params(deep=True))

# get probability score of each sample
loss = log_loss(true_value_test_data['Conscientious'], knc_test_data['Conscientious'])
print(loss)
print("------ T-Distributed Stochastic Neighbor Embedding n_components=2 of (predicted) test data ")
conscientious_indeces = knc_test_data.index[knc_test_data['Conscientious'] == 0]
none_conscientious_indeces = knc_test_data.index[knc_test_data['Conscientious'] == 1]
file_name = '{}/K-Neighbors-Classifier-Model_predicted_test_data_plot.png'.format(path_knc)
plot_data_cluster(principal_components_test_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'K-Neighbors-Classifier-Model n_components=2 of (predicted) test data plot', file_name, show=False, save=True)

# ------- display roc_auc curve
knc_roc_auc = roc_auc_score(true_value_test_data['Conscientious'], k_neigbors_classifier.predict(knc_test_x_embedded_data_frame))
fpr, tpr, thresholds = roc_curve(true_value_test_data['Conscientious'], k_neigbors_classifier.predict_proba(knc_test_x_embedded_data_frame)[:,1])
file_name = '{}/K-Neighbors-Classifier-Model_test_data_roc-curve.png'.format(path_knc)
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'K-Neighbors-Classifier-Model test data (area = %0.2f)' % knc_roc_auc,
               title = 'K-Neighbors-Classifier-Model test data', file_name = file_name, show=False, save=True)

print("------- Mini-Batch-K-Means Model")
# ------- Mini-Batch-K-Means Model
mbkm_x_embedded_data_frame = pd.DataFrame(data = principalComponents)
mbkm_train_data = train_data.copy()
miniBatchKMeans = MiniBatchKMeans(init="k-means++", n_clusters=2).fit(mbkm_x_embedded_data_frame) 
input_score = miniBatchKMeans.score(mbkm_x_embedded_data_frame) 
input_cluster_centers_ = miniBatchKMeans.cluster_centers_
print(input_score)
print(input_cluster_centers_)

print(" --- training data mbkm")
# --- training data
mbkm_train_data["Conscientious"] = miniBatchKMeans.predict(mbkm_x_embedded_data_frame) #input_x["Cluster"] = miniBatchKMeans.predict(input_x)
mbkm_train_data["Conscientious"] = mbkm_train_data["Conscientious"].astype("int")
mbkm_train_data["pId"] = input_data["pId"]

df = DataFrame()
for i in range(2):
    df['p' + str(i)] = 0

df[['p0', 'p1']] = soft_clustering_weights(mbkm_x_embedded_data_frame, input_cluster_centers_)
df['confidence'] = np.max(df[['p0', 'p1']].values, axis = 1)
mbkm_train_data["Confidence"] = df['confidence']

# plt.figure(figsize=(15,7))
# plt.title('Mini-Batch-K-Means-Model Confidence-Histogram plot', fontsize=16)
# plt.hist(df['confidence'][mbkm_train_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
# plt.hist(df['confidence'][mbkm_train_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
# plt.xlabel('Calculated Probability', fontsize=25)
# plt.ylabel('Number of records', fontsize=25)
# plt.legend(fontsize=15)
# plt.tick_params(axis='both', labelsize=25, pad=5)
# plt.show() 
# plt.close()

print(miniBatchKMeans.get_params(deep=True))
print(miniBatchKMeans.labels_)

print("------ Mini-Batch-K-Means-Model n_components=2 of (predicted) train data ")
conscientious_indeces = mbkm_train_data.index[mbkm_train_data['Conscientious'] == 0]
none_conscientious_indeces = mbkm_train_data.index[mbkm_train_data['Conscientious'] == 1]
file_name = '{}/Mini-Batch-K-Means-Model_predicted_train_data_plot.png'.format(path_mbkm)
plot_data_cluster(principalComponents, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'Mini-Batch-K-Means-Model n_components=2 of (predicted) train data plot', file_name, show=False, save=True)

# get probability score of each sample
loss = log_loss(y_result_output, mbkm_train_data['Conscientious'])
print(loss)

input_means_labels = pairwise_distances_argmin(mbkm_x_embedded_data_frame[:].values.copy(order='C'), input_cluster_centers_.copy(order='C'))
print(input_means_labels)

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
file_name = '{}/Mini-Batch-K-Means-Model_training-data_ROC_curve.png'.format(path_mbkm)
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'Mini-Batch-K-Means-Model train data (area = %0.2f)' % model_roc_auc, 
               title = 'Mini-Batch-K-Means-Model train data', file_name = file_name, show=False, save=True)

print(" --- test data mbkm")
# --- test data
mbkm_test_x_embedded_data_frame = pd.DataFrame(data = principal_components_test_x)
mbkm_test_data = test_data.copy()
mbkm_test_data["Conscientious"] = miniBatchKMeans.predict(mbkm_test_x_embedded_data_frame) 
mbkm_test_data["Conscientious"] = mbkm_test_data["Conscientious"].astype("int")
mbkm_test_data["pId"] = load_test_data["pId"]

input_score = miniBatchKMeans.score(mbkm_test_x_embedded_data_frame) #input_score = miniBatchKMeans.score(np.array(input_x)[0:1])
input_cluster_centers_ = miniBatchKMeans.cluster_centers_
print(input_score)
print(input_cluster_centers_)

df = DataFrame()
for i in range(2):
    df['p' + str(i)] = 0

df[['p0', 'p1']] = soft_clustering_weights(mbkm_test_x_embedded_data_frame, input_cluster_centers_)
df['confidence'] = np.max(df[['p0', 'p1']].values, axis = 1)
mbkm_test_data["Confidence"] = df['confidence']

# plt.figure(figsize=(15,7))
# plt.title('Mini-Batch-K-Means-Model Confidence-Histogram plot', fontsize=16)
# plt.hist(df['confidence'][mbkm_test_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
# plt.hist(df['confidence'][mbkm_test_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
# plt.xlabel('Calculated Probability', fontsize=25)
# plt.ylabel('Number of records', fontsize=25)
# plt.legend(fontsize=15)
# plt.tick_params(axis='both', labelsize=25, pad=5)
# plt.show() 
# plt.close()

print(miniBatchKMeans.get_params(deep=True))
print(miniBatchKMeans.labels_)

print("------ Mini-Batch-K-Means-Model n_components=2 of (predicted) test data ")
conscientious_indeces = mbkm_test_data.index[mbkm_test_data['Conscientious'] == 0]
none_conscientious_indeces = mbkm_test_data.index[mbkm_test_data['Conscientious'] == 1]
file_name = '{}/Mini-Batch-K-Means-Model_predicted_test_data_plot.png'.format(path_mbkm)
plot_data_cluster(principal_components_test_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'Mini-Batch-K-Means-Model n_components=2 of (predicted) test data plot', file_name, show=False, save=True)

# get probability score of each sample
loss = log_loss(true_value_test_data['Conscientious'], mbkm_test_data['Conscientious'])
print(loss)

input_means_labels = pairwise_distances_argmin(mbkm_test_x_embedded_data_frame[:].values.copy(order='C'), input_cluster_centers_.copy(order='C'))
print(input_means_labels)

# # ------- display roc_auc curve
mbkm_roc_auc = roc_auc_score(true_value_test_data['Conscientious'], miniBatchKMeans.predict(knc_test_x_embedded_data_frame))
fpr, tpr, thresholds = roc_curve(true_value_test_data['Conscientious'],  mbkm_test_data["Confidence"])
file_name = '{}/Mini-Batch-K-Means-Model_test-data_ROC_curve.png'.format(path_mbkm)
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'Mini-Batch-K-Means-Model test data (area = %0.2f)' % mbkm_roc_auc,
               title = 'Mini-Batch-K-Means-Model test data', file_name = file_name, show=False, save=True)

print("------- Gaussian Mixtures Model")
# ------- Gaussian Mixtures Model
gaussian_x_embedded_data_frame = pd.DataFrame(data = principalComponents)
gaussian_train_data = train_data.copy()
gaussianMixture = GaussianMixture(n_components=2, init_params='k-means++')#.fit(gaussian_x_embedded_data_frame)

print("--- training data gmm")
# --- training data
gaussianMixture.fit(gaussian_x_embedded_data_frame)
# # --- splitter
# splitter = ShuffleSplit(n_splits=10, train_size=0.4, test_size=0.4)
# #splitter = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=36851234)
# for _ in range(10):
#     for train_index, test_index in splitter.split(gaussian_x_embedded_data_frame): # , y_result_output):
#         print("TRAIN:", train_index, "TEST:", test_index)
#         X_train, X_test = gaussian_x_embedded_data_frame.values[train_index], gaussian_x_embedded_data_frame.values[test_index]
#         y_train, y_test = y_result_output[train_index], y_result_output[test_index]
#         gaussianMixture.fit(X_train, y_train)

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

# plt.figure(figsize=(15,7))
# plt.title('Gaussian-Mixtures-Model Confidence-Histogram plot', fontsize=16)
# plt.hist(gaussian_train_data['Confidence'][gaussian_train_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
# plt.hist(gaussian_train_data['Confidence'][gaussian_train_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
# plt.xlabel('Calculated Probability', fontsize=25)
# plt.ylabel('Number of records', fontsize=25)
# plt.legend(fontsize=15)
# plt.tick_params(axis='both', labelsize=25, pad=5)
# plt.show() 
# plt.close()

print(gaussianMixture.get_params(deep=True))

# get probability score of each sample
loss = log_loss(y_result_output, gaussian_train_data['Conscientious'])
print(loss)

input_means_labels = pairwise_distances_argmin(gaussian_x_embedded_data_frame[:].values.copy(order='C'), input_mean.copy(order='C'))
print(input_means_labels)

print("------ Gaussian-Mixtures-Model n_components=2 of (predicted) train data ")
conscientious_indeces = gaussian_train_data.index[gaussian_train_data['Conscientious'] == 0]
none_conscientious_indeces = gaussian_train_data.index[gaussian_train_data['Conscientious'] == 1]
file_name = '{}/Gaussian-Mixtures-Model_predicted_train_data_plot.png'.format(path_gmm)
plot_data_cluster(principalComponents, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'Gaussian-Mixtures-Model n_components=2 of (predicted) train data plot', file_name, show=False, save=True)

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
file_name = '{}/Gaussian-Mixtures-Model_training-data_ROC-curve.png'.format(path_gmm)
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'Gaussian-Mixtures-Model training data (area = %0.2f)' % model_roc_auc, 
               title = 'Gaussian-Mixtures-Model training data', file_name = file_name, show=False, save=True)

print("--- test data gmm")
# --- test data
gmm_test_x_embedded_data_frame = pd.DataFrame(data = principal_components_test_x)
gmm_test_data = test_data.copy()
gmm_test_data["Conscientious"] = gaussianMixture.predict(gmm_test_x_embedded_data_frame)
gmm_test_data["Conscientious"] = gmm_test_data["Conscientious"].astype("int")

prediction=gaussianMixture.predict_proba(gmm_test_x_embedded_data_frame)#[:,1]
print(prediction)
gmm_test_data["Confidence"] = np.max(prediction, axis = 1)

# plt.figure(figsize=(15,7))
# plt.title('Gaussian-Mixtures-Model PCA Confidence-Histogram plot', fontsize=16)
# plt.hist(gmm_test_data['Confidence'][gmm_test_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
# plt.hist(gmm_test_data['Confidence'][gmm_test_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
# plt.xlabel('Calculated Probability', fontsize=25)
# plt.ylabel('Number of records', fontsize=25)
# plt.legend(fontsize=15)
# plt.tick_params(axis='both', labelsize=25, pad=5)
# plt.show() 
# plt.close()

print(gaussianMixture.get_params(deep=True))

# get probability score of each sample
loss = log_loss(true_value_test_data['Conscientious'], gmm_test_data['Conscientious'])
print(loss)

input_means_labels = pairwise_distances_argmin(gmm_test_x_embedded_data_frame[:].values.copy(order='C'), input_mean.copy(order='C'))
print(input_means_labels)

print("------ Gaussian-Mixtures-Model n_components=2 of (predicted) test data ")
conscientious_indeces = gmm_test_data.index[gmm_test_data['Conscientious'] == 0]
none_conscientious_indeces = gmm_test_data.index[gmm_test_data['Conscientious'] == 1]
file_name = '{}/Gaussian-Mixtures-Model_predicted_test_data_plot.png'.format(path_gmm)
plot_data_cluster(principal_components_test_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'Gaussian-Mixtures-Model n_components=2 of (predicted) test data plot', file_name, show=False, save=True)

gmm_test_data["pId"] = load_test_data["pId"]

# ----------- gaussianMixture Cluster IDs plot with heighest confidence
_ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,31,34]
for id in _ids:
    temp = gaussian_train_data.loc[gaussian_train_data["pId"] == id]
    max_confi = temp['Confidence'].max()
    highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
    highest_confidet = temp.at[highest_confidet_index, 'Conscientious']
    gaussian_train_data.loc[gaussian_train_data.pId == id, 'Conscientious'] = highest_confidet

# ------- display roc_auc curve
model_roc_auc = roc_auc_score(true_value_test_data['Conscientious'], gaussianMixture.predict(gmm_test_x_embedded_data_frame))
fpr, tpr, thresholds = roc_curve(true_value_test_data['Conscientious'], gmm_test_data["Confidence"])
file_name = '{}/Gaussian-Mixtures-Model_test-data_ROC-curve.png'.format(path_gmm)
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'Gaussian-Mixtures-Model test data (area = %0.2f)' % model_roc_auc, 
               title = 'Gaussian-Mixtures-Model test data', file_name = file_name, show=False, save=True)

print("------- Linear Discriminant Analysis Model")
# ------- Linear Discriminant Analysis Model
lda_x_embedded_data_frame = pd.DataFrame(data = principalComponents)
lda_train_data = train_data.copy()
linearDiscriminantAnalysis = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
linearDiscriminantAnalysis.fit(lda_x_embedded_data_frame, y_result_output)

# # --- splitter
# splitter = ShuffleSplit(n_splits=10, train_size=0.4, test_size=0.4)
# #splitter = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=36851234)
# for _ in range(100):
#     for train_index, test_index in splitter.split(lda_x_embedded_data_frame): # , y_result_output):
#         print("TRAIN:", train_index, "TEST:", test_index)
#         X_train, X_test = lda_x_embedded_data_frame.values[train_index], lda_x_embedded_data_frame.values[test_index]
#         y_train, y_test = y_result_output[train_index], y_result_output[test_index]
#         linearDiscriminantAnalysis.fit(X_train, y_train)

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

# plt.figure(figsize=(15,7))
# plt.hist(lda_train_data['Confidence'][lda_train_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.7, color='b')
# plt.hist(lda_train_data['Confidence'][lda_train_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.7, color='r')
# plt.xlabel('Calculated Probability', fontsize=25)
# plt.ylabel('Number of records', fontsize=25)
# plt.legend(fontsize=15)
# plt.tick_params(axis='both', labelsize=25, pad=5)
# plt.show() 
# plt.close()

print(linearDiscriminantAnalysis.get_params(deep=True))

print(" ------ Linear-Discriminant-Analysis-Model n_components=2 of (predicted) train data")
conscientious_indeces = lda_train_data.index[lda_train_data['Conscientious'] == 0]
none_conscientious_indeces = lda_train_data.index[lda_train_data['Conscientious'] == 1]
file_name = '{}/Linear-Discriminant-Analysis-Model_predicted_train_data_plot.png'.format(path_lda)
plot_data_cluster(principalComponents, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'Linear-Discriminant-Analysis-Model n_components=2 of (predicted) train data plot', file_name, show=False, save=True)

lda_train_data["pId"] = input_data["pId"]

colors = {0:'b', 1:'r'}
plt.scatter(x=lda_train_data['Conscientious'], y=lda_train_data['pId'], alpha=0.5, c=lda_train_data['Conscientious'].map(colors))
#plt.show()
plt.close()

ax2 = lda_train_data.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=lda_train_data['Conscientious'].map(colors))
#plt.show()
plt.close()

_ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,31,34]
for id in _ids:
    temp = lda_train_data.loc[lda_train_data["pId"] == id]
    max_confi = temp['Confidence'].max()
    highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
    highest_confidet = temp.at[highest_confidet_index, 'Conscientious']
    lda_train_data.loc[lda_train_data.pId == id, 'Conscientious'] = highest_confidet
	
ax2 = lda_train_data.plot.scatter(x='Conscientious',  y='pId', c=lda_train_data['Conscientious'].map(colors))
plt.show()
plt.close()

# ------- display roc_auc curve
lda_roc_auc = roc_auc_score(y_result_output, linearDiscriminantAnalysis.predict(lda_x_embedded_data_frame))
fpr, tpr, thresholds = roc_curve(y_result_output, linearDiscriminantAnalysis.predict_proba(lda_x_embedded_data_frame)[:,1])
file_name = '{}/Linear-Discriminant-Analysis-Model_traingin-data_ROC-curve.png'.format(path_lda)
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'Linear-Discriminant-Analysis-Model train data (area = %0.2f)' % lda_roc_auc, 
               title = 'Linear-Discriminant-Analysis-Model train data', file_name = file_name, show=False, save=True)

print("================ transformend test validation input predictions informations")
# true_value_test_data = []
# #test_data['pId'] = load_test_data['pId']
# #ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
# # set real Conscientious values
# for i in range(r_num_test_data):
#     true_value_test_data.append([0])
#     if load_test_data['pId'].values[i] == 24 or load_test_data['pId'].values[i] == 25: #or load_test_data['pId'].values[i] == 29:
#         true_value_test_data[i] = [1]

lda_test_x_embedded_data_frame = pd.DataFrame(data = principal_components_test_x)
lda_test_data = test_data.copy()

result_array = linearDiscriminantAnalysis.predict(lda_test_x_embedded_data_frame)
lda_test_data["Conscientious"] = result_array
lda_test_data["Conscientious"] = lda_test_data["Conscientious"].astype("int")

prediction = linearDiscriminantAnalysis.predict_proba(lda_test_x_embedded_data_frame)
lda_test_data["Confidence"] = np.max(prediction, axis = 1)

# plt.figure(figsize=(15,7))
# plt.hist(lda_test_data['Confidence'][lda_test_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.7, color='b')
# plt.hist(lda_test_data['Confidence'][lda_test_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.7, color='r')
# plt.xlabel('Calculated Probability', fontsize=25)
# plt.ylabel('Number of records', fontsize=25)
# plt.legend(fontsize=15)
# plt.tick_params(axis='both', labelsize=25, pad=5)
# plt.show() 
# plt.close()

lda_test_data['pId'] = load_test_data['pId']

colors = {0:'b', 1:'r'}
plt.scatter(x=lda_test_data['Conscientious'], y=lda_test_data['pId'], alpha=0.5, c=lda_test_data['Conscientious'].map(colors))
#plt.show()
plt.close()

ax2 = lda_test_data.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=lda_test_data['Conscientious'].map(colors))
#plt.show()
plt.close()

print(" ------ Linear-Discriminant-Analysis-Model n_components=2 of (predicted) test data")
conscientious_indeces = lda_test_data.index[lda_test_data['Conscientious'] == 0]
none_conscientious_indeces = lda_test_data.index[lda_test_data['Conscientious'] == 1]
file_name = '{}/Linear-Discriminant-Analysis-Model_predicted_test_data_plot.png'.format(path_lda)
plot_data_cluster(principal_components_test_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'Linear-Discriminant-Analysis-Model n_components=2 of (predicted) test data plot', file_name, show=False, save=True)


# ----------- linearDiscriminantAnalysis Cluster IDs plot with heighest confidence
_ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
for id in _ids:
    temp = lda_test_data.loc[lda_test_data["pId"] == id]
    max_confi = temp['Confidence'].max()
    highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
    highest_confidet = temp.at[highest_confidet_index, 'Conscientious']
    lda_test_data.loc[lda_test_data.pId == id, 'Conscientious'] = highest_confidet
	
ax2 = lda_test_data.plot.scatter(x='Conscientious',  y='pId', c=lda_test_data['Conscientious'].map(colors))
plt.show()
plt.close()

# ------- display roc_auc curve
lda_roc_auc = roc_auc_score(true_value_test_data['Conscientious'], linearDiscriminantAnalysis.predict(lda_test_x_embedded_data_frame))
fpr, tpr, thresholds = roc_curve(true_value_test_data['Conscientious'], linearDiscriminantAnalysis.predict_proba(lda_test_x_embedded_data_frame)[:,1])
file_name = '{}/Linear-Discriminant-Analysis-Model_test-data_ROC-curve.png'.format(path_lda)
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'Linear-Discriminant-Analysis-Model test data (area = %0.2f)' % lda_roc_auc, 
               title = 'Linear-Discriminant-Analysis-Model test data', file_name = file_name, show=False, save=True)


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
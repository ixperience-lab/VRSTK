import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
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
    plt.plot(false_positive_rate, true_positive_rate, label=legend_label + ' (area = %0.2f)' % knc_roc_auc)
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

# selected colums from feature selection algorithmen
selected_column_array = ['HeartRate', 'RPeaks', 'RRI', 'RRMin', 'RRMean', 'RRMax', 'SDSD', 'SD1', 'SD2', 'SDNN', 'RMSSD', 'SEllipseArea', 'VLFAbs', 'LFAbs', 'HFAbs', 'VLFLog', 'LFLog',
                         'HFLog', 'LFNorm', 'HFNorm', 'LFHFRatio', 'FBTotal', 'onsets', 'peaks', 'amps', 'RawValueInMicroSiemens', 'FilteredValueInMicroSiemens', 'AF3.theta', 'AF3.alpha', 
                         'AF3.betaL', 'AF3.betaH', 'AF3.gamma', 'F7.theta', 'F7.alpha', 'F7.betaL', 'F7.betaH', 'F7.gamma', 'F3.theta', 'F3.alpha', 'F3.betaL', 'F3.betaH', 'F3.gamma', 
                         'FC5.theta', 'FC5.alpha', 'FC5.betaL', 'FC5.betaH', 'FC5.gamma', 'T7.theta', 'T7.alpha', 'T7.betaL', 'T7.betaH', 'T7.gamma', 'P7.theta', 'P7.alpha', 'P7.betaL', 
                         'P7.betaH', 'P7.gamma', 'O1.theta', 'O1.alpha', 'O1.betaL', 'O1.betaH', 'O1.gamma', 'O2.theta', 'O2.alpha', 'O2.betaL', 'O2.betaH', 'O2.gamma', 'P8.theta', 'P8.alpha',
                         'P8.betaL', 'P8.betaH', 'P8.gamma', 'T8.theta', 'T8.alpha', 'T8.betaL', 'T8.betaH', 'T8.gamma', 'FC6.theta', 'FC6.alpha', 'FC6.betaL', 'FC6.betaH', 'FC6.gamma', 
                         'F4.theta', 'F4.alpha', 'F4.betaL', 'F4.betaH', 'F4.gamma', 'F8.theta', 'F8.alpha', 'F8.betaL', 'F8.betaH', 'F8.gamma', 'AF4.theta', 'AF4.alpha', 'AF4.betaL', 
                         'AF4.betaH', 'AF4.gamma', 'LeftPupilDiameter', 'RightPupilDiameter', 'TotalFixationCounter', 'FixationCounter', 'TotalFixationDuration', 'FixationDuration', 
                         'MeasuredVelocity', 'SaccadeCounter', 'ActivatedModelIndex', 'LASTPAGE', 'TIME_SUM', 'DEG_TIME', 'CurrentPageNumber', 'StandardDeviationStraightLineAnswer',
                         'AbsoluteDerivationOfResponseValue', 'MEDIANForTRSI', 'EvaluatedTIMERSICalc', 'LightReflexesLeftPupilDiamter', 'LeftPupilDiameterDifferenceToMean', 'LightReflexesRightPupilDiamter',
                         'RightMeanPupilDiameter', 'RightPupilDiameterDifferenceToMean', 'GlobalTIMERSICalc', 'EvaluatedGlobalTIMERSICalc', 'theta', 'alpha', 'betaL', 'betaH', 'gamma']

# ------- fitler columns of train data
train_data = input_data.drop(columns=['Conscientious', 'time', 'pId'])
#train_data = train_data[selected_column_array]

# count rows and columns
c_num = train_data.shape[1]
print(c_num)

# -------  filter columns of test data 
test_data = load_test_data.drop(columns=['time', 'pId'])
#test_data = test_data[selected_column_array]
r_num_test_data = test_data.shape[0]
test_x = test_data.iloc[:, :].values
print("================ transformend test validation input predictions informations")
true_value_test_data = []
# ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
# set real Conscientious values
for i in range(r_num_test_data):
    true_value_test_data.append(0)
    if load_test_data['pId'].values[i] == 24 or load_test_data['pId'].values[i] == 25 or load_test_data['pId'].values[i] == 28:
        true_value_test_data[i] = 1
true_value_test_data = pd.DataFrame({ "Conscientious" : true_value_test_data})      
print(true_value_test_data["Conscientious"].values)

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

# set sensor and validity score weights
weight_ecg = 2/5       #train_data.loc[:,1:26]                                 -> count() = 26
weight_eda = 3/5       #train_data.loc[:,27:31]                                -> count() = 5
weight_eeg = 1/5       #train_data.loc[:,32:107]  , train_data.loc[:,141:145]  -> count() = 76, 5
weight_eye = 2/5       #train_data.loc[:,108:117] , train_data.loc[:,130:137]  -> count() = 10, 8
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
path = "./output/T-Distributed_Stochastic_Neighbor_Embedding_{}".format(input_data_type)
#if os.path.exists(path):
#    shutil.rmtree(path, ignore_errors=True)
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
path_dbscan = "{}/Spectral-Clustering-Model".format(path)
if not os.path.exists(path_dbscan):
    os.mkdir(path_dbscan, mode)

print("------ T-Distributed Stochastic Neighbor Embedding n_components=2 of (True) train data ")
# ------ T-Distributed Stochastic Neighbor Embedding n_components=2 of train data
tsne_model = TSNE(n_components=2, learning_rate=500.0 , init='pca', perplexity=30.0)
X_embedded = tsne_model.fit_transform(x)
print(X_embedded.shape)
conscientious_indeces = input_data.index[input_data['Conscientious'] == 0]
none_conscientious_indeces = input_data.index[input_data['Conscientious'] == 1]
file_name = '{}/True_train_data_plot.png'.format(path)
plot_data_cluster(X_embedded, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'T-Distributed Stochastic Neighbor Embedding n_components=2 of (True) train data  plot', file_name, show=False, save=True)
# plt.figure(figsize=(15,10))
# ax = plt.axes(projection='3d')
# ax.scatter(X_embedded[conscientious_indeces.tolist(),0], X_embedded[conscientious_indeces.tolist(),1], X_embedded[conscientious_indeces.tolist(),2], c="b", linewidth=0.5)
# ax.scatter(X_embedded[none_conscientious_indeces.tolist(),0], X_embedded[none_conscientious_indeces.tolist(),1], X_embedded[none_conscientious_indeces.tolist(),2], c="r", linewidth=0.5)
# plt.title('T-Distributed Stochastic Neighbor Embedding n_components=3 of (True) train data  plot', fontsize=16)
# plt.show()
# plt.close()

print("------ T-Distributed Stochastic Neighbor Embedding n_components=2 of (True) test data")
# ------ T-Distributed Stochastic Neighbor Embedding n_components=2 of test data
test_x_embedded = tsne_model.fit_transform(transformed_test_x)
print(test_x_embedded.shape)
conscientious_indeces = true_value_test_data.index[true_value_test_data['Conscientious'] == 0]
none_conscientious_indeces = true_value_test_data.index[true_value_test_data['Conscientious'] == 1]
file_name = '{}/True_test_data_plot.png'.format(path)
plot_data_cluster(test_x_embedded, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'T-Distributed Stochastic Neighbor Embedding n_components=2 of (True) test data plot', file_name, show=False, save=True)
# plt.figure(figsize=(15,10))
# ax = plt.axes(projection='3d')
# ax.scatter(test_x_embedded[conscientious_indeces.tolist(),0], test_x_embedded[conscientious_indeces.tolist(),1], test_x_embedded[conscientious_indeces.tolist(),2], c="b", linewidth=0.5)
# ax.scatter(test_x_embedded[none_conscientious_indeces.tolist(),0], test_x_embedded[none_conscientious_indeces.tolist(),1], test_x_embedded[none_conscientious_indeces.tolist(),2], c="r", linewidth=0.5)
# plt.title('T-Distributed Stochastic Neighbor Embedding n_components=3 of (True) test data plot', fontsize=16)
# plt.show()
# plt.close()

print("------- K-Neighbors-Classifier-Model")
# ------- K-Neighbors-Classifier-Model
knc_x_embedded_data_frame = pd.DataFrame(data = X_embedded)
knc_train_data = train_data.copy()
# --- training (fitting)
k_neigbors_classifier = KNeighborsClassifier(n_neighbors=190, weights='uniform', algorithm='ball_tree')
k_neigbors_classifier.fit(knc_x_embedded_data_frame, y_result_output) 

# --- splitter
# splitter = ShuffleSplit(n_splits=9, train_size=0.4, test_size=0.4)
# #splitter = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=36851234)
# for train_index, test_index in splitter.split(knc_x_embedded_data_frame): # , y_result_output):
# 	print("TRAIN:", train_index, "TEST:", test_index)
# 	X_train, X_test = knc_x_embedded_data_frame.values[train_index], knc_x_embedded_data_frame.values[test_index]
# 	y_train, y_test = y_result_output[train_index], y_result_output[test_index]
# 	k_neigbors_classifier.fit(X_train, y_train)

input_score = k_neigbors_classifier.score(knc_x_embedded_data_frame, y_result_output) 
print(input_score)
# --- train data predictions 
knc_train_data["Conscientious"] = k_neigbors_classifier.predict(knc_x_embedded_data_frame) 
knc_train_data["Conscientious"] = knc_train_data["Conscientious"].astype("int")
knc_train_data["pId"] = input_data["pId"]

prediction = k_neigbors_classifier.predict_proba(knc_x_embedded_data_frame)
knc_train_data["Confidence"] = np.max(prediction, axis = 1)

plt.figure(figsize=(15,10))
plt.title('K-Neighbors-Classifier-Model train data Confidence-Histogram plot', fontsize=16)
plt.hist(knc_train_data['Confidence'][knc_train_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
plt.hist(knc_train_data['Confidence'][knc_train_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
#plt.show() 
plt.close()

print(k_neigbors_classifier.get_params(deep=True))

# get probability score of each sample
loss = log_loss(y_result_output, knc_train_data['Conscientious'])
print(loss)
print("------ K-Neighbors-Classifier-Model n_components=2 of (predicted) train data ")
conscientious_indeces = knc_train_data.index[knc_train_data['Conscientious'] == 0]
none_conscientious_indeces = knc_train_data.index[knc_train_data['Conscientious'] == 1]
file_name = '{}/K-Neighbors-Classifier-Model_predicted_train_data_plot.png'.format(path_knc)
plot_data_cluster(X_embedded, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'K-Neighbors-Classifier-Model n_components=2 of (predicted) train data plot', file_name, show=False, save=True)

# ------- display roc_auc curve
knc_roc_auc = roc_auc_score(input_data[["Conscientious"]], k_neigbors_classifier.predict(knc_x_embedded_data_frame))
fpr, tpr, thresholds = roc_curve(input_data[["Conscientious"]], k_neigbors_classifier.predict_proba(knc_x_embedded_data_frame)[:,1])
file_name = '{}/K-Neighbors-Classifier-Model_train_data_roc-curve.png'.format(path_knc)
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'K-Neighbors-Classifier-Model train data', 
               title = 'K-Neighbors-Classifier-Model train data', file_name = file_name, show=False, save=True)

# --- test data predictions 
knc_test_x_embedded_data_frame = pd.DataFrame(data = test_x_embedded)
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

plt.figure(figsize=(15,7))
plt.title('K-Neighbors-Classifier-Model test data Confidence-Histogram plot', fontsize=16)
plt.hist(knc_test_data['Confidence'][knc_test_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
plt.hist(knc_test_data['Confidence'][knc_test_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
#plt.show() 
plt.close()

print(k_neigbors_classifier.get_params(deep=True))

# get probability score of each sample
loss = log_loss(true_value_test_data['Conscientious'], knc_test_data['Conscientious'])
print(loss)
print("------ T-Distributed Stochastic Neighbor Embedding n_components=2 of (predicted) test data ")
conscientious_indeces = knc_test_data.index[knc_test_data['Conscientious'] == 0]
none_conscientious_indeces = knc_test_data.index[knc_test_data['Conscientious'] == 1]
file_name = '{}/K-Neighbors-Classifier-Model_predicted_test_data_plot.png'.format(path_knc)
plot_data_cluster(test_x_embedded, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'K-Neighbors-Classifier-Model n_components=2 of (predicted) test data plot', file_name, show=False, save=True)

# ------- display roc_auc curve
knc_roc_auc = roc_auc_score(true_value_test_data['Conscientious'], k_neigbors_classifier.predict(knc_test_x_embedded_data_frame))
fpr, tpr, thresholds = roc_curve(true_value_test_data['Conscientious'], k_neigbors_classifier.predict_proba(knc_test_x_embedded_data_frame)[:,1])
file_name = '{}/K-Neighbors-Classifier-Model_test_data_roc-curve.png'.format(path_knc)
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'K-Neighbors-Classifier-Model test data', 
               title = 'K-Neighbors-Classifier-Model test data', file_name = file_name, show=False, save=True)

sys.exit()

print("------- Mini-Batch-K-Means Model")
# ------- Mini-Batch-K-Means Model
mbkm_x_embedded_data_frame = pd.DataFrame(data = X_embedded)
mbkm_train_data = train_data.copy()
miniBatchKMeans = MiniBatchKMeans(init="k-means++", n_clusters=2).fit(mbkm_x_embedded_data_frame) #miniBatchKMeans = MiniBatchKMeans(n_clusters=2).fit(input_x)
input_score = miniBatchKMeans.score(mbkm_x_embedded_data_frame) #input_score = miniBatchKMeans.score(np.array(input_x)[0:1])
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

plt.figure(figsize=(15,7))
plt.title('Mini-Batch-K-Means-Model Confidence-Histogram plot', fontsize=16)
plt.hist(df['confidence'][mbkm_train_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
plt.hist(df['confidence'][mbkm_train_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 
plt.close()

print(miniBatchKMeans.get_params(deep=True))
print(miniBatchKMeans.labels_)

print("------ Mini-Batch-K-Means-Model n_components=2 of (predicted) train data ")
conscientious_indeces = mbkm_train_data.index[mbkm_train_data['Conscientious'] == 0]
none_conscientious_indeces = mbkm_train_data.index[mbkm_train_data['Conscientious'] == 1]
file_name = '{}/Mini-Batch-K-Means-Model_predicted_train_data_plot.png'.format(path_mbkm)
plot_data_cluster(X_embedded, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
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
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'Mini-Batch-K-Means-Model train data', 
               title = 'Mini-Batch-K-Means-Model train data', file_name = file_name, show=False, save=True)

print(" --- test data mbkm")
# --- test data
mbkm_test_x_embedded_data_frame = pd.DataFrame(data = test_x_embedded)
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

plt.figure(figsize=(15,7))
plt.title('Mini-Batch-K-Means-Model Confidence-Histogram plot', fontsize=16)
plt.hist(df['confidence'][mbkm_test_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
plt.hist(df['confidence'][mbkm_test_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 
plt.close()

print(miniBatchKMeans.get_params(deep=True))
print(miniBatchKMeans.labels_)

print("------ Mini-Batch-K-Means-Model n_components=2 of (predicted) test data ")
conscientious_indeces = mbkm_test_data.index[mbkm_test_data['Conscientious'] == 0]
none_conscientious_indeces = mbkm_test_data.index[mbkm_test_data['Conscientious'] == 1]
file_name = '{}/Mini-Batch-K-Means-Model_predicted_test_data_plot.png'.format(path_mbkm)
plot_data_cluster(test_x_embedded, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'Mini-Batch-K-Means-Model n_components=2 of (predicted) test data plot', file_name, show=False, save=True)
# plt.figure(figsize=(15,10))
# plt.scatter(test_x_embedded[conscientious_indeces,0], test_x_embedded[conscientious_indeces,1], c="b")
# plt.scatter(test_x_embedded[none_conscientious_indeces,0], test_x_embedded[none_conscientious_indeces,1], c="r")
# plt.title('Mini-Batch-K-Means-Model n_components=2 of (predicted) test data plot', fontsize=16)
# file_name = '{}/Mini-Batch-K-Means-Model_predicted_test_data_plot.png'.format(path_mbkm)
# plt.savefig(file_name)
# #plt.show()
# plt.close()

# get probability score of each sample
loss = log_loss(true_value_test_data['Conscientious'], mbkm_test_data['Conscientious'])
print(loss)

input_means_labels = pairwise_distances_argmin(mbkm_test_x_embedded_data_frame[:].values.copy(order='C'), input_cluster_centers_.copy(order='C'))
print(input_means_labels)

# # ------- display roc_auc curve
mbkm_roc_auc = roc_auc_score(true_value_test_data['Conscientious'], miniBatchKMeans.predict(knc_test_x_embedded_data_frame))
fpr, tpr, thresholds = roc_curve(true_value_test_data['Conscientious'],  mbkm_test_data["Confidence"])
file_name = '{}/Mini-Batch-K-Means-Model_test-data_ROC_curve.png'.format(path_mbkm)
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'Mini-Batch-K-Means-Model test data', 
               title = 'Mini-Batch-K-Means-Model test data', file_name = file_name, show=False, save=True)

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

print("--- training data gmm")
# --- training data
gaussian_train_data["Conscientious"] = gaussianMixture.predict(gaussian_x_embedded_data_frame)
gaussian_train_data["Conscientious"] = gaussian_train_data["Conscientious"].astype("int")

prediction=gaussianMixture.predict_proba(gaussian_x_embedded_data_frame)#[:,1]
print(prediction)
gaussian_train_data["Confidence"] = np.max(prediction, axis = 1)

plt.figure(figsize=(15,7))
plt.title('Gaussian-Mixtures-Model Confidence-Histogram plot', fontsize=16)
plt.hist(gaussian_train_data['Confidence'][gaussian_train_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
plt.hist(gaussian_train_data['Confidence'][gaussian_train_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 
plt.close()

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
plot_data_cluster(X_embedded, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
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
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'Gaussian-Mixtures-Model training data', 
               title = 'Gaussian-Mixtures-Model training data', file_name = file_name, show=False, save=True)

print("--- test data gmm")
# --- test data
gmm_test_x_embedded_data_frame = pd.DataFrame(data = test_x_embedded)
gmm_test_data = test_data.copy()
gmm_test_data["Conscientious"] = gaussianMixture.predict(gmm_test_x_embedded_data_frame)
gmm_test_data["Conscientious"] = gmm_test_data["Conscientious"].astype("int")

prediction=gaussianMixture.predict_proba(gmm_test_x_embedded_data_frame)#[:,1]
print(prediction)
gmm_test_data["Confidence"] = np.max(prediction, axis = 1)

plt.figure(figsize=(15,7))
plt.title('Gaussian-Mixtures-Model PCA Confidence-Histogram plot', fontsize=16)
plt.hist(gmm_test_data['Confidence'][gmm_test_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.5, color='b')
plt.hist(gmm_test_data['Confidence'][gmm_test_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.5, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 
plt.close()

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
plot_data_cluster(test_x_embedded, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
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
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'Gaussian-Mixtures-Model test data', 
               title = 'Gaussian-Mixtures-Model test data', file_name = file_name, show=False, save=True)

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
plt.close()

print(linearDiscriminantAnalysis.get_params(deep=True))

print(" ------ Linear-Discriminant-Analysis-Model n_components=2 of (predicted) train data")
conscientious_indeces = lda_train_data.index[lda_train_data['Conscientious'] == 0]
none_conscientious_indeces = lda_train_data.index[lda_train_data['Conscientious'] == 1]
file_name = '{}/Linear-Discriminant-Analysis-Model_predicted_train_data_plot.png'.format(path_lda)
plot_data_cluster(X_embedded, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
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
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'Linear-Discriminant-Analysis-Model train data', 
               title = 'Linear-Discriminant-Analysis-Model train data', file_name = file_name, show=False, save=True)

print("================ transformend test validation input predictions informations")
true_value_test_data = []
#test_data['pId'] = load_test_data['pId']
#ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
# set real Conscientious values
for i in range(r_num_test_data):
    true_value_test_data.append([0])
    if load_test_data['pId'].values[i] == 24 or load_test_data['pId'].values[i] == 25: #or load_test_data['pId'].values[i] == 29:
        true_value_test_data[i] = [1]

lda_test_x_embedded_data_frame = pd.DataFrame(data = test_x_embedded)
lda_test_data = test_data.copy()

result_array = linearDiscriminantAnalysis.predict(lda_test_x_embedded_data_frame)
lda_test_data["Conscientious"] = result_array
lda_test_data["Conscientious"] = lda_test_data["Conscientious"].astype("int")

prediction = linearDiscriminantAnalysis.predict_proba(lda_test_x_embedded_data_frame)
lda_test_data["Confidence"] = np.max(prediction, axis = 1)

plt.figure(figsize=(15,7))
plt.hist(lda_test_data['Confidence'][lda_test_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.7, color='b')
plt.hist(lda_test_data['Confidence'][lda_test_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.7, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 
plt.close()

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
plot_data_cluster(test_x_embedded, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
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
lda_roc_auc = roc_auc_score(true_value_test_data, linearDiscriminantAnalysis.predict(lda_test_x_embedded_data_frame))
fpr, tpr, thresholds = roc_curve(true_value_test_data, linearDiscriminantAnalysis.predict_proba(lda_test_x_embedded_data_frame)[:,1])
file_name = '{}/Linear-Discriminant-Analysis-Model_test-data_ROC-curve.png'.format(path_lda)
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'Linear-Discriminant-Analysis-Model test data', 
               title = 'Linear-Discriminant-Analysis-Model test data', file_name = file_name, show=False, save=True)


print(" --- SpectralClustering-Model")
# --- SpectralClustering-Model
dbscan_x_embedded_data_frame = pd.DataFrame(data = X_embedded)
dbscan_train_data = train_data.copy()
spectral_clustring_model = SpectralClustering(n_clusters=2, assign_labels='cluster_qr', affinity='nearest_neighbors')
print(" --- created cluster")
spectral_clustring_model.fit(dbscan_x_embedded_data_frame)
print(" --- created cluster")
print(spectral_clustring_model.labels_)
print(" --- created cluster")

dbscan_train_data["Conscientious"] = spectral_clustring_model.fit_predict(dbscan_x_embedded_data_frame)
dbscan_train_data["Conscientious"] = dbscan_train_data["Conscientious"].astype("int")

dbscan_train_data[dbscan_train_data["Conscientious"] > 0] = 1
dbscan_train_data[dbscan_train_data["Conscientious"] == -1] = 0

#print(dbscan_train_data["Conscientious"].values.tolist())

print(spectral_clustring_model.get_params(deep=True))
dbscan_train_data["pId"] = input_data["pId"]

colors = {0:'b', 1:'r'}
plt.scatter(x=dbscan_train_data['Conscientious'], y=dbscan_train_data['pId'], alpha=0.5, c=dbscan_train_data['Conscientious'].map(colors))
plt.show()
plt.close()

	
ax2 = dbscan_train_data.plot.scatter(x='Conscientious',  y='pId', c=dbscan_train_data['Conscientious'].map(colors))
plt.show()
plt.close()

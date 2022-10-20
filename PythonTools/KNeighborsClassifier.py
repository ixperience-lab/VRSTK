from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
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

# exc_cols = [col for col in train_data.columns if col not in ['DegTimeLowQuality', 'EvaluatedGlobalTIMERSICalc']]
# train_data.loc[train_data.DegTimeLowQuality > 0, exc_cols] *= 1.5
# train_data.loc[train_data.EvaluatedGlobalTIMERSICalc >= 1, exc_cols] *= 1.5

# count rows and columns
c_num = train_data.shape[1]
print(c_num)

# -------  filter columns of test data 
test_data = load_test_data.drop(columns=['time', 'pId'])
#test_data = test_data[selected_column_array]

# exc_cols = [col for col in test_data.columns if col not in ['DegTimeLowQuality', 'EvaluatedGlobalTIMERSICalc']]
# test_data.loc[test_data.DegTimeLowQuality > 0, exc_cols] *= 1.5
# test_data.loc[test_data.EvaluatedGlobalTIMERSICalc >= 1, exc_cols] *= 1.5

r_num_test_data = test_data.shape[0]
test_x = test_data.iloc[:, :].values
print("================ transformend test validation input predictions informations")
true_value_test_data = []
# ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
# set real Conscientious values
for i in range(r_num_test_data):
    true_value_test_data.append(0)
    if load_test_data['pId'].values[i] == 24 or load_test_data['pId'].values[i] == 25: # or load_test_data['pId'].values[i] == 28:
        true_value_test_data[i] = 1
true_value_test_data = pd.DataFrame({ "Conscientious" : true_value_test_data})      
print(true_value_test_data["Conscientious"].values)

# ------ Normalizing
# Separating out the features
x_train = train_data.loc[:, :].values
# Separating out the target
y_result_output = np.array(input_data[["Conscientious"]].values.flatten())
print(y_result_output)
# Standardizing the features of train data
transformed_train_x = StandardScaler().fit_transform(x_train)
# Standardizing the features of Test data
transformed_test_x = StandardScaler().fit_transform(test_x)

# set sensor and validity score weights
weight_ecg = 2/5       #train_data.loc[:,1:26]                                 -> count() = 26
weight_eda = 3/5       #train_data.loc[:,27:31]                                -> count() = 5
weight_eeg = 1/5       #train_data.loc[:,32:107]  , train_data.loc[:,141:145]  -> count() = 76, 5
weight_eye = 3/5       #train_data.loc[:,108:117] , train_data.loc[:,130:137]  -> count() = 10, 8
weight_pages = 1       #train_data.loc[:,118:129] , train_data.loc[:,138:140]  -> count() = 12, 3

if input_data_type == 0:
    transformed_train_x[:,0:26]    = transformed_train_x[:,0:26]    * weight_ecg
    transformed_train_x[:,26:31]   = transformed_train_x[:,26:31]   * weight_eda
    transformed_train_x[:,31:107]  = transformed_train_x[:,31:107]  * weight_eeg
    transformed_train_x[:,152:157] = transformed_train_x[:,152:157] * weight_eeg
    transformed_train_x[:,107:129] = transformed_train_x[:,107:129] * weight_eye
    transformed_train_x[:,141:149] = transformed_train_x[:,141:149] * weight_eye
    transformed_train_x[:,129:141] = transformed_train_x[:,129:141] * weight_pages
    transformed_train_x[:,149:152] = transformed_train_x[:,149:152] * weight_pages
    
    transformed_test_x[:,0:26]    = transformed_test_x[:,0:26]    * weight_ecg
    transformed_test_x[:,26:31]   = transformed_test_x[:,26:31]   * weight_eda
    transformed_test_x[:,31:107]  = transformed_test_x[:,31:107]  * weight_eeg
    transformed_test_x[:,152:157] = transformed_test_x[:,152:157] * weight_eeg
    transformed_test_x[:,107:129] = transformed_test_x[:,107:129] * weight_eye
    transformed_test_x[:,141:149] = transformed_test_x[:,141:149] * weight_eye
    transformed_test_x[:,129:141] = transformed_test_x[:,129:141] * weight_pages
    transformed_test_x[:,149:152] = transformed_test_x[:,149:152] * weight_pages
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
path = "./output/K-Neighbors-Classifier-Model_{}".format(input_data_type)
if not os.path.exists(path):
    os.mkdir(path, mode)


print("------ Transformed (True) train data")
# ------ Transformed (True) train data
conscientious_indeces = input_data.index[input_data['Conscientious'] == 0]
none_conscientious_indeces = input_data.index[input_data['Conscientious'] == 1]
file_name = '{}/Transformed_train_data_plot.png'.format(path)
plot_data_cluster(transformed_train_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'Transformed (True) train data  plot', file_name, show=False, save=True)

print("------ Transformed (True) test data")
# ------ Transformed (True) test data
conscientious_indeces = true_value_test_data.index[true_value_test_data['Conscientious'] == 0]
none_conscientious_indeces = true_value_test_data.index[true_value_test_data['Conscientious'] == 1]
file_name = '{}/Transformed_True_test_data_plot.png'.format(path)
plot_data_cluster(transformed_test_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'Transformed (True) test data (True) test data plot', file_name, show=False, save=True)

print("------- K-Neighbors-Classifier-Model")
# ------- K-Neighbors-Classifier-Model
knc_x_embedded_data_frame = pd.DataFrame(data = transformed_train_x)
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

print(k_neigbors_classifier.get_params(deep=True))

# get probability score of each sample
loss = log_loss(y_result_output, knc_train_data['Conscientious'])
print(loss)
print("------ K-Neighbors-Classifier-Model n_components=2 of (predicted) train data ")
conscientious_indeces = knc_train_data.index[knc_train_data['Conscientious'] == 0]
none_conscientious_indeces = knc_train_data.index[knc_train_data['Conscientious'] == 1]
file_name = '{}/K-Neighbors-Classifier-Model_predicted_train_data_plot.png'.format(path)
plot_data_cluster(transformed_train_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'K-Neighbors-Classifier-Model n_components=2 of (predicted) train data plot', file_name, show=False, save=True)

# ------- display roc_auc curve
knc_roc_auc = roc_auc_score(input_data[["Conscientious"]], k_neigbors_classifier.predict(knc_x_embedded_data_frame))
fpr, tpr, thresholds = roc_curve(input_data[["Conscientious"]], k_neigbors_classifier.predict_proba(knc_x_embedded_data_frame)[:,1])
file_name = '{}/K-Neighbors-Classifier-Model_train_data_roc-curve.png'.format(path)
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'K-Neighbors-Classifier-Model train data (area = %0.2f)' % knc_roc_auc, 
               title = 'K-Neighbors-Classifier-Model train data', file_name = file_name, show=False, save=True)

# --- test data predictions 
knc_test_x_embedded_data_frame = pd.DataFrame(data = transformed_test_x)
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
	
# ax2 = knc_test_data.plot.scatter(x='Conscientious',  y='pId', c=knc_test_data['Conscientious'].map(colors))
# plt.show()
# plt.close()

print(k_neigbors_classifier.get_params(deep=True))
print(accuracy_score(true_value_test_data['Conscientious'], knc_test_data['Conscientious']))
input_score = k_neigbors_classifier.score(knc_test_x_embedded_data_frame,  true_value_test_data['Conscientious']) 
print(input_score)
# get probability score of each sample
loss = log_loss(true_value_test_data['Conscientious'], knc_test_data['Conscientious'])
print(loss)
print("------ K-Neighbors-Classifier-Model n_components=2 of (predicted) test data ")
conscientious_indeces = knc_test_data.index[knc_test_data['Conscientious'] == 0]
none_conscientious_indeces = knc_test_data.index[knc_test_data['Conscientious'] == 1]
file_name = '{}/K-Neighbors-Classifier-Model_predicted_test_data_plot.png'.format(path)
plot_data_cluster(transformed_test_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                 'K-Neighbors-Classifier-Model n_components=2 of (predicted) test data plot', file_name, show=False, save=True)

# ------- display roc_auc curve
knc_roc_auc = roc_auc_score(true_value_test_data['Conscientious'], k_neigbors_classifier.predict(knc_test_x_embedded_data_frame))
fpr, tpr, thresholds = roc_curve(true_value_test_data['Conscientious'], k_neigbors_classifier.predict_proba(knc_test_x_embedded_data_frame)[:,1])
file_name = '{}/K-Neighbors-Classifier-Model_test_data_roc-curve.png'.format(path)
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'K-Neighbors-Classifier-Model test data (area = %0.2f)' % knc_roc_auc,
               title = 'K-Neighbors-Classifier-Model test data', file_name = file_name, show=False, save=True)


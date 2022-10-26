from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import precision_recall_curve, log_loss, accuracy_score, f1_score, roc_auc_score, roc_curve 
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.utils import np_utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#from xgboost import XGBRegressor
#from catboost import CatBoostRegressor, Pool
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.augmentations import ClassificationSMOTE
import torch

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

# method to random shuffle a data frame
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop('Conscientious')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

# input_data_type = { all_sensors = 0, ecg = 1, eda = 2, eeg = 3, eye = 4, pages = 5 }
input_data_type = 0

# read csv train data as pandas data frame
input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrameKopie.csv", sep=";", decimal=',')			# plan of sensors weighting:
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

# # updates Conscientious to subjektive 
# for i in range(input_data.shape[1]):
#     if input_data['pId'].values[i] == 14 or input_data['pId'].values[i] == 15 or input_data['pId'].values[i] == 16: # or load_test_data['pId'].values[i] == 28:
#         input_data['Conscientious'].values[i] = 1

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
#scaler = MinMaxScaler(feature_range=(0, 1))
#transformed_train_x = scaler.fit_transform(x_train)
# Standardizing the features of Test data
#transformed_test_x = scaler.fit_transform(test_x)

transformed_train_x = StandardScaler().fit_transform(x_train)
# Standardizing the features of Test data
transformed_test_x = StandardScaler().fit_transform(test_x)

# # set sensor and validity score weights
weight_ecg = 1       #train_data.loc[:,1:26]                                 -> count() = 26
weight_eda = 1       #train_data.loc[:,27:31]                                -> count() = 5
weight_eeg = 1       #train_data.loc[:,32:107]  , train_data.loc[:,141:145]  -> count() = 76, 5
weight_eye = 1       #train_data.loc[:,108:117] , train_data.loc[:,130:137]  -> count() = 10, 8
weight_pages = 1       #train_data.loc[:,118:129] , train_data.loc[:,138:140]  -> count() = 12, 3

# weight_ecg = 1/5       #train_data.loc[:,1:26]                                 -> count() = 26
# weight_eda = 2/5       #train_data.loc[:,27:31]                                -> count() = 5
# weight_eeg = 1/5       #train_data.loc[:,32:107]  , train_data.loc[:,141:145]  -> count() = 76, 5
# weight_eye = 1/5       #train_data.loc[:,108:117] , train_data.loc[:,130:137]  -> count() = 10, 8
# weight_pages = 2       #train_data.loc[:,118:129] , train_data.loc[:,138:140]  -> count() = 12, 3

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

# if input_data_type == 1:
# 	transformed_train_x[:,:] = transformed_train_x[:,:] * weight_ecg
# 	transformed_test_x[:,:]  = transformed_test_x[:,:]  * weight_ecg
# if input_data_type == 2:
# 	transformed_train_x[:,:] = transformed_train_x[:,:] * weight_eda
# 	transformed_test_x[:,:]  = transformed_test_x[:,:]  * weight_eda
# if input_data_type == 3:
# 	transformed_train_x[:,:] = transformed_train_x[:,:] * weight_eeg
# 	transformed_test_x[:,:]  = transformed_test_x[:,:]  * weight_eeg
# if input_data_type == 4:
# 	transformed_train_x[:,:] = transformed_train_x[:,:] * weight_eye
# 	transformed_test_x[:,:]  = transformed_test_x[:,:]  * weight_eye
# if input_data_type == 5:
# 	transformed_train_x[:,:] = transformed_train_x[:,:] * weight_pages
# 	transformed_test_x[:,:]  = transformed_test_x[:,:]  * weight_pages

print("Create output directory")
# --- create dir
mode = 0o666
if not os.path.exists("./output"):
    os.mkdir("./output", mode)
path = "./output/DeepLearning-Model_{}".format(input_data_type)
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

# print("------ T-Distributed Stochastic Neighbor Embedding n_components=2 of (True) train data ")
# # ------ T-Distributed Stochastic Neighbor Embedding n_components=2 of train data
# tsne_model = TSNE(n_components=2, learning_rate=500.0 , init='pca', perplexity=30.0)
# transformed_train_x = tsne_model.fit_transform(transformed_train_x)
# print(transformed_train_x.shape)
# conscientious_indeces = input_data.index[input_data['Conscientious'] == 0]
# none_conscientious_indeces = input_data.index[input_data['Conscientious'] == 1]
# file_name = '{}/tsne_True_train_data_plot.png'.format(path)
# plot_data_cluster(transformed_train_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
#                  'T-Distributed Stochastic Neighbor Embedding n_components=2 of (True) train data  plot', file_name, show=False, save=True)

# print("------ T-Distributed Stochastic Neighbor Embedding n_components=2 of (True) test data")
# # ------ T-Distributed Stochastic Neighbor Embedding n_components=2 of test data
# transformed_test_x = tsne_model.fit_transform(transformed_test_x)
# print(transformed_test_x.shape)
# conscientious_indeces = true_value_test_data.index[true_value_test_data['Conscientious'] == 0]
# none_conscientious_indeces = true_value_test_data.index[true_value_test_data['Conscientious'] == 1]
# file_name = '{}/tsne_True_test_data_plot.png'.format(path)
# plot_data_cluster(transformed_test_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
#                  'T-Distributed Stochastic Neighbor Embedding n_components=2 of (True) test data plot', file_name, show=False, save=True)

# print("------ Principal Component Analysis n_components=2 of train data")
# # ------ Principal Component Analysis n_components=2 of train data
# pca = PCA(n_components=2)
# transformed_train_x = pca.fit_transform(transformed_train_x)
# #print(pca.score(x)) # Debug only
# print(pca.explained_variance_ratio_)  # Debug only

# conscientious_indeces = input_data.index[input_data['Conscientious'] == 0]
# none_conscientious_indeces = input_data.index[input_data['Conscientious'] == 1]
# plt.figure(figsize=(15,10))
# plt.scatter(transformed_train_x[conscientious_indeces.tolist(),0], transformed_train_x[conscientious_indeces.tolist(),1], c="b")
# plt.scatter(transformed_train_x[none_conscientious_indeces.tolist(),0], transformed_train_x[none_conscientious_indeces.tolist(),1], c="r")
# plt.title('Principal Component Analysis train data n_components=2 plot', fontsize=16)
# file_name = '{}/True_principal_components_train_data_plot.png'.format(path)
# plt.savefig(file_name)
# plt.close()


# transformed_test_x = pca.fit_transform(transformed_test_x)
# #print(pca.score(transformed_test_x)) # Debug only
# print(pca.explained_variance_ratio_)  # Debug only

# conscientious_indeces = true_value_test_data.index[true_value_test_data['Conscientious'] == 0]
# none_conscientious_indeces = true_value_test_data.index[true_value_test_data['Conscientious'] == 1]
# plt.figure(figsize=(15,10))
# plt.scatter(transformed_test_x[conscientious_indeces.tolist(),0], transformed_test_x[conscientious_indeces.tolist(),1], c="b")
# plt.scatter(transformed_test_x[none_conscientious_indeces.tolist(),0], transformed_test_x[none_conscientious_indeces.tolist(),1], c="r")
# plt.title('Principal Component Analysis test data n_components=2 plot', fontsize=16)
# file_name = '{}/True_principal_components_test_data_plot.png'.format(path)
# plt.savefig(file_name)
# plt.close()

# ---- creates tensorflow tensors as shuffled datasets
# shuffled_train_dataset = dataframe_to_dataset(train_dataframe) 
# shuffled_validation_dataset = dataframe_to_dataset(validation_dataframe)

print("------- -Model")
# ------- -Model
x_train_data_frame = pd.DataFrame(data = transformed_train_x)
#x_train_data_frame['Conscientious'] = input_data['Conscientious']
_train_data = train_data.copy()

# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
# train_index, test_index = sss.split(x_train_data_frame, np.array(input_data["Conscientious"].values.flatten()))
# train_dataframe, validation_dataframe = transformed_train_x[train_index], transformed_train_x[test_index]
# y_train_true_output, y_validation_true_output = np.array(input_data["Conscientious"].values.flatten())[train_index], np.array(input_data["Conscientious"].values.flatten())[test_index]

train_dataframe, validation_dataframe, y_train_true_output, y_validation_true_output = train_test_split(transformed_train_x, 
                                    np.array(input_data["Conscientious"].values.flatten()), test_size=0.4,  shuffle=True, 
                                    stratify=np.array(input_data["Conscientious"].values.flatten()), random_state=42)

#validation_dataframe = x_train_data_frame.sample(frac=0.5, random_state=1337)
#train_dataframe = x_train_data_frame.drop(validation_dataframe.index)
#y_train_true_output = np.array(input_data.Conscientious.drop(validation_dataframe.index).values.flatten())
#y_validation_true_output = np.array(input_data.Conscientious.values[validation_dataframe.index].flatten())
print(y_train_true_output)
print(y_validation_true_output)
print("Using %d samples for training and %d for validation"  % (len(train_dataframe), len(validation_dataframe)))

x_test_data_frame = pd.DataFrame(data=transformed_test_x)
_test_data_copy = test_data.copy()

# For Keras, convert dataframe to array values (Inbuilt requirement of Keras)
X = train_dataframe.astype('float32')#.to_numpy(dtype='float32')
print(X.shape)
Y = y_train_true_output
#Y = np_utils.to_categorical(Y, num_classes=2)
print(Y.shape)

v_X = validation_dataframe.astype('float32')#.to_numpy(dtype='float32')
print(v_X.shape)
v_Y = y_validation_true_output
#v_Y = np_utils.to_categorical(v_Y, num_classes=2)
print(v_Y.shape)
#sys.exit()

t_X = x_test_data_frame.to_numpy(dtype='float32')
t_Y = np.array(true_value_test_data["Conscientious"].values.flatten())
#t_Y = np_utils.to_categorical(t_Y, num_classes=2)
print(t_X.shape)
print(t_Y.shape)

# ---- hyper parameters
learning_rate = 0.001
dropout_rate  = 0.1
batch_size    = 64
num_epochs    = 100
#19 # adam-op
num_classes   = 1

# # model creation
# model = Sequential()
# model.add(Flatten(input_dim = X.shape[1]))
# model.add(BatchNormalization())
# model.add(Dropout(dropout_rate))
# model.add(Dense(2048, activation = 'relu'))
# model.add(BatchNormalization())
# model.add(Dropout(dropout_rate))
# model.add(Dense(2048, activation = 'relu'))
# model.add(BatchNormalization())
# model.add(Dropout(dropout_rate))
# model.add(Dense(2048, activation = 'relu'))
# #model.add(Dense(2048, input_dim = X.shape[1], activation = 'relu')) # input layer requires input_dim param
# model.add(BatchNormalization())
# model.add(Dropout(dropout_rate))
# #model.add(Dense(32, activation = 'sigmoid'))
# #model.add(BatchNormalization())
# #model.add(Dropout(dropout_rate))
# #model.add(Dense(8, activation = 'relu'))
# #model.add(BatchNormalization())
# #model.add(Dense(256, activation = 'relu'))
# #model.add(BatchNormalization())
# #model.add(Dropout(dropout_rate))
# #model.add(Dense(512, activation = 'relu'))
# #model.add(BatchNormalization())
# #model.add(Dropout(dropout_rate))
# #model.add(Dense(256, activation = 'relu'))
# #model.add(BatchNormalization())
# #model.add(Dropout(dropout_rate))
# #model.add(Dense(128, activation = 'relu'))
# #model.add(BatchNormalization())
# #model.add(Dropout(dropout_rate))
# #model.add(Dense(64, activation = 'sigmoid'))
# #model.add(BatchNormalization())
# #model.add(Flatten())
# model.add(Dense(num_classes, activation='sigmoid'))#activation='softmax'))

# #optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
# # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
# #     initial_learning_rate=learning_rate,
# #     decay_steps=1000,
# #     decay_rate=0.9)
# # optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
# optimizer = keras.optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
# #model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
# model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

# # model training
# history = model.fit(X, Y, epochs=num_epochs, shuffle=True, batch_size=batch_size, verbose=2,  validation_data = (v_X, v_Y))

# # list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# file_name = '{}/model_history_accuracy_plot.png'.format(path)
# plt.savefig(file_name)
# #plt.show()
# plt.close()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# file_name = '{}/model_history_loss_plot.png'.format(path)
# plt.savefig(file_name)
# #plt.show()
# plt.close()

# loss, acc =  model.evaluate(v_X, v_Y)
# print("model loss: %.2f, acc: (%.2f%%)" % (loss*100, acc*100))

# predictions =  model.predict(t_X)
# # print(predictions)
# # print(tf.argmax(predictions, axis=-1).numpy())
# # predictions_transformed = tf.argmax(predictions, axis=-1).numpy()
# predictions_transformed = []
# for i, predicted in enumerate(predictions):
#     if predicted[0] > 0.5:
#         predictions_transformed.append(1)
#     else:
#         predictions_transformed.append(0)

# loss, acc =  model.evaluate(t_X, t_Y)
# print("model loss: %.2f, acc: (%.2f%%)" % (loss*100, acc*100))
# _test_data_copy['Conscientious'] = predictions_transformed

# conscientious_indeces = _test_data_copy.index[_test_data_copy['Conscientious'] == 0]
# none_conscientious_indeces = _test_data_copy.index[_test_data_copy['Conscientious'] == 1]
# file_name = '{}/Predicted_test_data_plot.png'.format(path)
# plot_data_cluster(transformed_test_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
#               'Predicted test data plot', file_name, show=False, save=True)

# # ------- display roc_auc curve
# lda_roc_auc = roc_auc_score(true_value_test_data["Conscientious"], predictions_transformed)
# fpr, tpr, thresholds = roc_curve(true_value_test_data["Conscientious"], predictions)#predictions[:,1])
# file_name = '{}/DL-Model_test-data_ROC-curve.png'.format(path)
# plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'DL-Model test data auc (area = %0.2f)' % lda_roc_auc, 
#                title = 'DL-Model test data', file_name = file_name, show=False, save=True)

# precision, recall, thresholds = precision_recall_curve(true_value_test_data["Conscientious"], predictions_transformed)
# print(precision)
# print(recall)
# print(thresholds)

# f1_score_value = f1_score(true_value_test_data["Conscientious"], predictions_transformed, average=None)
# print(f1_score_value)

# display = PrecisionRecallDisplay.from_predictions(true_value_test_data["Conscientious"], predictions_transformed, name="DL-Model")
# _ = display.ax_.set_title("2-class Precision-Recall curve")
# file_name = '{}/DL-Model_test-data_Precision-Recall-curve.png'.format(path)
# plt.savefig(file_name)
# #plt.show()
# plt.close()



# # ---- xgboost model
# my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# my_model.fit(X, Y, eval_set=[(v_X, v_Y)], verbose=False)

# predictions =  my_model.predict(t_X)
# print(predictions)

# predictions_transformed = []
# for i, predicted in enumerate(predictions):
#     print(predicted)
#     if predicted > 0.5:
#         predictions_transformed.append(1)
#     else:
#         predictions_transformed.append(0)

# _test_data_copy['Conscientious'] = predictions_transformed
# conscientious_indeces = _test_data_copy.index[_test_data_copy['Conscientious'] == 0]
# none_conscientious_indeces = _test_data_copy.index[_test_data_copy['Conscientious'] == 1]
# file_name = '{}/xgboost_Predicted_test_data_plot.png'.format(path)
# plot_data_cluster(transformed_test_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
#               'Predicted test data plot', file_name, show=False, save=True)

# # ------- display roc_auc curve
# lda_roc_auc = roc_auc_score(true_value_test_data["Conscientious"], predictions_transformed)
# fpr, tpr, thresholds = roc_curve(true_value_test_data["Conscientious"], predictions)#predictions[:,1])
# file_name = '{}/xgboost_DL-Model_test-data_ROC-curve.png'.format(path)
# plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'DL-Model test data auc (area = %0.2f)' % lda_roc_auc, 
#                title = 'DL-Model test data', file_name = file_name, show=False, save=True)

# precision, recall, thresholds = precision_recall_curve(true_value_test_data["Conscientious"], predictions_transformed)
# print(precision)
# print(recall)
# print(thresholds)

# f1_score_value = f1_score(true_value_test_data["Conscientious"], predictions_transformed, average=None)
# print(f1_score_value)

# display = PrecisionRecallDisplay.from_predictions(true_value_test_data["Conscientious"], predictions_transformed, name="DL-Model")
# _ = display.ax_.set_title("2-class Precision-Recall curve")
# file_name = '{}/xgboost_DL-Model_test-data_Precision-Recall-curve.png'.format(path)
# plt.savefig(file_name)
# plt.close()


# ---- catboost model
# PARAMS_CATBOOST_REGRESSOR = dict()
# PARAMS_CATBOOST_REGRESSOR['learning_rate']=0.1
# PARAMS_CATBOOST_REGRESSOR['use_best_model']= True
# PARAMS_CATBOOST_REGRESSOR['logging_level'] = 'Silent'
# PARAMS_CATBOOST_REGRESSOR['l2_leaf_reg'] = 1.0 # lambda, default 3, S: 300
# clf = CatBoostRegressor(**PARAMS_CATBOOST_REGRESSOR)
# clf.fit([(X, Y)], use_best_model=True, eval_set=[(v_X, v_Y)])

# predictions =  clf.predict(t_X)
# print(predictions)

# predictions_transformed = []
# for i, predicted in enumerate(predictions):
#     print(predicted)
#     if predicted > 0.25:
#         predictions_transformed.append(1)
#     else:
#         predictions_transformed.append(0)

# _test_data_copy['Conscientious'] = predictions_transformed
# conscientious_indeces = _test_data_copy.index[_test_data_copy['Conscientious'] == 0]
# none_conscientious_indeces = _test_data_copy.index[_test_data_copy['Conscientious'] == 1]
# file_name = '{}/xgboost_Predicted_test_data_plot.png'.format(path)
# plot_data_cluster(transformed_test_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
#               'Predicted test data plot', file_name, show=False, save=True)

# # ------- display roc_auc curve
# lda_roc_auc = roc_auc_score(true_value_test_data["Conscientious"], predictions_transformed)
# fpr, tpr, thresholds = roc_curve(true_value_test_data["Conscientious"], predictions)#predictions[:,1])
# file_name = '{}/xgboost_DL-Model_test-data_ROC-curve.png'.format(path)
# plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'DL-Model test data auc (area = %0.2f)' % lda_roc_auc, 
#                title = 'DL-Model test data', file_name = file_name, show=False, save=True)

# precision, recall, thresholds = precision_recall_curve(true_value_test_data["Conscientious"], predictions_transformed)
# print(precision)
# print(recall)
# print(thresholds)

# f1_score_value = f1_score(true_value_test_data["Conscientious"], predictions_transformed, average=None)
# print(f1_score_value)

# display = PrecisionRecallDisplay.from_predictions(true_value_test_data["Conscientious"], predictions_transformed, name="DL-Model")
# _ = display.ax_.set_title("2-class Precision-Recall curve")
# file_name = '{}/xgboost_DL-Model_test-data_Precision-Recall-curve.png'.format(path)
# plt.savefig(file_name)
# plt.close()


# ---- tabnet
tab_net_model = TabNetClassifier(optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=0.001), scheduler_params={"step_size":10, "gamma":0.9},
                                 scheduler_fn=torch.optim.lr_scheduler.StepLR, mask_type='entmax')#, seed=42) # "sparsemax"#verbose=1, seed=42)

aug = ClassificationSMOTE(p=0.2)

tab_net_model.fit(X_train=X, y_train=Y, eval_set=[(X,Y),(v_X, v_Y)], eval_name=['train', 'valid'], max_epochs=100 , patience=20, augmentations=aug,
               batch_size=512, virtual_batch_size=128, eval_metric=['auc','accuracy'], num_workers=0, weights=1, drop_last=False)

predictions = tab_net_model.predict_proba(t_X)[:,1]
print(predictions)

predictions_transformed = []
for i, predicted in enumerate(predictions):
    #print(predicted)
    if predicted > 0.7:
        predictions_transformed.append(1)
    else:
        predictions_transformed.append(0)

# summarize history for accuracy
plt.plot(tab_net_model.history['train_accuracy'])
plt.plot(tab_net_model.history['valid_accuracy'])
plt.title('tabnet accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
file_name = '{}/tabnet_history_accuracy_plot.png'.format(path)
plt.savefig(file_name)
plt.close()

_test_data_copy['Conscientious'] = predictions_transformed
conscientious_indeces = _test_data_copy.index[_test_data_copy['Conscientious'] == 0]
none_conscientious_indeces = _test_data_copy.index[_test_data_copy['Conscientious'] == 1]
file_name = '{}/TabNetClassifier_Predicted_test_data_plot.png'.format(path)
plot_data_cluster(transformed_test_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
              'Tabnet predicted test data plot', file_name, show=False, save=True)

# ------- display roc_auc curve
lda_roc_auc = roc_auc_score(true_value_test_data["Conscientious"], predictions_transformed)
fpr, tpr, thresholds = roc_curve(true_value_test_data["Conscientious"], predictions)#predictions[:,1])
file_name = '{}/TabNetClassifier_DL-Model_test-data_ROC-curve.png'.format(path)
plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'DL-Model test data auc (area = %0.2f)' % lda_roc_auc, 
               title = 'Tabnet-Model test data', file_name = file_name, show=False, save=True)

precision, recall, thresholds = precision_recall_curve(true_value_test_data["Conscientious"], predictions)
print(precision)
print(recall)
print(thresholds)

f1_score_value = f1_score(true_value_test_data["Conscientious"], predictions_transformed, average=None)
print(f1_score_value)

test_acc = accuracy_score(y_pred=true_value_test_data["Conscientious"], y_true=predictions_transformed)
print(test_acc)

display = PrecisionRecallDisplay.from_predictions(true_value_test_data["Conscientious"], predictions_transformed, name="DL-Model")
_ = display.ax_.set_title("2-class Precision-Recall curve")
file_name = '{}/TabNetClassifier_DL-Model_test-data_Precision-Recall-curve.png'.format(path)
plt.savefig(file_name)
plt.close()

# --- save tabnet model
file_name = '{}/trained_tabnet_model'.format(path)
saved_filepath = tab_net_model.save_model(file_name)

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
from catboost import CatBoostRegressor, Pool
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

def reduce_dimension_with_selected_model(train_data, test_data, model):
    train = model.fit_transform(train_data)
    test = model.fit_transform(test_data)
    return train, test

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
input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')			# plan of sensors weighting:

# read cvs test data
load_test_data = pd.read_csv("All_Participents_Condition-C_WaveSum_DataFrame.csv", sep=";", decimal=',')			# plan of sensors weighting:

# ------- fitler columns of train data
train_data = input_data.drop(columns=['Conscientious', 'time', 'pId'])

# count rows and columns
c_num = train_data.shape[1]
print(c_num)

# -------  filter columns of test data 
test_data = load_test_data.drop(columns=['time', 'pId'])

r_num_test_data = test_data.shape[0]
test_x = test_data.iloc[:, :].values
print("================ transformend test validation input predictions informations")
true_value_test_data = []
# ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
# set real Conscientious values
for i in range(r_num_test_data):
    true_value_test_data.append(0)
    if load_test_data['pId'].values[i] == 24 or load_test_data['pId'].values[i] == 25: 
        true_value_test_data[i] = 1
true_value_test_data = pd.DataFrame({ "Conscientious" : true_value_test_data})      
print(true_value_test_data["Conscientious"].values)

# ------ Normalizing
# Separating out the features
x_train = train_data.loc[:, :].values
# Separating out the target
y_result_output = np.array(input_data[["Conscientious"]].values.flatten())
print(y_result_output)

transformed_train_x = StandardScaler().fit_transform(x_train)
# Standardizing the features of Test data
transformed_test_x = StandardScaler().fit_transform(test_x)

print("Create output directory")
# --- create dir
mode = 0o666
if not os.path.exists("./output"):
    os.mkdir("./output", mode)
path = "./output/Brute_force_parameter_optimization_results_{}".format(input_data_type)
if not os.path.exists(path):
    os.mkdir(path, mode)

tsne_model = TSNE(n_components=2, learning_rate=500.0 , init='pca', perplexity=30.0)
pca_model = PCA(n_components=2)

#------------------------------------------------------------------------------------ Begin

dimension_reduction_type = 0 # 0, 1, 2

# # set sensor and validity score weights
weight_ecg = 1/5       
weight_eda = 2/5       
weight_eeg = 1/5       
weight_eye = 1/5       
weight_pages = 2      

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



print("------ T-Distributed Stochastic Neighbor Embedding n_components=2 of (True) train data ")
# # ------ T-Distributed Stochastic Neighbor Embedding n_components=2 of train data
# tsne_model = TSNE(n_components=2, learning_rate=500.0 , init='pca', perplexity=30.0)
# transformed_train_x = tsne_model.fit_transform(transformed_train_x)
# print(transformed_train_x.shape)
transformed_train_x, transformed_test_x =  reduce_dimension_with_selected_model(transformed_train_x, transformed_test_x, tsne_model)

# print("------ T-Distributed Stochastic Neighbor Embedding n_components=2 of (True) test data")
# # ------ T-Distributed Stochastic Neighbor Embedding n_components=2 of test data
# transformed_test_x = tsne_model.fit_transform(transformed_test_x)
# print(transformed_test_x.shape)

print("------ Principal Component Analysis n_components=2 of train data")
# # ------ Principal Component Analysis n_components=2 of train data
# pca = PCA(n_components=2)
# transformed_train_x = pca.fit_transform(transformed_train_x)
# #print(pca.score(x)) # Debug only
# print(pca.explained_variance_ratio_)  # Debug only
transformed_train_x, transformed_test_x =  reduce_dimension_with_selected_model(transformed_train_x, transformed_test_x, pca_model)

# transformed_test_x = pca.fit_transform(transformed_test_x)
# #print(pca.score(transformed_test_x)) # Debug only
# print(pca.explained_variance_ratio_)  # Debug only


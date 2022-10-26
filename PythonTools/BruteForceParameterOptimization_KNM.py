from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import precision_recall_curve, log_loss, accuracy_score, f1_score, roc_auc_score, roc_curve 
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.augmentations import ClassificationSMOTE
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from numpy import random
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
print(" ------ load data")
# read csv train data as pandas data frame
input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')			# plan of sensors weighting:
# read cvs test data
load_test_data = pd.read_csv("All_Participents_Condition-C_WaveSum_DataFrame.csv", sep=";", decimal=',')			# plan of sensors weighting:

print(" ------ filter data")
# ------- fitler columns of train data
train_data = input_data.drop(columns=['Conscientious', 'time', 'pId'])
# count rows and columns
c_num = train_data.shape[1]
# -------  filter columns of test data 
test_data = load_test_data.drop(columns=['time', 'pId'])
r_num_test_data = test_data.shape[0]
test_x = test_data.iloc[:, :].values
print("------ transformend test validation input predictions informations")
true_value_test_data = []
# ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
# set real Conscientious values
for i in range(r_num_test_data):
    true_value_test_data.append(0)
    if load_test_data['pId'].values[i] == 24 or load_test_data['pId'].values[i] == 25: 
        true_value_test_data[i] = 1
true_value_test_data = pd.DataFrame({ "Conscientious" : true_value_test_data})      

# ------ Normalizing
print(" ------ Data normalizing with StandardScaler model")
# Separating out the features
x_train = train_data.loc[:, :].values
# Separating out the target
y_result_output = np.array(input_data[["Conscientious"]].values.flatten())
transformed_train_x = StandardScaler().fit_transform(x_train)
# Standardizing the features of Test data
transformed_test_x = StandardScaler().fit_transform(test_x)

# ------ Create output directory
print(" ------ Create output directory")
mode = 0o666
if not os.path.exists("./output"):
    os.mkdir("./output", mode)
path = "./output/Brute_force_parameter_optimization_results_KNM_{}".format(input_data_type)
if not os.path.exists(path):
    os.mkdir(path, mode)

# ------ Create dimension reduction models
print(" ------ Create dimension reduction models")
tsne_model = TSNE(n_components=2, learning_rate=500.0 , init='pca', perplexity=30.0, n_jobs=-1)
pca_model = PCA(n_components=2)

#------------------------------------------------------------------------------------ Begin bruteforce parameter optimization
select_dimension_reduction_method_array = [0, 1, 2]
print(select_dimension_reduction_method_array)
weights_array = np.linspace(0, 1, 11)
print(weights_array)

col_names =  ['dimension_reduction_type', 'weight_ecg', 'weight_eda', 'weight_eeg', 'weight_eye', 'weight_pages', 'n_neighbors', 'accuracy_train_data', 'prediction_threshold', 'roc_auc', 'f1_score_value', 'accuracy']
knm_weights_data_frame  = pd.DataFrame(columns = col_names)
#knm_weights_data_frame.loc[len(knm_weights_data_frame)] = [2, 4, 5, 2, 4, 5, 2, 4, 5, 2, 4, 5]

# until key input with crtl + c
for d in select_dimension_reduction_method_array:
    print(" ------ Setting dimension reduction algorithm type 0=None, 1=tSNE, 2=PCA")
    dimension_reduction_type = d # 0, 1, 2
    for e in weights_array:
        weight_ecg = e
        for ed in weights_array:
            weight_eda = ed
            for ee in weights_array:
                weight_eeg = ee
                for ey in weights_array:
                    weight_eye = ey
                    for p in weights_array:
                        weight_pages = p

                        if weight_ecg == 0 and weight_eda == 0 and weight_eeg == 0 and weight_eye == 0 and weight_pages == 0:
                            continue
                        
                        #last running weights: dimension_reduction_type: 0, weight_ecg: 0.0, weight_eda: 0.4, weight_eeg: 0.0, weight_eye: 0.4, weight_pages: 0.5
                        if weight_ecg == 0 and weight_eda < 0.4:
                            continue

                        # # set sensor and validity score weights
                        # weight_pages = random.choice(weights_array)     #1  
                        parameter_infos_to_save = "--- Parameter to optimize\ndimension_reduction_type: {}, weight_ecg: {}, weight_eda: {}, weight_eeg: {}, weight_eye: {}, weight_pages: {}".format(
                                                dimension_reduction_type, weight_ecg, weight_eda, weight_eeg, weight_eye, weight_pages)

                        print(parameter_infos_to_save)
                        #sys.exit()
                        print(" ------ Setting sensor weights on data")
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

                        if dimension_reduction_type == 1:
                            print("------ T-Distributed Stochastic Neighbor Embedding n_components=2 of (True) train data ")
                            # ------ T-Distributed Stochastic Neighbor Embedding n_components=2 of train data
                            transformed_train_x, transformed_test_x =  reduce_dimension_with_selected_model(transformed_train_x, transformed_test_x, tsne_model)
                        if dimension_reduction_type == 2:
                            print("------ Principal Component Analysis n_components=2 of train data")
                            # ------ Principal Component Analysis n_components=2 of train data
                            transformed_train_x, transformed_test_x =  reduce_dimension_with_selected_model(transformed_train_x, transformed_test_x, pca_model)

                        print(" ------ Data Splitting into train and validation data")
                        # ------ Data Splitting into train and validation data
                        x_test_data_frame = pd.DataFrame(data=transformed_test_x)
                        
                        t_X = x_test_data_frame.to_numpy(dtype='float32')
                        print("t_X data shape : {}".format(t_X.shape))
                        t_Y = np.array(true_value_test_data["Conscientious"].values.flatten()) #t_Y = np_utils.to_categorical(t_Y, num_classes=2)
                        print("t_Y data shape : {}".format(t_Y.shape))

                        # ---- hyper parameters
                        print("Setting up hyperparameter for dl model, by using standard value")
                        num_classes   = 1
                        prediction_threshold = 0.5
                        accuracy_threshold_to_save = 0.7

                        # ---- k-neighbors-classifier
                        n_neighbors = 190
                        weights='uniform'
                        algorithm='ball_tree'
                        X = pd.DataFrame(data = transformed_train_x)
                        Y = pd.DataFrame(data = y_result_output)
                        print("KNeighborsClassifier (n_neighbors={}, weights={}, algorithm={})".format(n_neighbors, weights, algorithm))
                        k_neigbors_classifier_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, n_jobs=-1)
                        k_neigbors_classifier_model.fit(X, Y) 

                        # --- test data predictions
                        accuracy_train_data = k_neigbors_classifier_model.score(X, Y) 

                        # --- test data predictions
                        prediction = k_neigbors_classifier_model.predict(t_X) 
                        prediction = prediction.astype("int")
                        confidence = k_neigbors_classifier_model.predict_proba(t_X)[:,1]
                        accuracy = k_neigbors_classifier_model.score(t_X, t_Y) 

                        predictions_transformed = []
                        for i, predicted in enumerate(confidence):
                            if predicted > prediction_threshold:
                                predictions_transformed.append(1)
                            else:
                                predictions_transformed.append(0)

                        roc_auc = roc_auc_score(y_true=t_Y, y_score=predictions_transformed)
                        print("lda_roc_auc: ".format(roc_auc))
                        fpr, tpr, thresholds = roc_curve(y_true=t_Y, y_score=predictions_transformed)
                        print("fpr: {} ; tpr: {} ; thresholds: {}".format(fpr, tpr, thresholds))
                        precision, recall, thresholds = precision_recall_curve(y_true=t_Y, probas_pred=confidence)
                        print("precision: {} ; recall: {} ; thresholds: {}".format(precision, recall, thresholds))
                        f1_score_value = f1_score(y_true=t_Y, y_pred=predictions_transformed, average=None)
                        print("f1_score_value: {}".format(f1_score_value))
                        accuracy = accuracy_score(y_true=t_Y, y_pred=predictions_transformed)
                        print("accuracy_score: {}".format(accuracy))

                        if accuracy > accuracy_threshold_to_save:
                            file_name = "{}/k-neighbors-classifier-restuls.txt".format(path)
                            knc_parameter = parameter_infos_to_save + "\n--- Model k-neighbors-classifier restuls on test data:\nn_neighbors: {}, accuracy_train_data: {}, prediction_threshold: {}, roc_auc: {}, f1_score_value: {}, accuracy: {}\n".format(
                                                    n_neighbors, accuracy_train_data, prediction_threshold, roc_auc, f1_score_value, accuracy)
                            with open(file_name, 'a') as file:
                                file.write(f'\n{knc_parameter}')
                            
                            knm_weights_data_frame.loc[len(knm_weights_data_frame)] = [dimension_reduction_type, weight_ecg, weight_eda, weight_eeg, weight_eye, weight_pages, 
                                                                                       n_neighbors, accuracy_train_data, prediction_threshold, roc_auc, f1_score_value, accuracy]
                            data_frame_file_name = "{}/k-neighbors-classifier-restuls.csv".format(path)                                                                                      
                            knm_weights_data_frame.to_csv(data_frame_file_name, index=False) 


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.matlib
import os
from os.path import exists

def plot_roc_curve(true_positive_rate, false_positive_rate, legend_label, title, file_name, show=False, save=False):
    plt.figure(figsize=(15,10))
    plt.plot(false_positive_rate, true_positive_rate, label=legend_label)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(title, fontsize=18)
    plt.grid(which="major", alpha=0.6)
    plt.grid(which="minor", alpha=0.6)
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
    plt.tight_layout() 
    if save:
        plt.savefig(file_name)
    if show:
        plt.show()
    plt.close()

def plot_data_cluster(data, conscientious_indeces_list, none_conscientious_indeces_list, title, file_name, show=False, save=False):
    plt.figure(figsize=(15,10))
    plt.scatter(data[conscientious_indeces_list, 0], data[conscientious_indeces_list, 1], c="b")
    plt.scatter(data[none_conscientious_indeces_list, 0], data[none_conscientious_indeces_list, 1], c="r")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontsize=18)
    plt.tight_layout() 
    if save:
        plt.savefig(file_name)
    if show:
        plt.show()
    plt.close()

def write_matrix_and_report_to_file(file_name, content):
    if exists(file_name):
        os.remove(file_name)
    file = open(file_name, "w")
    file.write(content)
    file.close()

# input_data_type = { all_sensors = 0, ecg = 1, eda = 2, eeg = 3, eye = 4, pages = 5 }
input_data_type = 0
intervals = 3 
# 0=scaled; 1=scaled+pca; 2=scaled+tSNE
dimensions = [0, 1, 2]               
p_thresholds = [0.3, 0.5, 0.7] 
max_range = 200
test_size=0.25
# 3=Selection_over_5_ratio; 56=Kaiser_Rule;
n_components=56                      

# read csv train data as pandas data frame
input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')			# plan of sensors weighting:

# ------- fitler columns of train data
train_data = input_data.drop(columns=['Conscientious', 'time', 'pId'])

# count rows and columns
c_num = train_data.shape[1]
print(c_num)

# ------ Normalizing
# Separating out the features
x_train = train_data.loc[:, :].values
# Separating out the target
y_result_output = np.array(input_data[["Conscientious"]].values.flatten())
print(y_result_output)
# Standardizing the features of train data
transformed_train_x = StandardScaler().fit_transform(x_train)
# Standardizing the features of Test data

# set sensor and validity score weights
weight_ecg = 1  # 2/5     
weight_eda = 1  # 3/5     
weight_eeg = 1  # 1/5     
weight_eye = 1  # 3/5     
weight_pages = 1 # 1

if input_data_type == 0:
    transformed_train_x[:,0:26]    = transformed_train_x[:,0:26]    * weight_ecg
    transformed_train_x[:,26:31]   = transformed_train_x[:,26:31]   * weight_eda
    transformed_train_x[:,31:107]  = transformed_train_x[:,31:107]  * weight_eeg
    transformed_train_x[:,152:157] = transformed_train_x[:,152:157] * weight_eeg
    transformed_train_x[:,107:129] = transformed_train_x[:,107:129] * weight_eye
    transformed_train_x[:,141:149] = transformed_train_x[:,141:149] * weight_eye
    transformed_train_x[:,129:141] = transformed_train_x[:,129:141] * weight_pages
    transformed_train_x[:,149:152] = transformed_train_x[:,149:152] * weight_pages

print("Create output directory")
# --- create dir
mode = 0o666
if not exists("./output"):
    os.mkdir("./output", mode)
if not exists("./output/K-Neighbors-Classifier-Model"):
    os.mkdir("./output/K-Neighbors-Classifier-Model", mode)

transformed_train_x_temp = transformed_train_x

for interval in range(1, intervals):
    for dimension in dimensions:
        for p_threshold in p_thresholds:
            
            path = "./output/K-Neighbors-Classifier-Model/K-Neighbors-Classifier-Model_{}_{}_{}_{}_{}".format(input_data_type, n_components, interval, dimension, int(p_threshold*10))
            if not exists(path):
                os.mkdir(path, mode)
            
            propability_threshold = p_threshold
            transformed_train_x = transformed_train_x_temp

            print("------ Transformed (True) train data")
            # ------ Transformed (True) train data
            conscientious_indeces = input_data.index[input_data['Conscientious'] == 0]
            none_conscientious_indeces = input_data.index[input_data['Conscientious'] == 1]
            file_name = '{}/Transformed_train_data_plot.png'.format(path)
            plot_data_cluster(transformed_train_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
                            'Transformed (True) train data  plot', file_name, show=False, save=True)

            print("------- K-Neighbors-Classifier-Model")
            # ------- K-Neighbors-Classifier-Model
            # knc_train_data = train_data.copy()
            # --- training (fitting)

            if dimension == 1:
                print("------ Principal Component Analysis test explainable variance of given features in train data")
                # test explainable variance of given features
                pca = PCA(n_components=n_components)
                print(pca.get_params(True))
                transformed_train_x = pca.fit_transform(transformed_train_x)
                #transformed_test_x = pca.fit_transform(transformed_test_x)

            if dimension == 2:
                print("------ T-Distributed Stochastic Neighbor Embedding of (True) train data ")
                # ------ T-Distributed Stochastic Neighbor Embedding of train data
                # ValueError: 'n_components' should be inferior to 4 for the barnes_hut algorithm as it relies on quad-tree or oct-tree.
                tsne_model = TSNE(n_components=3, learning_rate=500.0 , init='pca', perplexity=30.0)
                transformed_train_x = tsne_model.fit_transform(transformed_train_x)

            X_train, X_test, y_train, y_test = train_test_split(transformed_train_x, 
                                                                np.array(input_data["Conscientious"].values.flatten()), test_size=test_size,  shuffle=True, 
                                                                stratify=np.array(input_data["Conscientious"].values.flatten()))

            error_rates = []
            for a in range(1, max_range):
                k = a
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                preds = knn.predict(X_test)
                error_rates.append(np.mean(abs(y_test - preds)))

            plt.figure(figsize=(15,10))
            plt.plot(range(1, max_range),error_rates,color='blue', linestyle='dashed', marker='o',
                    markerfacecolor='red', markersize=10)
            plt.title('Error Rate vs. K Value', fontsize=18)
            plt.xlabel('K', fontsize=16)
            plt.ylabel('Error Rate', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(which="major", alpha=0.6)
            plt.grid(which="minor", alpha=0.6)
            plt.tight_layout() 
            file_name = '{}/K-Neighbors-Classifier-Model_error_rate_vs_K_value.png'.format(path)
            plt.savefig(file_name)
            plt.close()

            n_neighbors_error = error_rates.index(max(error_rates)) + 1
            if n_neighbors_error == 0 or n_neighbors_error == 1:
                error_rates[0] = 0
                #error_rates[1] = 0
                n_neighbors_error = error_rates.index(max(error_rates)) + 1

            matrix = confusion_matrix(y_test, preds)
            print(matrix)
            file_name = '{}/K-Neighbors-Classifier-Model_error_rate_vs_K_value_confusion_Matrix.txt'.format(path)
            write_matrix_and_report_to_file(file_name, np.array2string(matrix))

            report = classification_report(y_test, preds)
            print(report)
            file_name = '{}/K-Neighbors-Classifier-Model_error_rate_vs_K_value_report.txt'.format(path)
            write_matrix_and_report_to_file(file_name, report)

            acc = []
            # Will take some time
            for i in range(1, max_range):
                neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
                yhat = neigh.predict(X_test)
                acc.append(metrics.accuracy_score(y_test, yhat))

            plt.figure(figsize=(15,10))
            plt.plot(range(1, max_range),acc,color = 'blue',linestyle='dashed', 
                    marker='o',markerfacecolor='red', markersize=10)
            plt.title('accuracy vs. K Value', fontsize=18)
            plt.xlabel('K', fontsize=16)
            plt.ylabel('Accuracy', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(which="major", alpha=0.6)
            plt.grid(which="minor", alpha=0.6)
            plt.tight_layout() 
            file_name = '{}/K-Neighbors-Classifier-Model_accuracy_vs_K_value.png'.format(path)
            plt.savefig(file_name)
            plt.close()
            print("Maximum accuracy: ",max(acc),"at K =",acc.index(max(acc)))

            n_neighbors_acc = acc.index(max(acc)) + 1
            if n_neighbors_acc == 0 or n_neighbors_acc == 1:
                acc[0] = 0
                #acc[1] = 0
                n_neighbors_acc = acc.index(max(acc)) + 1

            n_neighbors = n_neighbors_acc
            if n_neighbors_acc > n_neighbors_error:
                n_neighbors = n_neighbors_error

            print("Maximum accuracy: ", acc[n_neighbors],"at K =", n_neighbors)
            file_name = '{}/K-Neighbors-Classifier-Model_maximum_accuracy_report.txt'.format(path)
            write_matrix_and_report_to_file(file_name, "Maximum accuracy: {}  at K = {}".format(acc[n_neighbors], n_neighbors))

            # ------------------------------------------------------------------------------------------------------------------
            n_neighbors = n_neighbors
            k_neigbors_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
            # --- train data predictions 
            k_neigbors_classifier.fit(X_train, y_train) 

            input_score = k_neigbors_classifier.score(X_train, y_train) 
            print(input_score)

            predictions = k_neigbors_classifier.predict_proba(X_train)[:,1]
            print(predictions)

            predictions_transformed = []
            for i, predicted in enumerate(predictions):
                if predicted > propability_threshold:
                    predictions_transformed.append(1)
                else:
                    predictions_transformed.append(0)

            print(k_neigbors_classifier.get_params(deep=True))
            matrix = confusion_matrix(y_train, predictions_transformed)
            print(matrix)
            file_name = '{}/K-Neighbors-Classifier-Model_train_data_confusion_Matrix.txt'.format(path)
            write_matrix_and_report_to_file(file_name, np.array2string(matrix))

            report = classification_report(y_train, predictions_transformed)
            print(report)
            file_name = '{}/K-Neighbors-Classifier-Model_train_deta_report.txt'.format(path)
            write_matrix_and_report_to_file(file_name, report)

            print("------ K-Neighbors-Classifier-Model n_components=2 of (predicted) train data ")
            conscientious_indeces = []
            counter = 0 
            for val in predictions_transformed:
                if val == 0:
                    conscientious_indeces.append(counter)
                counter += 1

            none_conscientious_indeces = []
            counter = 0 
            for val in predictions_transformed:
                if val == 1:
                    none_conscientious_indeces.append(counter)
                counter += 1

            #print(none_conscientious_indeces)
            file_name = '{}/K-Neighbors-Classifier-Model_predicted_train_data_plot.png'.format(path)
            plot_data_cluster(X_train, conscientious_indeces, none_conscientious_indeces, 
                            'K-Neighbors-Classifier-Model n_components=2 of (predicted) train data plot', file_name, show=False, save=True)

            # ------- display roc_auc curve
            knc_roc_auc = roc_auc_score(y_train, predictions_transformed)
            fpr, tpr, thresholds = roc_curve(y_train, predictions)
            file_name = '{}/K-Neighbors-Classifier-Model_train_data_roc-curve.png'.format(path)
            plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'KNNC AUC (area = %0.2f)' % knc_roc_auc, 
                        title = 'K-Neighbors-Classifier-Model train data', file_name = file_name, show=False, save=True)

            # --------------------------------------------------------------------------------------------------------------------------
            # --- test data predictions 
            predictions = k_neigbors_classifier.predict_proba(X_test)[:,1]
            print(predictions)

            predictions_transformed = []
            for i, predicted in enumerate(predictions):
                if predicted > propability_threshold:
                    predictions_transformed.append(1)
                else:
                    predictions_transformed.append(0)
                
            matrix = confusion_matrix(y_test, predictions_transformed)
            print(matrix)
            file_name = '{}/K-Neighbors-Classifier-Model_test_data_confusion_Matrix.txt'.format(path)
            write_matrix_and_report_to_file(file_name, np.array2string(matrix))

            report = classification_report(y_test, predictions_transformed)
            print(report)
            file_name = '{}/K-Neighbors-Classifier-Model_test_deta_report.txt'.format(path)
            write_matrix_and_report_to_file(file_name, report)

            print(k_neigbors_classifier.get_params(deep=True))
            print(accuracy_score(y_test, predictions_transformed))
            input_score = k_neigbors_classifier.score(X_test,  y_test) 
            print(input_score)

            print("------ K-Neighbors-Classifier-Model (predicted) test data ")
            conscientious_indeces = []
            counter = 0 
            for val in predictions_transformed:
                if val == 0:
                    conscientious_indeces.append(counter)
                counter += 1

            none_conscientious_indeces = []
            counter = 0 
            for val in predictions_transformed:
                if val == 1:
                    none_conscientious_indeces.append(counter)
                counter += 1
            
            file_name = '{}/K-Neighbors-Classifier-Model_predicted_test_data_plot.png'.format(path)
            plot_data_cluster(X_test, conscientious_indeces, none_conscientious_indeces, 
                            'K-Neighbors-Classifier-Model (predicted) test data', file_name, show=False, save=True)

            # ------- display roc_auc curve
            knc_roc_auc = roc_auc_score(y_test, predictions_transformed)
            fpr, tpr, thresholds = roc_curve(y_test, predictions)
            file_name = '{}/K-Neighbors-Classifier-Model_test_data_roc-curve.png'.format(path)
            plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'KNNC AUC (area = %0.2f)' % knc_roc_auc,
                        title = 'K-Neighbors-Classifier-Model test data', file_name = file_name, show=False, save=True)
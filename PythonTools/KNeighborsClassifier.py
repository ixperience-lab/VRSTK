from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.matlib
import sys
import os
from os.path import exists

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

def write_matrix_and_report_to_file(file_name, content):
    if exists(file_name):
        os.remove(file_name)
    file = open(file_name, "w")
    file.write(content)
    file.close()

# input_data_type = { all_sensors = 0, ecg = 1, eda = 2, eeg = 3, eye = 4, pages = 5 }
input_data_type = 0

#input_data_copy = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrameKopie.csv", sep=";", decimal=',')			# plan of sensors weighting:
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
#train_data_copy = input_data_copy.drop(columns=['Conscientious', 'time', 'pId'])
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
    if load_test_data['pId'].values[i] == 24 or load_test_data['pId'].values[i] == 25: # or load_test_data['pId'].values[i] == 28:
        true_value_test_data[i] = 1
true_value_test_data = pd.DataFrame({ "Conscientious" : true_value_test_data})      
print(true_value_test_data["Conscientious"].values)

# ------ Normalizing
# Separating out the features
#x_train_copy = train_data_copy.loc[:, :].values
x_train = train_data.loc[:, :].values
# Separating out the target
#y_result_output_copy = np.array(input_data_copy[["Conscientious"]].values.flatten())
y_result_output = np.array(input_data[["Conscientious"]].values.flatten())
print(y_result_output)
# Standardizing the features of train data
#transformed_train_x_copy = StandardScaler().fit_transform(x_train_copy)
transformed_train_x = StandardScaler().fit_transform(x_train)
# Standardizing the features of Test data
transformed_test_x = StandardScaler().fit_transform(test_x)

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
if not exists("./output"):
    os.mkdir("./output", mode)
path = "./output/K-Neighbors-Classifier-Model_{}".format(input_data_type)
if not exists(path):
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
#knc_x_embedded_data_frame = pd.DataFrame(data = transformed_train_x)
knc_train_data = train_data.copy()
# --- training (fitting)

print("------ Principal Component Analysis test explainable variance of given features in train data")
# test explainable variance of given features
pca = PCA(n_components=3)
transformed_train_x = pca.fit_transform(transformed_train_x)
transformed_test_x = pca.fit_transform(transformed_test_x)

# print("------ T-Distributed Stochastic Neighbor Embedding n_components=2 of (True) train data ")
# # ------ T-Distributed Stochastic Neighbor Embedding n_components=2 of train data
# tsne_model = TSNE(n_components=3, learning_rate=500.0 , init='pca', perplexity=30.0)
# # transformed_train_x_copy = tsne_model.fit_transform(transformed_train_x_copy)
# transformed_train_x = tsne_model.fit_transform(transformed_train_x)
# transformed_test_x = tsne_model.fit_transform(transformed_test_x)

knc_x_embedded_data_frame = pd.DataFrame(data = transformed_train_x)
# knc_x_embedded_data_frame_copy = pd.DataFrame(data = transformed_train_x_copy)
# X_train, X_test, y_train, y_test = train_test_split(knc_x_embedded_data_frame_copy, y_result_output_copy, test_size=0.3)
# X_train, X_test, y_train, y_test = train_test_split(knc_x_embedded_data_frame, y_result_output, test_size=0.25)

X_train = transformed_train_x
X_test = transformed_test_x
y_train = y_result_output
y_test = true_value_test_data['Conscientious']

error_rates = []
max_range = 200
#knc_test_x_embedded_data_frame = pd.DataFrame(data = transformed_test_x)
for a in range(1, max_range):
    k = a
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    error_rates.append(np.mean(abs(y_test - preds)))

plt.figure(figsize=(15,10))
plt.plot(range(1, max_range),error_rates,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
file_name = '{}/K-Neighbors-Classifier-Model_error_rate_vs_K_value.png'.format(path)
plt.savefig(file_name)
plt.show()
plt.close()

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
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
file_name = '{}/K-Neighbors-Classifier-Model_accuracy_vs_K_value.png'.format(path)
plt.savefig(file_name)
plt.show()
plt.close()
print("Maximum accuracy: ",max(acc),"at K =",acc.index(max(acc)))

n_neighbors = acc.index(max(acc))
if n_neighbors == 0 or n_neighbors == 1:
    acc[0] = 0
    acc[1] = 0
    n_neighbors = acc.index(max(acc))

print("Maximum accuracy: ", acc[n_neighbors],"at K =", n_neighbors)
file_name = '{}/K-Neighbors-Classifier-Model_maximum_accuracy_report.txt'.format(path)
write_matrix_and_report_to_file(file_name, "Maximum accuracy: {}  at K = {}".format(acc[n_neighbors], n_neighbors))

#sys.exit()
# ------------------------------------------------------------------------------------------------------------------
n_neighbors = n_neighbors
k_neigbors_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
k_neigbors_classifier.fit(knc_x_embedded_data_frame, y_result_output) 

input_score = k_neigbors_classifier.score(knc_x_embedded_data_frame, y_result_output) 
print(input_score)
# --- train data predictions 
knc_train_data["Conscientious"] = k_neigbors_classifier.predict(knc_x_embedded_data_frame) 
knc_train_data["Conscientious"] = knc_train_data["Conscientious"].astype("int")
knc_train_data["pId"] = input_data["pId"]

prediction = k_neigbors_classifier.predict_proba(knc_x_embedded_data_frame)
knc_train_data["Confidence"] = np.max(prediction, axis = 1)

print(k_neigbors_classifier.get_params(deep=True))
matrix = confusion_matrix(y_result_output, knc_train_data['Conscientious'])
print(matrix)
file_name = '{}/K-Neighbors-Classifier-Model_train_data_confusion_Matrix.txt'.format(path)
write_matrix_and_report_to_file(file_name, np.array2string(matrix))

report = classification_report(y_result_output, knc_train_data["Conscientious"])
print(report)
file_name = '{}/K-Neighbors-Classifier-Model_train_deta_report.txt'.format(path)
write_matrix_and_report_to_file(file_name, report)

# get probability score of each sample
# loss = log_loss(y_result_output, knc_train_data['Conscientious'])
# print(loss)
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
	
matrix = confusion_matrix(true_value_test_data['Conscientious'], knc_test_data['Conscientious'])
print(matrix)
file_name = '{}/K-Neighbors-Classifier-Model_test_data_confusion_Matrix.txt'.format(path)
write_matrix_and_report_to_file(file_name, np.array2string(matrix))

report = classification_report(true_value_test_data['Conscientious'], knc_test_data['Conscientious'])
print(report)
file_name = '{}/K-Neighbors-Classifier-Model_test_deta_report.txt'.format(path)
write_matrix_and_report_to_file(file_name, report)

print(k_neigbors_classifier.get_params(deep=True))
print(accuracy_score(true_value_test_data['Conscientious'], knc_test_data['Conscientious']))
input_score = k_neigbors_classifier.score(knc_test_x_embedded_data_frame,  true_value_test_data['Conscientious']) 
print(input_score)
# get probability score of each sample
# loss = log_loss(true_value_test_data['Conscientious'], knc_test_data['Conscientious'])
# print(loss)

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


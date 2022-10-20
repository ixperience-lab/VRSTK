import numpy as np
import numpy.matlib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
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

# set real conscientius values
# for i in range(r_num):
#     if input_data["pId"].values[i] == 14 or input_data["pId"].values[i] == 15 or input_data["pId"].values[i] == 16:
#         input_data['Conscientious'].values[i] = 0

true_value_test_data = []
# ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
# set real Conscientious values
r_num_test_data = load_test_data.shape[0]
for i in range(r_num_test_data):
    true_value_test_data.append(0)
    if load_test_data['pId'].values[i] == 24 or load_test_data['pId'].values[i] == 25 or load_test_data['pId'].values[i] == 28:
        true_value_test_data[i] = 1
true_value_test_data = pd.DataFrame({ "Conscientious" : true_value_test_data})      
print(true_value_test_data["Conscientious"].values)

# ----- set sensor and validity score weights
# -----
weight_ecg = 2/5      
weight_eda = 2/5       
weight_eeg = 1/5       
weight_eye = 1/5       
weight_pages = 1 

# filter train data 
train_data = input_data.drop(columns=['Conscientious', 'time', 'pId'])
# get input_data shape
r_num = train_data.shape[0]
print(r_num)
c_num = train_data.shape[1]
print(c_num)

# filter test data 
test_data = load_test_data.drop(columns=['time', 'pId'])

#r_num_test_data = test_data.shape[0]
test_data_x = test_data.iloc[:, :].values

# ------ create normalizer 
# to normalize data for creating more varianz in the data
scaler = StandardScaler()
# ------ normalize train data
x = train_data.iloc[:, :].values
scaler.fit(x)
transformed_train_data_x = scaler.transform(x)
# separating train output data as target 'Conscientious'
true_value_train_data_y = np.array(input_data[["Conscientious"]].values.flatten()) 
# ------ normalize test data
scaler.fit(test_data_x)
transformed_test_data_x = scaler.transform(test_data_x)

if input_data_type == 0:
	transformed_train_data_x[:,0:26]    = transformed_train_data_x[:,0:26]    * weight_ecg
	transformed_train_data_x[:,26:31]   = transformed_train_data_x[:,26:31]   * weight_eda
	transformed_train_data_x[:,31:107]  = transformed_train_data_x[:,31:107]  * weight_eeg
	transformed_train_data_x[:,152:157] = transformed_train_data_x[:,152:157] * weight_eeg
	transformed_train_data_x[:,107:129] = transformed_train_data_x[:,107:129] * weight_eye
	transformed_train_data_x[:,141:149] = transformed_train_data_x[:,141:149] * weight_eye
	transformed_train_data_x[:,129:141] = transformed_train_data_x[:,129:141] * weight_pages
	transformed_train_data_x[:,149:152] = transformed_train_data_x[:,149:152] * weight_pages

	transformed_test_data_x[:,0:26]    = transformed_test_data_x[:,0:26]    * weight_ecg
	transformed_test_data_x[:,26:31]   = transformed_test_data_x[:,26:31]   * weight_eda
	transformed_test_data_x[:,31:107]  = transformed_test_data_x[:,31:107]  * weight_eeg
	transformed_test_data_x[:,152:157] = transformed_test_data_x[:,152:157] * weight_eeg
	transformed_test_data_x[:,107:129] = transformed_test_data_x[:,107:129] * weight_eye
	transformed_test_data_x[:,141:149] = transformed_test_data_x[:,141:149] * weight_eye
	transformed_test_data_x[:,129:141] = transformed_test_data_x[:,129:141] * weight_pages
	transformed_test_data_x[:,149:152] = transformed_test_data_x[:,149:152] * weight_pages

print("Create output directory")
# --- create dir
mode = 0o666
if not os.path.exists("./output"):
    os.mkdir("./output", mode)
path = "./output/Linear-Discriminant-Analysis_Dimension_Reduction_{}".format(input_data_type)
if not os.path.exists(path):
    os.mkdir(path, mode)

# ------ training data plot
conscientious_indeces = input_data.index[input_data['Conscientious'] == 0]
none_conscientious_indeces = input_data.index[input_data['Conscientious'] == 1]
file_name = '{}/True_train_data_plot.png'.format(path)
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1, 1, 1)
for i in range(c_num - 1):
    ax.scatter(x[conscientious_indeces.tolist(), i], x[conscientious_indeces.tolist(), i+1], c="b")
    ax.scatter(x[none_conscientious_indeces.tolist(), i], x[none_conscientious_indeces.tolist(), i+1], c="r")
ax.set_title("Not transformed (True) train data  plot", fontsize=16)
plt.savefig(file_name)
plt.close()

file_name = '{}/Transformed_True_train_data_plot.png'.format(path)
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1, 1, 1)
for i in range(c_num - 1):
    ax.scatter(transformed_train_data_x[conscientious_indeces.tolist(), i], transformed_train_data_x[conscientious_indeces.tolist(), i+1], c="b")
    ax.scatter(transformed_train_data_x[none_conscientious_indeces.tolist(), i], transformed_train_data_x[none_conscientious_indeces.tolist(), i+1], c="r")
ax.set_title("Transformed (True) train data  plot", fontsize=16)
plt.savefig(file_name)
plt.close()

# ------ test data plot
conscientious_indeces = true_value_test_data.index[true_value_test_data['Conscientious'] == 0]
none_conscientious_indeces = true_value_test_data.index[true_value_test_data['Conscientious'] == 1]
file_name = '{}/True_test_data_plot.png'.format(path)
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1, 1, 1)
for i in range(c_num - 1):
    ax.scatter(test_data_x[conscientious_indeces.tolist(), i], test_data_x[conscientious_indeces.tolist(), i+1], c="b")
    ax.scatter(test_data_x[none_conscientious_indeces.tolist(), i], test_data_x[none_conscientious_indeces.tolist(), i+1], c="r")
ax.set_title("Not transformed (True) test data plot", fontsize=16)
plt.savefig(file_name)
plt.close()

file_name = '{}/Transformed_True_test_data_plot.png'.format(path)
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1, 1, 1)
for i in range(c_num - 1):
    ax.scatter(transformed_test_data_x[conscientious_indeces.tolist(), i], transformed_test_data_x[conscientious_indeces.tolist(), i+1], c="b")
    ax.scatter(transformed_test_data_x[none_conscientious_indeces.tolist(), i], transformed_test_data_x[none_conscientious_indeces.tolist(), i+1], c="r")
ax.set_title("Transformed (True) test data plot", fontsize=16)
plt.savefig(file_name)
plt.close()

# ----- dimension reduction on k neihbors with 1d
print("------- LDA dimension reduction to 1d")
# ------ create and train linearDiscriminantAnalysis
linearDiscriminantAnalysis = LinearDiscriminantAnalysis(n_components=1)
# ------ training lda fit
linearDiscriminantAnalysis.fit(transformed_train_data_x, true_value_train_data_y)
lda_train_components = linearDiscriminantAnalysis.transform(transformed_train_data_x)
lda_test_components = linearDiscriminantAnalysis.transform(transformed_test_data_x)

print("------- K-Neighbors-Classifier-Model")
# ------- K-Neighbors-Classifier-Model
knc_x_embedded_data_frame = pd.DataFrame(data = lda_train_components)
knc_train_data = train_data.copy()
# --- training (fitting)
k_neigbors_classifier = KNeighborsClassifier(n_neighbors=190, weights='uniform', algorithm='ball_tree')
k_neigbors_classifier.fit(knc_x_embedded_data_frame, true_value_train_data_y) 

# --- train data predictions 
knc_train_data["Conscientious"] = k_neigbors_classifier.predict(knc_x_embedded_data_frame) 
knc_train_data["Conscientious"] = knc_train_data["Conscientious"].astype("int")
knc_train_data["pId"] = input_data["pId"]

prediction = k_neigbors_classifier.predict_proba(knc_x_embedded_data_frame)
knc_train_data["Confidence"] = np.max(prediction, axis = 1)

print("Print train data result informations")
input_score = k_neigbors_classifier.score(knc_x_embedded_data_frame, true_value_train_data_y) 
print(input_score)
knc_parameters = k_neigbors_classifier.get_params(deep=True)
print(knc_parameters)
loss = log_loss(true_value_train_data_y, knc_train_data['Conscientious'])
print(loss)
accuracy = accuracy_score(true_value_train_data_y, knc_train_data['Conscientious'])
print(accuracy)


#true_value_test_data["Conscientious"]
# --- test data predictions 
knc_test_x_embedded_data_frame = pd.DataFrame(data = lda_test_components)
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

print("Print test data result inforamtions")

input_score = k_neigbors_classifier.score(knc_test_x_embedded_data_frame, true_value_test_data) 
print(input_score)
knc_parameters = k_neigbors_classifier.get_params(deep=True)
print(knc_parameters)
loss = log_loss(true_value_test_data['Conscientious'], knc_test_data['Conscientious'])
print(loss)
accuracy = accuracy_score(true_value_test_data['Conscientious'], knc_test_data['Conscientious'])
print(accuracy)
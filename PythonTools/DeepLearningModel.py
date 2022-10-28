from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import precision_recall_curve, log_loss, accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
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

# set sensor and validity score weights
weight_ecg = 1 # 1/5      
weight_eda = 1 # 2/5      
weight_eeg = 1 # 1/5      
weight_eye = 1 # 1/5 
weight_pages = 1 # 2      

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
# tsne_model = TSNE(n_components=3, learning_rate=500.0 , init='pca', perplexity=30.0)
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
# pca = PCA(n_components=3)
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

# ---- hyper parameters
iterration = 1
learning_rate = 0.001
step_size = 10
gamma = 0.9
batch_size = 512
virtual_batch_size = 64
num_epochs = 100
patience = 10
num_classes = 1

aug = ClassificationSMOTE(p=0.2)
# ---- tabnet
tab_net_model = TabNetClassifier(optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=learning_rate), scheduler_params={"step_size":step_size, "gamma":gamma},
                                 scheduler_fn=torch.optim.lr_scheduler.StepLR, mask_type='entmax')
temp_path = path
for iter in range(iterration):
    path = "{}/iterration_{}".format(temp_path, iter)
    if not os.path.exists(path):
        os.mkdir(path, mode)
    train_dataframe, validation_dataframe, y_train_true_output, y_validation_true_output = train_test_split(transformed_train_x, 
                                        np.array(input_data["Conscientious"].values.flatten()), test_size=0.4,  shuffle=True, 
                                        stratify=np.array(input_data["Conscientious"].values.flatten()))

    test_dataframe, validation_dataframe, y_test_true_output, y_validation_true_output = train_test_split(validation_dataframe, 
                                    y_validation_true_output, test_size=0.5,  shuffle=True, 
                                    stratify=y_validation_true_output)

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

    t_X = test_dataframe.astype('float32') #x_test_data_frame.to_numpy(dtype='float32')
    t_Y = y_test_true_output #np.array(true_value_test_data["Conscientious"].values.flatten())
    #t_Y = np_utils.to_categorical(t_Y, num_classes=2)
    print(t_X.shape)
    print(t_Y.shape)

    tab_net_model.fit(X_train=X, y_train=Y, eval_set=[(X,Y),(v_X, v_Y)], eval_name=['train', 'valid'], max_epochs=num_epochs , patience=10, augmentations=aug,
                batch_size=batch_size, virtual_batch_size=virtual_batch_size, eval_metric=['auc','accuracy'], drop_last=False)

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

    # _test_data_copy['Conscientious'] = predictions_transformed
    # conscientious_indeces = _test_data_copy.index[_test_data_copy['Conscientious'] == 0]
    # none_conscientious_indeces = _test_data_copy.index[_test_data_copy['Conscientious'] == 1]
    # file_name = '{}/TabNetClassifier_Predicted_test_data_plot.png'.format(path)
    # plot_data_cluster(transformed_test_x, conscientious_indeces.tolist(), none_conscientious_indeces.tolist(), 
    #             'Tabnet predicted test data plot', file_name, show=False, save=True)

    # # ------- display roc_auc curve
    # lda_roc_auc = roc_auc_score(true_value_test_data["Conscientious"], predictions_transformed)
    # fpr, tpr, thresholds = roc_curve(true_value_test_data["Conscientious"], predictions)#predictions[:,1])
    # file_name = '{}/TabNetClassifier_DL-Model_test-data_ROC-curve.png'.format(path)
    # plot_roc_curve(true_positive_rate = tpr, false_positive_rate = fpr, legend_label = 'DL-Model test data auc (area = %0.2f)' % lda_roc_auc, 
    #             title = 'Tabnet-Model test data', file_name = file_name, show=False, save=True)

    #precision, recall, thresholds = precision_recall_curve(true_value_test_data["Conscientious"], predictions)
    precision, recall, thresholds = precision_recall_curve(t_Y, predictions)
    print(precision)
    print(recall)
    print(thresholds)

    #matrix = confusion_matrix(true_value_test_data['Conscientious'], predictions_transformed)
    matrix = confusion_matrix(t_Y, predictions_transformed)
    print(matrix)
    file_name = '{}/TabNetClassifier_DL-Model_test_data_confusion_Matrix.txt'.format(path)
    write_matrix_and_report_to_file(file_name, np.array2string(matrix))

    #report = classification_report(true_value_test_data['Conscientious'], predictions_transformed)
    report = classification_report(t_Y, predictions_transformed)
    print(report)
    file_name = '{}/TabNetClassifier_DL-Model_test_deta_report.txt'.format(path)
    write_matrix_and_report_to_file(file_name, report)

    #f1_score_value = f1_score(true_value_test_data["Conscientious"], predictions_transformed, average=None)
    f1_score_value = f1_score(t_Y, predictions_transformed, average=None)
    print(f1_score_value)

    #test_acc = accuracy_score(y_pred=true_value_test_data["Conscientious"], y_true=predictions_transformed)
    test_acc = accuracy_score(y_pred=t_Y, y_true=predictions_transformed)
    print(test_acc)

    #display = PrecisionRecallDisplay.from_predictions(true_value_test_data["Conscientious"], predictions_transformed, name="DL-Model")
    display = PrecisionRecallDisplay.from_predictions(t_Y, predictions_transformed, name="DL-Model")
    _ = display.ax_.set_title("2-class Precision-Recall curve")
    file_name = '{}/TabNetClassifier_DL-Model_test-data_Precision-Recall-curve.png'.format(path)
    plt.savefig(file_name)
    plt.close()

    # --- save tabnet model
    file_name = '{}/trained_tabnet_model'.format(path)
    saved_filepath = tab_net_model.save_model(file_name)

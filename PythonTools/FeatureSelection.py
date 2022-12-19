import numpy as np
import numpy.matlib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
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

# drop rows with id == 21
# id_21_list_to_drop = load_test_data[load_test_data.pId == 21].index
# print(id_21_list_to_drop.tolist())
# load_test_data = load_test_data.drop(id_21_list_to_drop.tolist())

true_value_test_data = []
# ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
# ids = [22, 23, 24, 25, 26, 27, 28, 29] drop version
# set real Conscientious values
for i in range(load_test_data.shape[0]):
    true_value_test_data.append([0])
    if load_test_data['pId'].values[i] == 24 or load_test_data['pId'].values[i] == 25:
        true_value_test_data[i] = [1]

# ---------- VarianceThreshold
print("VarianceThreshold feature selector:")
features_selection_model = VarianceThreshold(threshold=(.8 * (1 - .8)))
selected_training_data_features = features_selection_model.fit_transform(input_data.drop(columns=['Conscientious', 'time', 'pId']))
selected_training_features = features_selection_model.get_feature_names_out()
print(selected_training_features)
print(selected_training_data_features.shape)
#print(selected_training_data_features[0:1, :])
selected_test_data_features = load_test_data[selected_training_features].values
# selected_test_data_features = features_selection_model.fit_transform(load_test_data.drop(columns=['time', 'pId']))
# print(features_selection_model.get_feature_names_out())
# print(selected_test_data_features.shape)
# #print(selected_test_data_features[0:1, :])

# ---------- SelectKBest f_classif
# print("SelectKBest f_classif feature selector:")
# features_selection_model = SelectKBest(f_classif, k=2)
# selected_training_data_features = features_selection_model.fit_transform(input_data.drop(columns=['Conscientious', 'time', 'pId']), np.array(input_data[["Conscientious"]].values.flatten()))
# selected_training_features = features_selection_model.get_feature_names_out()
# print(selected_training_features)
# print(selected_training_data_features.shape)
# #print(selected_training_data_features[0:1, :])
# selected_test_data_features = load_test_data[selected_training_features].values
#selected_test_data_features = features_selection_model.fit_transform(load_test_data.drop(columns=['time', 'pId']), true_value_test_data)
#print(features_selection_model.get_feature_names_out())
#print(selected_test_data_features.shape)
#print(selected_test_data_features[0:1, :])

# ---------- SelectKBest f_regression
# print("SelectKBest f_regression feature selector:")
# features_selection_model = SelectKBest(f_regression, k=2)
# selected_training_data_features = features_selection_model.fit_transform(input_data.drop(columns=['Conscientious', 'time', 'pId']), np.array(input_data[["Conscientious"]].values.flatten()))
# print(features_selection_model.get_feature_names_out())
# print(selected_training_data_features.shape)
# #print(selected_training_data_features[0:1, :])
# selected_test_data_features = features_selection_model.fit_transform(load_test_data.drop(columns=['time', 'pId']), true_value_test_data)
# print(features_selection_model.get_feature_names_out())
# print(selected_test_data_features.shape)
# #print(selected_test_data_features[0:1, :])


# selected_c_num = selected_training_data_features.shape[1]
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1)
# for i in range(selected_c_num - 1):
# 	ax.plot(selected_training_data_features[input_data["Conscientious"]==0, i], selected_training_data_features[input_data["Conscientious"]==0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
# 	ax.plot(selected_training_data_features[input_data["Conscientious"]==1, i], selected_training_data_features[input_data["Conscientious"]==1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)    
# ax.set_title("SelectKBest features model results plot")
# ax.set_xticks(())
# ax.set_yticks(())
# plt.show()
# plt.close()

# selected_c_num = selected_test_data_features.shape[1]
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1)
# for i in range(selected_c_num - 1):
# 	ax.plot(selected_test_data_features[:, i], selected_test_data_features[:, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
# ax.set_title("SelectKBest features model results plot")
# ax.set_xticks(())
# ax.set_yticks(())
# plt.show()
# plt.close()


weight_ecg = 1/11       #train_data.loc[:,1:26]                                 -> count() = 26
weight_eda = 1/11       #train_data.loc[:,27:31]                                -> count() = 5
weight_eeg = 4/11       #train_data.loc[:,32:107]  , train_data.loc[:,141:145]  -> count() = 76, 5
weight_eye = 1/11       #train_data.loc[:,108:117] , train_data.loc[:,130:137]  -> count() = 10, 8
weight_pages = 4/11     #train_data.loc[:,118:129] , train_data.loc[:,138:140]  -> count() = 12, 3

# filter train data 
#train_data = input_data.drop(columns=['Conscientious', 'time', 'pId'])
train_data = pd.DataFrame(data=selected_training_data_features)
# get input_data shape
r_num = train_data.shape[0]
print(r_num)
c_num = train_data.shape[1]
print(c_num)

# exc_cols = [col for col in train_data.columns if col not in ['DegTimeLowQuality', 'EvaluatedGlobalTIMERSICalc']]

# train_data.loc[train_data.DegTimeLowQuality > 0, exc_cols] *= 2.0
# train_data.loc[train_data.EvaluatedGlobalTIMERSICalc >= 1, exc_cols] *= 2.0

# filter test data 
#test_data = load_test_data.drop(columns=['time', 'pId'])
test_data = pd.DataFrame(data=selected_test_data_features)

# exc_cols = [col for col in test_data.columns if col not in ['DegTimeLowQuality', 'EvaluatedGlobalTIMERSICalc']]

# test_data.loc[test_data.DegTimeLowQuality > 0, exc_cols] *= 2.0
# test_data.loc[test_data.EvaluatedGlobalTIMERSICalc >= 1, exc_cols] *= 2.0

r_num_test_data = test_data.shape[0]
test_data_x = test_data.iloc[:, :].values

# ------ create normalizer 
# to normalize data for creating more varianz in the data
#transformed_train_data_x = train_data.iloc[:, :].values
#transformed_test_data_x = test_data_x
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

# if input_data_type == 0:
# 	transformed_train_data_x[:,0:26]    = transformed_train_data_x[:,0:26]    * weight_ecg
# 	transformed_train_data_x[:,26:31]   = transformed_train_data_x[:,26:31]   * weight_eda
# 	transformed_train_data_x[:,31:107]  = transformed_train_data_x[:,31:107]  * weight_eeg
# 	transformed_train_data_x[:,140:145] = transformed_train_data_x[:,140:145] * weight_eeg
# 	transformed_train_data_x[:,107:117] = transformed_train_data_x[:,107:117] * weight_eye
# 	transformed_train_data_x[:,129:137] = transformed_train_data_x[:,129:137] * weight_eye
# 	transformed_train_data_x[:,117:129] = transformed_train_data_x[:,117:129] * weight_pages
# 	transformed_train_data_x[:,137:140] = transformed_train_data_x[:,137:140] * weight_pages

# 	transformed_test_data_x[:,0:26]    = transformed_test_data_x[:,0:26]    * weight_ecg
# 	transformed_test_data_x[:,26:31]   = transformed_test_data_x[:,26:31]   * weight_eda
# 	transformed_test_data_x[:,31:107]  = transformed_test_data_x[:,31:107]  * weight_eeg
# 	transformed_test_data_x[:,140:145] = transformed_test_data_x[:,140:145] * weight_eeg
# 	transformed_test_data_x[:,107:117] = transformed_test_data_x[:,107:117] * weight_eye
# 	transformed_test_data_x[:,129:137] = transformed_test_data_x[:,129:137] * weight_eye
# 	transformed_test_data_x[:,117:129] = transformed_test_data_x[:,117:129] * weight_pages
# 	transformed_test_data_x[:,137:140] = transformed_test_data_x[:,137:140] * weight_pages
# if input_data_type == 1:
# 	transformed_train_data_x[:,:] = transformed_train_data_x[:,:] * weight_ecg
# 	transformed_test_data_x[:,:]  = transformed_test_data_x[:,:]  * weight_ecg
# if input_data_type == 2:
# 	transformed_train_data_x[:,:] = transformed_train_data_x[:,:] * weight_eda
# 	transformed_test_data_x[:,:]  = transformed_test_data_x[:,:]  * weight_eda
# if input_data_type == 3:
# 	transformed_train_data_x[:,:] = transformed_train_data_x[:,:] * weight_eeg
# 	transformed_test_data_x[:,:]  = transformed_test_data_x[:,:]  * weight_eeg
# if input_data_type == 4:
# 	transformed_train_data_x[:,:] = transformed_train_data_x[:,:] * weight_eye
# 	transformed_test_data_x[:,:]  = transformed_test_data_x[:,:]  * weight_eye
# if input_data_type == 5:
# 	transformed_train_data_x[:,:] = transformed_train_data_x[:,:] * weight_pages
# 	transformed_test_data_x[:,:]  = transformed_test_data_x[:,:]  * weight_pages

# # data set with out eda features
# result = np.concatenate((transformed_train_data_x[:,0:26], transformed_train_data_x[:,31:107]), axis=1)
# #result = np.concatenate((result, transformed_train_data_x[:,26:31]), axis=1)
# result = np.concatenate((result, transformed_train_data_x[:,107:117]), axis=1)
# #result = np.concatenate((result, transformed_train_data_x[:,117:129]), axis=1)
# result = np.concatenate((result, transformed_train_data_x[:,129:137]), axis=1)
# #result = np.concatenate((result, transformed_train_data_x[:,137:140]), axis=1)
# result = np.concatenate((result, transformed_train_data_x[:,140:145]), axis=1)
# print(result.shape)
# transformed_train_data_x = result

# result = np.concatenate((transformed_test_data_x[:,0:26], transformed_test_data_x[:,31:107]), axis=1)
# #result = np.concatenate((result, transformed_test_data_x[:,26:31]), axis=1)
# result = np.concatenate((result, transformed_test_data_x[:,107:117]), axis=1)
# #result = np.concatenate((result, transformed_test_data_x[:,117:129]), axis=1)
# result = np.concatenate((result, transformed_test_data_x[:,129:137]), axis=1)
# #result = np.concatenate((result, transformed_test_data_x[:,137:140]), axis=1)
# result = np.concatenate((result, transformed_test_data_x[:,140:145]), axis=1)
# print(result.shape)
# transformed_test_data_x = result


linearDiscriminantAnalysis = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
#splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=36851234)
splitter = ShuffleSplit(n_splits=100, train_size=0.4, test_size=.25)

test_fold_predictions = []
for train_index, test_index in splitter.split(transformed_train_data_x, true_value_train_data_y):
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = transformed_train_data_x[train_index], transformed_train_data_x[test_index]
	y_train, y_test = true_value_train_data_y[train_index], true_value_train_data_y[test_index]
	linearDiscriminantAnalysis.fit(X_train, y_train)
	test_fold_predictions.append([y_test, linearDiscriminantAnalysis.predict(X_test)])

_conscientious = linearDiscriminantAnalysis.predict(transformed_train_data_x)
print(_conscientious)
prediction = linearDiscriminantAnalysis.predict_proba(transformed_train_data_x)
_confidence = np.max(prediction, axis = 1)
print(_confidence)
print(transformed_train_data_x.shape)

train_data["Conscientious"] = _conscientious
train_data["Confidence"] = _confidence
train_data["pId"] = input_data["pId"]
colors = {0:'b', 1:'r'}
ax2 = train_data.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=train_data['Conscientious'].map(colors))
plt.show()
plt.close()

lda_roc_auc = roc_auc_score(true_value_train_data_y, linearDiscriminantAnalysis.predict(transformed_train_data_x))
fpr, tpr, thresholds = roc_curve(true_value_train_data_y, linearDiscriminantAnalysis.predict_proba(transformed_train_data_x)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Linear-Discriminant-Analysis-Model (area = %0.2f)' % lda_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Linear-Discriminant-Analysis-Model test data ROC curve')
plt.show()
plt.close()

# --- validation part
colors = {0:'b', 1:'r'}
true_value_test_data = []
test_data['pId'] = load_test_data['pId']
#ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
#ids = [22, 23, 24, 25, 26, 27, 28, 29] dropped version
# set real Conscientious values
for i in range(r_num_test_data):
    true_value_test_data.append([0])
    if test_data['pId'].values[i] == 24 or test_data['pId'].values[i] == 25:
        true_value_test_data[i] = [1]

result_array = linearDiscriminantAnalysis.predict(transformed_test_data_x)
test_data["Conscientious"] = result_array
test_data["Conscientious"] = test_data["Conscientious"].astype("int")

prediction = linearDiscriminantAnalysis.predict_proba(transformed_test_data_x)
test_data["Confidence"] = np.max(prediction, axis = 1)

ax2 = test_data.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=test_data['Conscientious'].map(colors))
plt.show()
plt.close()

# ----------- linearDiscriminantAnalysis Cluster IDs plot with heighest confidence
_ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
#_ids = [22, 23, 24, 25, 26, 27, 28] # dropped version
for id in _ids:
    temp = test_data.loc[test_data["pId"] == id]
    max_confi = temp['Confidence'].max()
    print(max_confi)
    highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
    highest_confidet = temp.at[highest_confidet_index, 'Conscientious']
    test_data.loc[test_data.pId == id, 'Conscientious'] = highest_confidet 
	
ax2 = test_data.plot.scatter(x='Conscientious',  y='pId', c=test_data['Conscientious'].map(colors))
plt.show()
plt.close()

# ------- display roc_auc curve
lda_roc_auc = roc_auc_score(true_value_test_data, linearDiscriminantAnalysis.predict(transformed_test_data_x))
fpr, tpr, thresholds = roc_curve(true_value_test_data, linearDiscriminantAnalysis.predict_proba(transformed_test_data_x)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Linear-Discriminant-Analysis-Model (area = %0.2f)' % lda_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Linear-Discriminant-Analysis-Model test data ROC curve')
plt.show()
plt.close()

import numpy as np
import numpy.matlib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
# weight_ecg = 1/11       #train_data.loc[:,1:26]                                 -> count() = 26
# weight_eda = 1/11       #train_data.loc[:,27:31]                                -> count() = 5
# weight_eeg = 4/11       #train_data.loc[:,32:107]  , train_data.loc[:,141:145]  -> count() = 76, 5
# weight_eye = 1/11       #train_data.loc[:,108:117] , train_data.loc[:,130:137]  -> count() = 10, 8
# weight_pages = 4/11     #train_data.loc[:,118:129] , train_data.loc[:,138:140]  -> count() = 12, 3
# -----
# weight_ecg = 2/5      
# weight_eda = 3/5       
# weight_eeg = 1/5       
# weight_eye = 2/5       
# weight_pages = 1       
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

exc_cols = [col for col in train_data.columns if col not in ['DegTimeLowQuality', 'EvaluatedGlobalTIMERSICalc']]
train_data.loc[train_data.DegTimeLowQuality > 0, exc_cols] *= 2.0
train_data.loc[train_data.EvaluatedGlobalTIMERSICalc >= 1, exc_cols] *= 2.0

# filter test data 
test_data = load_test_data.drop(columns=['time', 'pId'])

exc_cols = [col for col in test_data.columns if col not in ['DegTimeLowQuality', 'EvaluatedGlobalTIMERSICalc']]
test_data.loc[test_data.DegTimeLowQuality > 0, exc_cols] *= 2.0
test_data.loc[test_data.EvaluatedGlobalTIMERSICalc >= 1, exc_cols] *= 2.0

#r_num_test_data = test_data.shape[0]
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

if input_data_type == 1:
	transformed_train_data_x[:,:] = transformed_train_data_x[:,:] * weight_ecg
	transformed_test_data_x[:,:]  = transformed_test_data_x[:,:]  * weight_ecg
if input_data_type == 2:
	transformed_train_data_x[:,:] = transformed_train_data_x[:,:] * weight_eda
	transformed_test_data_x[:,:]  = transformed_test_data_x[:,:]  * weight_eda
if input_data_type == 3:
	transformed_train_data_x[:,:] = transformed_train_data_x[:,:] * weight_eeg
	transformed_test_data_x[:,:]  = transformed_test_data_x[:,:]  * weight_eeg
if input_data_type == 4:
	transformed_train_data_x[:,:] = transformed_train_data_x[:,:] * weight_eye
	transformed_test_data_x[:,:]  = transformed_test_data_x[:,:]  * weight_eye
if input_data_type == 5:
	transformed_train_data_x[:,:] = transformed_train_data_x[:,:] * weight_pages
	transformed_test_data_x[:,:]  = transformed_test_data_x[:,:]  * weight_pages

# --- data set with out eda features
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

print("Create output directory")
# --- create dir
mode = 0o666
if not os.path.exists("./output"):
    os.mkdir("./output", mode)
path = "./output/Linear-Discriminant-Analysis_{}".format(input_data_type)
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

# ------ create and train linearDiscriminantAnalysis
#linearDiscriminantAnalysis = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
linearDiscriminantAnalysis = LinearDiscriminantAnalysis()
# ------ training lda fit
#linearDiscriminantAnalysis.fit(transformed_train_data_x, true_value_train_data_y)
linearDiscriminantAnalysis.fit(transformed_train_data_x)

sys.exit()

n_repeats=5
# ------ training lda with splitter : train_test_split 
#X_train, X_test, y_train, y_test = train_test_split(transformed_train_data_x, true_value_train_data_y, test_size=0.4, random_state=0)
#linearDiscriminantAnalysis.fit(X_train, y_train)
# ------ training lda with splitter  
# splitter = ShuffleSplit(n_splits=20, train_size=0.4, test_size=0.4)
# for _ in range(n_repeats):
# 	for train_index, test_index in splitter.split(transformed_train_data_x):
# 		print("TRAIN:", train_index, "TEST:", test_index)
# 		X_train, X_test = transformed_train_data_x[train_index], transformed_train_data_x[test_index]
# 		y_train, y_test = true_value_train_data_y[train_index], true_value_train_data_y[test_index]
# 		linearDiscriminantAnalysis.fit(X_train, y_train)

# splitter = RepeatedStratifiedKFold(n_splits=10, n_repeats=n_repeats)
# for train_index, test_index in splitter.split(transformed_train_data_x, true_value_train_data_y):
# 	print("TRAIN:", train_index, "TEST:", test_index)
# 	X_train, X_test = transformed_train_data_x[train_index], transformed_train_data_x[test_index]
# 	y_train, y_test = true_value_train_data_y[train_index], true_value_train_data_y[test_index]
# 	linearDiscriminantAnalysis.fit(X_train, y_train)

_conscientious = linearDiscriminantAnalysis.predict(transformed_train_data_x)
print(_conscientious)
prediction = linearDiscriminantAnalysis.predict_proba(transformed_train_data_x)
_confidence = np.max(prediction, axis = 1)
print(_confidence)
print(transformed_train_data_x.shape)
print("acc train data")
print(accuracy_score(true_value_train_data_y, _conscientious))

conscientious_indeces = input_data.index[_conscientious == 0]
none_conscientious_indeces = input_data.index[_conscientious == 1]

file_name = '{}/Transformed_predicted_train_data_plot.png'.format(path)
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1, 1, 1)
for i in range(c_num - 1):
    ax.scatter(transformed_train_data_x[conscientious_indeces.tolist(), i], transformed_train_data_x[conscientious_indeces.tolist(), i+1], c="b")
    ax.scatter(transformed_train_data_x[none_conscientious_indeces.tolist(), i], transformed_train_data_x[none_conscientious_indeces.tolist(), i+1], c="r")
ax.set_title("Transformed (True) train data  plot", fontsize=16)
plt.savefig(file_name)
plt.close()

lda_roc_auc = roc_auc_score(true_value_train_data_y, linearDiscriminantAnalysis.predict(transformed_train_data_x))
fpr, tpr, thresholds = roc_curve(true_value_train_data_y, linearDiscriminantAnalysis.predict_proba(transformed_train_data_x)[:,1])
file_name = '{}/Linear-Discriminant-Analysis-Model_train_data_roc-curve.png'.format(path)
plot_roc_curve(true_positive_rate=tpr, false_positive_rate=fpr, legend_label='Linear-Discriminant-Analysis-Model train data (area = %0.2f)' % lda_roc_auc, 
               title='Linear-Discriminant-Analysis-Model train data', file_name=file_name, show=False, save=True)

# --- validation part
test_data["Conscientious"] = linearDiscriminantAnalysis.predict(transformed_test_data_x)
test_data["Conscientious"] = test_data["Conscientious"].astype("int")

prediction = linearDiscriminantAnalysis.predict_proba(transformed_test_data_x)
test_data["Confidence"] = np.max(prediction, axis = 1)

colors = {0:'b', 1:'r'}
test_data['pId'] = load_test_data['pId']
# ax2 = test_data.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=test_data['Conscientious'].map(colors))
# plt.show()
# plt.close()

# ----------- linearDiscriminantAnalysis Cluster IDs plot with heighest confidence
_ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
for id in _ids:
	temp = test_data.loc[test_data["pId"] == id]
	max_confi = temp['Confidence'].max()
	highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
	highest_confidet = temp.at[highest_confidet_index, 'Conscientious']
	test_data.loc[test_data.pId == id, 'Conscientious'] = highest_confidet 
	
# ax2 = test_data.plot.scatter(x='Conscientious',  y='pId', c=test_data['Conscientious'].map(colors))
# plt.show()
# plt.close()
print("acc test data")
print(accuracy_score(true_value_test_data['Conscientious'], test_data["Conscientious"]))

# ------- display roc_auc curve
lda_roc_auc = roc_auc_score(true_value_test_data['Conscientious'], linearDiscriminantAnalysis.predict(transformed_test_data_x))
fpr, tpr, thresholds = roc_curve(true_value_test_data['Conscientious'], linearDiscriminantAnalysis.predict_proba(transformed_test_data_x)[:,1])
file_name = '{}/Linear-Discriminant-Analysis-Model_test_data_roc-curve.png'.format(path)
plot_roc_curve(true_positive_rate=tpr, false_positive_rate=fpr, legend_label='Linear-Discriminant-Analysis-Model test data (area = %0.2f)' % lda_roc_auc, 
               title='Linear-Discriminant-Analysis-Model test data', file_name=file_name, show=False, save=True)

# ------ test data plot
conscientious_indeces = test_data.index[test_data["Conscientious"] == 0]
none_conscientious_indeces = test_data.index[test_data["Conscientious"] == 1]

file_name = '{}/Transformed_predicted_test_data_plot.png'.format(path)
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1, 1, 1)
for i in range(c_num - 1):
    ax.scatter(transformed_test_data_x[conscientious_indeces.tolist(), i], transformed_test_data_x[conscientious_indeces.tolist(), i+1], c="b")
    ax.scatter(transformed_test_data_x[none_conscientious_indeces.tolist(), i], transformed_test_data_x[none_conscientious_indeces.tolist(), i+1], c="r")
ax.set_title("Transformed (True) test data plot", fontsize=16)
plt.savefig(file_name)
plt.close()


sys.exit()

# lda_transformed_train_data_x = linearDiscriminantAnalysis.transform(transformed_train_data_x)
# print(lda_transformed_train_data_x.shape)
# print(lda_transformed_train_data_x[0:5])
# lda_c_num = lda_transformed_train_data_x.shape[1]
# fig = plt.figure(figsize=(10, 10))
# colors = ["#4EACC5", "#FF9C34"]
# ax = fig.add_subplot(1, 1, 1)
# for i in range(lda_c_num - 1):
# 	ax.plot(lda_transformed_train_data_x[input_data["Conscientious"]==0, i], lda_transformed_train_data_x[input_data["Conscientious"]==0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
# 	ax.plot(lda_transformed_train_data_x[input_data["Conscientious"]==1, i], lda_transformed_train_data_x[input_data["Conscientious"]==1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)
# ax.set_title("Linear Discriminant Analysis transformed training data with true cluster values  Plot")
# ax.set_xticks(())
# ax.set_yticks(())
# plt.show()
# plt.close()

# define method to evaluate model
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
#scores = cross_val_score(linearDiscriminantAnalysis, transformed_train_data_x, true_value_train_data_y, scoring='accuracy', cv=cv, n_jobs=-1)
#print("=========================== evaluate model")
#print(scores)
#print(np.mean(scores))   

df11=pd.DataFrame(linearDiscriminantAnalysis.coef_[0].reshape(-1,1), train_data.columns, columns=["Weight"])
df12=pd.DataFrame(linearDiscriminantAnalysis.intercept_[0].reshape(-1,1), ["Bias"], columns=["Weight"])
resulty = pd.concat([df12, df11], axis=0)
print("====================== fit informations")
print(resulty)

train_data["Conscientious"] = linearDiscriminantAnalysis.predict(transformed_train_data_x)
train_data["Conscientious"] = train_data["Conscientious"].astype("int")

prediction = linearDiscriminantAnalysis.predict_proba(transformed_train_data_x)
train_data["Confidence"] = np.max(prediction, axis = 1)

plt.figure(figsize=(15,7))
plt.hist(train_data['Confidence'][train_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.7, color='b')
plt.hist(train_data['Confidence'][train_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.7, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 

print(linearDiscriminantAnalysis.get_params(deep=True))

# # Linear Discriminant Analysis
# fig = plt.figure(figsize=(10, 10))
# colors = ["#4EACC5", "#FF9C34"]
# ax = fig.add_subplot(1, 1, 1)
# for i in range(c_num - 1):
# 	ax.plot(transformed_train_data_x[train_data["Conscientious"]==0, i], transformed_train_data_x[train_data["Conscientious"]==0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
# 	ax.plot(transformed_train_data_x[train_data["Conscientious"]==1, i], transformed_train_data_x[train_data["Conscientious"]==1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)
# ax.set_title("Linear Discriminant Analysis Training Data Plot")
# ax.set_xticks(())
# ax.set_yticks(())
# plt.show()

train_data["pId"] = input_data["pId"]

colors = {0:'b', 1:'r'}
plt.scatter(x=train_data['Conscientious'], y=train_data['pId'], alpha=0.5, c=train_data['Conscientious'].map(colors))
plt.show()

ax2 = train_data.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=train_data['Conscientious'].map(colors))
plt.show()

_ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,31,34]
for id in _ids:
	temp = train_data.loc[train_data["pId"] == id]
	max_confi = temp['Confidence'].max()
	highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
	highest_confidet = temp.at[highest_confidet_index, 'Conscientious']
	train_data.Conscientious.loc[train_data.pId == id] = highest_confidet
	
ax2 = train_data.plot.scatter(x='Conscientious',  y='pId', c=train_data['Conscientious'].map(colors))
plt.show()

# ------- display roc_auc curve
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
plt.savefig('Linear-Discriminant-Analysis-Model train data ROC curve')
plt.show()

print("================ transformend test validation input predictions informations")
true_value_test_data = []
test_data['pId'] = load_test_data['pId']
#ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
# set real Conscientious values
for i in range(r_num_test_data):
    true_value_test_data.append([0])
    if test_data['pId'].values[i] == 24 or test_data['pId'].values[i] == 25:
        true_value_test_data[i] = [1]

# test_data["PredictionScoreFactor"] = pd.Series(data = np.zeros(r_num_test_data))
# result_array = []
# for i in range(r_num_test_data):
#     input_sample = transformed_test_data_x[1].reshape(1, -1)
#     predicted_result = linearDiscriminantAnalysis.predict(input_sample)
#     result_array.append(predicted_result)

# print(accuracy_score(true_value_test_data, result_array))
# print(confusion_matrix(true_value_test_data, result_array))
# print(classification_report(true_value_test_data, result_array))

# test_data['Conscientious'] = pd.Series(result_array)

result_array = linearDiscriminantAnalysis.predict(transformed_test_data_x)
test_data["Conscientious"] = result_array
test_data["Conscientious"] = test_data["Conscientious"].astype("int")

prediction = linearDiscriminantAnalysis.predict_proba(transformed_test_data_x)
test_data["Confidence"] = np.max(prediction, axis = 1)

plt.figure(figsize=(15,7))
plt.hist(test_data['Confidence'][test_data["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.7, color='b')
plt.hist(test_data['Confidence'][test_data["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.7, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 

fig = plt.figure(figsize=(10, 10))
colors = ["#4EACC5", "#FF9C34"]
ax = fig.add_subplot(1, 1, 1)
for i in range(c_num - 1):
	ax.plot(transformed_test_data_x[true_value_test_data == 0, i], transformed_test_data_x[true_value_test_data == 0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
	ax.plot(transformed_test_data_x[true_value_test_data == 1, i], transformed_test_data_x[true_value_test_data == 1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)
ax.set_title("Linear Discriminant Analysis Test Data Plot")
ax.set_xticks(())
ax.set_yticks(())
plt.show()

colors = {0:'b', 1:'r'}
plt.scatter(x=test_data['Conscientious'], y=test_data['pId'], alpha=0.5, c=test_data['Conscientious'].map(colors))
plt.show()

ax2 = test_data.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=test_data['Conscientious'].map(colors))
plt.show()

# ----------- linearDiscriminantAnalysis Cluster IDs plot with heighest confidence
_ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
for id in _ids:
	temp = test_data.loc[test_data["pId"] == id]
	max_confi = temp['Confidence'].max()
	highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
	highest_confidet = temp.at[highest_confidet_index, 'Conscientious']
	test_data.Conscientious.loc[test_data.pId == id] = highest_confidet
	
ax2 = test_data.plot.scatter(x='Conscientious',  y='pId', c=test_data['Conscientious'].map(colors))
plt.show()

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

import numpy as np
import numpy.matlib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# read csv train data
#input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')			# plan of sensors weighting:
#input_data = pd.read_csv("All_Participents_ECG_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/9
#input_data = pd.read_csv("All_Participents_EDA_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/9
#input_data = pd.read_csv("All_Participents_EEG_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/9
#input_data = pd.read_csv("All_Participents_EYE_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/9
input_data = pd.read_csv("All_Participents_PAGES_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',') 	# weight with 3/9

# read cvs test data
#load_test_data = pd.read_csv("All_Participents_Condition-C_WaveSum_DataFrame.csv", sep=";", decimal=',')			# plan of sensors weighting:
#load_test_data = pd.read_csv("All_Participents_Condition-C_ECG_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/9
#load_test_data = pd.read_csv("All_Participents_Condition-C_EDA_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/9
#load_test_data = pd.read_csv("All_Participents_Condition-C_EEG_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 1/9
#load_test_data = pd.read_csv("All_Participents_Condition-C_EYE_WaveSum_DataFrame.csv", sep=";", decimal=',') 		# weight with 2/9
load_test_data = pd.read_csv("All_Participents_Condition-C_PAGES_WaveSum_DataFrame.csv", sep=";", decimal=',') 	# weight with 3/9

# set real conscientius values
# for i in range(r_num):
#     if input_data["pId"].values[i] == 14 or input_data["pId"].values[i] == 15 or input_data["pId"].values[i] == 16:
#         input_data['Conscientious'].values[i] = 0

# filter train data 
input_x = input_data.drop(columns=['Conscientious', 'time', 'pId'])
# get input_data shape
r_num = input_x.shape[0]
print(r_num)
c_num = input_x.shape[1]
print(c_num)

# filter test data 
test_data = load_test_data.drop(columns=['time', 'pId'])
r_num_test_data = test_data.shape[0]
test_x = test_data.iloc[:, :].values

# create normalizer 
# to normalize data for creating more varianz in the data
scaler = StandardScaler()

# ------ normalize test data
scaler.fit(test_x)
transformed_test_x = scaler.transform(test_x)

# ------ normalize train data
# separating out the features
x = input_x.iloc[:, :].values
scaler.fit(x)
transformed_x = scaler.transform(x)

# separating train output data as target 'Conscientious'
y = np.array(input_data[["Conscientious"]].values.flatten()) 

# ------ create and train linearDiscriminantAnalysis
linearDiscriminantAnalysis = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
linearDiscriminantAnalysis.fit(transformed_x, y)

# define method to evaluate model
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
#scores = cross_val_score(linearDiscriminantAnalysis, transformed_x, y, scoring='accuracy', cv=cv, n_jobs=-1)
#print("=========================== evaluate model")
#print(scores)
#print(np.mean(scores))   

df11=pd.DataFrame(linearDiscriminantAnalysis.coef_[0].reshape(-1,1), input_x.columns, columns=["Weight"])
df12=pd.DataFrame(linearDiscriminantAnalysis.intercept_[0].reshape(-1,1), ["Bias"], columns=["Weight"])
resulty = pd.concat([df12, df11], axis=0)
print("====================== fit informations")
#np.set_printoptions(threshold=np.inf)
print(resulty)

result_array = linearDiscriminantAnalysis.predict(transformed_x)
input_x["Conscientious"] = result_array
input_x["Conscientious"] = input_x["Conscientious"].astype("int")

prediction = linearDiscriminantAnalysis.predict_proba(transformed_x)
input_x["Confidence"] = np.max(prediction, axis = 1)

plt.figure(figsize=(15,7))
#plt.hist(prediction[:,1][input_data['Conscientious']==0], bins=50, label='Cluster Conscientious', alpha=0.7)
plt.hist(input_x['Confidence'][input_x["Conscientious"]==0], bins=50, label='Cluster Conscientious', alpha=0.7, color='b')
plt.hist(input_x['Confidence'][input_x["Conscientious"]==1], bins=50, label='Cluster None-Conscientious', alpha=0.7, color='r')
plt.xlabel('Calculated Probability', fontsize=25)
plt.ylabel('Number of records', fontsize=25)
plt.legend(fontsize=15)
plt.tick_params(axis='both', labelsize=25, pad=5)
plt.show() 

print(linearDiscriminantAnalysis.get_params(deep=True))

# Linear Discriminant Analysis
fig = plt.figure(figsize=(10, 10))
colors = ["#4EACC5", "#FF9C34"]
ax = fig.add_subplot(1, 1, 1)
for i in range(c_num - 1):
	ax.plot(transformed_x[0, i], transformed_x[0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
	ax.plot(transformed_x[1, i], transformed_x[1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)
#cluster_center = input_mean[0]
#ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='b', markeredgecolor="k", markersize=6, zorder=2)
#cluster_center = input_mean[1]
#ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='r', markeredgecolor="k", markersize=6, zorder=2)
ax.set_title("Linear Discriminant Analysis Training Data Plot")
ax.set_xticks(())
ax.set_yticks(())
plt.show()

input_x["pId"] = input_data["pId"]

colors = {0:'b', 1:'r'}
plt.scatter(x=input_x['Conscientious'], y=input_x['pId'], alpha=0.5, c=input_x['Conscientious'].map(colors))
plt.show()

ax2 = input_x.plot.scatter(x='pId',  y='Confidence', alpha=0.5, c=input_x['Conscientious'].map(colors))
plt.show()

_ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20,31,34]
for id in _ids:
	temp = input_x.loc[input_x["pId"] == id]
	max_confi = temp['Confidence'].max()
	highest_confidet_index = temp[temp.Confidence == max_confi].index.tolist()[0]
	temp['Conscientious'] = temp.at[highest_confidet_index, 'Conscientious']
	input_x.Conscientious[input_x.pId == id] = temp['Conscientious'].values[0]
	
ax2 = input_x.plot.scatter(x='Conscientious',  y='pId', c=input_x['Conscientious'].map(colors))
plt.show()

print("================ transformend test validation input predictions informations")
true_value_test_data = []
test_data['pId'] = load_test_data['pId']
#ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
# set real Conscientious values
for i in range(r_num_test_data):
    true_value_test_data.append([0])
    if test_data['pId'].values[i] == 24 or test_data['pId'].values[i] == 25 or test_data['pId'].values[i] == 29:
        true_value_test_data[i] = [1]

# test_data["PredictionScoreFactor"] = pd.Series(data = np.zeros(r_num_test_data))
# result_array = []
# for i in range(r_num_test_data):
#     input_sample = transformed_test_x[1].reshape(1, -1)
#     predicted_result = linearDiscriminantAnalysis.predict(input_sample)
#     result_array.append(predicted_result)

# print(accuracy_score(true_value_test_data, result_array))
# print(confusion_matrix(true_value_test_data, result_array))
# print(classification_report(true_value_test_data, result_array))

# test_data['Conscientious'] = pd.Series(result_array)

result_array = linearDiscriminantAnalysis.predict(transformed_test_x)
test_data["Conscientious"] = result_array
test_data["Conscientious"] = test_data["Conscientious"].astype("int")

prediction = linearDiscriminantAnalysis.predict_proba(transformed_test_x)
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
	ax.plot(transformed_test_x[0, i], transformed_test_x[0, i+1], "w", markerfacecolor='b', marker=".", zorder=1)
	ax.plot(transformed_test_x[1, i], transformed_test_x[1, i+1], "w", markerfacecolor='r', marker=".", zorder=1)
#cluster_center = input_mean[0]
#ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='b', markeredgecolor="k", markersize=6, zorder=2)
#cluster_center = input_mean[1]
#ax.plot(cluster_center[0], cluster_center[1], "o", markerfacecolor='r', markeredgecolor="k", markersize=6, zorder=2)
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
	temp['Conscientious'] = temp.at[highest_confidet_index, 'Conscientious']
	test_data.Conscientious[test_data.pId == id] = temp['Conscientious'].values[0]
	
ax2 = test_data.plot.scatter(x='Conscientious',  y='pId', c=test_data['Conscientious'].map(colors))
plt.show()

# ------- display roc_auc curve
lda_roc_auc = roc_auc_score(true_value_test_data, linearDiscriminantAnalysis.predict(transformed_test_x))
fpr, tpr, thresholds = roc_curve(true_value_test_data, linearDiscriminantAnalysis.predict_proba(transformed_test_x)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Linear-Discriminant-Analysis-Model (area = %0.2f)' % lda_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Linear-Discriminant-Analysis-Model ROC curve')
plt.show()

#ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
#for id in ids:
#    temp = test_data.loc[test_data["pId"] == id]
#    first =  temp[temp.Conscientious == 0].shape[0]
#    second =  temp[temp.Conscientious == 1].shape[0]
#    print(first)
#    print(second)
#    if first > second: 
#        test_data.Conscientious[test_data.pId == id] = 0
#    if first < second: 
#        test_data.Conscientious[test_data.pId == id] = 1
#ax2 = test_data.plot.scatter(x='Conscientious', y='pId', c='Conscientious', colormap='viridis')
# show the plot
#plt.show()

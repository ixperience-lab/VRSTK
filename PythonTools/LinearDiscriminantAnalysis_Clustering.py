import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')

#input_data.set_index('time').plot()

r_num = input_data.shape[0]
print(r_num)
c_num = input_data.shape[1]
print(c_num)

# nomalize input data to create more varianz in the data
scaler = StandardScaler()

for i in range(r_num):
    if input_data["pId"].values[i] == 14 or input_data["pId"].values[i] == 15 or input_data["pId"].values[i] == 16:
        input_data['Conscientious'].values[i] = 0

load_test_data = pd.read_csv("All_Participents_Condition-C_WaveSum_DataFrame.csv", sep=";", decimal=',')

test_data = load_test_data.drop(columns=['time', 'pId'])
r_num_test_data = test_data.shape[0]
test_x = test_data.iloc[:, :].values
scaler.fit(test_x)
#print(scaler.mean_)
transformed_test_x = scaler.transform(test_x)


#lda_x_data = input_data.drop(columns=['Conscientious', 'time'])
lda_x_data = input_data.drop(columns=['Conscientious', 'time', 'pId'])

#print(lda_x_data.head(1))
#print(test_data.head(1))


#------ Normalizing
# Separating out the features
x = lda_x_data.iloc[:, :].values
scaler.fit(x)
#print(scaler.mean_)
transformed_x = scaler.transform(x)

# Separating out the target
y = np.array(input_data[["Conscientious"]].values.flatten()) #input_data[["Conscientious"]].values.to_numpy() #input_data.iloc[:,154].values#y = input_data.iloc[:,['Conscientious']].values

lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
lda.fit(transformed_x, y)

#Define method to evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#evaluate model
scores = cross_val_score(lda, transformed_x, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("=========================== evaluate model")
print(scores)
print(np.mean(scores))   

df11=pd.DataFrame(lda.coef_[0].reshape(-1,1), lda_x_data.columns, columns=["Weight"])
df12=pd.DataFrame(lda.intercept_[0].reshape(-1,1), ["Bias"], columns=["Weight"])
resulty = pd.concat([df12, df11], axis=0)
print("====================== fit informations")
np.set_printoptions(threshold=np.inf)
print(resulty)

result_array = lda.predict(transformed_x)
#print(result_array)

lda_x_data['Conscientious'] = pd.Series(result_array)
#lda_x_data['pId'] = input_data['pId']
#ax2 = lda_x_data.plot.scatter(x='Conscientious', y='pId', c='Conscientious', colormap='viridis')
# show the plot
#plt.show()

print("================ transformend test validation input predictions informations")
true_value_test_data = []
test_data['pId'] = load_test_data['pId']
#ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
for i in range(r_num_test_data):
    true_value_test_data.append([0])
    if test_data['pId'].values[i] == 24 or test_data['pId'].values[i] == 25 or test_data['pId'].values[i] == 29:
        true_value_test_data[i] = [1]
#print(true_value_test_data)

#test_data["PredictionScoreFactor_0"] = pd.Series(data = np.zeros(r_num_test_data))
#test_data["PredictionScoreFactor_1"] = pd.Series(data = np.zeros(r_num_test_data))
test_data["PredictionScoreFactor"] = pd.Series(data = np.zeros(r_num_test_data))
result_array = []
for i in range(r_num_test_data):
    input_sample = transformed_test_x[1].reshape(1, -1)
    predicted_result = lda.predict(input_sample)
    result_array.append(predicted_result)
    #calculatet_score = lda.score(input_sample, true_value_test_data[i])
    #test_data.PredictionScoreFactor[i] = calculatet_score
    #test_data.PredictionScoreFactor_0[i] = lda.predict_proba(input_sample)[0][0]
    #test_data.PredictionScoreFactor_1[i] = lda.predict_proba(input_sample)[0][1]
    #print(lda.predict_proba(input_sample))
    #print(lda.predict_log_proba(input_sample))

#print(result_array)

print(accuracy_score(true_value_test_data, result_array))
print(confusion_matrix(true_value_test_data, result_array))
print(classification_report(true_value_test_data, result_array))

test_data['Conscientious'] = pd.Series(result_array)
#test_data['pId'] = load_test_data['pId']
#ax2 = test_data.plot.scatter(x='PredictionScoreFactor_0', y='PredictionScoreFactor_1', c='Conscientious', colormap='viridis')
ax2 = test_data.plot.scatter(x='Conscientious', y='pId', c='Conscientious', colormap='viridis')
# show the plot
plt.show()


lda_roc_auc = roc_auc_score(true_value_test_data, lda.predict(transformed_test_x))
fpr, tpr, thresholds = roc_curve(true_value_test_data, lda.predict_proba(transformed_test_x)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='LDA Model (area = %0.2f)' % lda_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('LDA_ROC')
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

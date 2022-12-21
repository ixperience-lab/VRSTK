# Link: https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b


#importing libraries
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#import statsmodels.api as sm
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import sys
import os

from scipy import stats
#Loading the dataset
#x = load_boston()
#df = pd.DataFrame(x.data, columns = x.feature_names)
#df["MEDV"] = x.target
#X = df.drop("MEDV",1)   #Feature Matrix
#y = df["MEDV"]          #Target Variable
#df.head()

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

# # -------  filter columns of test data 
test_data = load_test_data.drop(columns=['time', 'pId'])

# r_num_test_data = test_data.shape[0]
# test_x = test_data.iloc[:, :].values
# print("================ transformend test validation input predictions informations")
# true_value_test_data = []
# # ids = [21, 22, 23, 24, 25, 26, 27, 28, 29]
# # set real Conscientious values
# for i in range(r_num_test_data):
#     true_value_test_data.append(0)
#     if load_test_data['pId'].values[i] == 24 or load_test_data['pId'].values[i] == 25: # or load_test_data['pId'].values[i] == 28:
#         true_value_test_data[i] = 1
# true_value_test_data = pd.DataFrame({ "Conscientious" : true_value_test_data})      
# print(true_value_test_data["Conscientious"].values)

# # ------ Normalizing
# # Separating out the features
# x_train = train_data.loc[:, :].values
# # Separating out the target
# y_result_output = np.array(input_data[["Conscientious"]].values.flatten())
# print(y_result_output)
# # Standardizing the features of train data
# transformed_train_x = StandardScaler().fit_transform(x_train)
# # Standardizing the features of Test data
# transformed_test_x = StandardScaler().fit_transform(test_x)

# set sensor and validity score weights
# weight_ecg = 2/5       
# weight_eda = 3/5       
# weight_eeg = 1/5       
# weight_eye = 3/5       
# weight_pages = 1       

#print(train_data.iloc[:,0:26])
ecg_pages_dataframe = pd.concat([train_data.iloc[:,0:26], train_data.iloc[:,129:141], train_data.iloc[:,149:152]], axis=1)
#print(ecg_pages_dataframe.head(1))

eda_pages_dataframe = pd.concat([train_data.iloc[:,26:31], train_data.iloc[:,129:141], train_data.iloc[:,149:152]], axis=1)

eeg_pages_dataframe = pd.concat([train_data.iloc[:,31:107], train_data.iloc[:,152:157], train_data.iloc[:,129:141], train_data.iloc[:,149:152]], axis=1)

eye_pages_dataframe = pd.concat([train_data.iloc[:,107:129], train_data.iloc[:,141:149], train_data.iloc[:,129:141], train_data.iloc[:,149:152]], axis=1)

#sys.exit()

# if input_data_type == 0:
#     transformed_train_x[:,0:26]    = transformed_train_x[:,0:26]    * weight_ecg
#     transformed_train_x[:,26:31]   = transformed_train_x[:,26:31]   * weight_eda
#     transformed_train_x[:,31:107]  = transformed_train_x[:,31:107]  * weight_eeg
#     transformed_train_x[:,152:157] = transformed_train_x[:,152:157] * weight_eeg
#     transformed_train_x[:,107:129] = transformed_train_x[:,107:129] * weight_eye
#     transformed_train_x[:,141:149] = transformed_train_x[:,141:149] * weight_eye
#     transformed_train_x[:,129:141] = transformed_train_x[:,129:141] * weight_pages
#     transformed_train_x[:,149:152] = transformed_train_x[:,149:152] * weight_pages
    
#     transformed_test_x[:,0:26]    = transformed_test_x[:,0:26]    * weight_ecg
#     transformed_test_x[:,26:31]   = transformed_test_x[:,26:31]   * weight_eda
#     transformed_test_x[:,31:107]  = transformed_test_x[:,31:107]  * weight_eeg
#     transformed_test_x[:,152:157] = transformed_test_x[:,152:157] * weight_eeg
#     transformed_test_x[:,107:129] = transformed_test_x[:,107:129] * weight_eye
#     transformed_test_x[:,141:149] = transformed_test_x[:,141:149] * weight_eye
#     transformed_test_x[:,129:141] = transformed_test_x[:,129:141] * weight_pages
#     transformed_test_x[:,149:152] = transformed_test_x[:,149:152] * weight_pages


print("Create output directory")
# --- create dir
mode = 0o666
if not os.path.exists("./output"):
    os.mkdir("./output", mode)
path = "./output/Correlation_Matrix_Sklearn_{}".format(input_data_type)
if not os.path.exists(path):
    os.mkdir(path, mode)

def show_correlationMatrix(data_frame):
    #Using Pearson Correlation ecg
    plt.figure(figsize=(17,10))
    correlation = data_frame.corr()
    sns.heatmap(correlation, annot=True)#, cmap=plt.cm.Reds)
    plt.show()

    #Correlation with output variable
    print("EvaluatedGlobalTIMERSICalc")
    cor_target = abs(correlation["GlobalTIMERSICalc"])
    #Selecting highly correlated features
    relevant_features = cor_target[cor_target > 0]
    print(relevant_features)

    #Correlation with output variable
    print("\nDegTimeLowQuality")
    cor_target = abs(correlation["DegTimeLowQuality"])
    #Selecting highly correlated features
    relevant_features = cor_target[cor_target > 0]
    print(relevant_features)

#show_correlationMatrix(ecg_pages_dataframe)

#show_correlationMatrix(eda_pages_dataframe)

#show_correlationMatrix(eeg_pages_dataframe)

#show_correlationMatrix(eye_pages_dataframe)

# Kruskal-Wallis-Test durchführen 
result = stats.kruskal(train_data.iloc[:,0], train_data.iloc[:,151])
print(result)

sys.exit()

# highly correlated with each other -> keep only one variable and drop the other
#print(input_data[["HeartRate","int"]].corr())
#print(input_data[["str","HeartRate"]].corr())


f = plt.figure(figsize=(28, 32))
plt.matshow(input_data.corr(), fignum=f.number)
plt.xticks(range(input_data.select_dtypes(['number']).shape[1]), input_data.select_dtypes(['number']).columns, fontsize=8, rotation=45)
plt.yticks(range(input_data.select_dtypes(['number']).shape[1]), input_data.select_dtypes(['number']).columns, fontsize=8)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=8)
plt.title('Correlation Matrix', fontsize=16);
plt.show()

from pandas.plotting import radviz
plt.figure();
radviz(input_data, "pId");
plt.show()

from re import S
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
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

# read csv train data as pandas data frame
input_data_copy = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrameKopie.csv", sep=";", decimal=',')			# plan of sensors weighting:
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

# figure id = 13
# features: pId, LeftPupilDiameter, LeftEyeOpenness, RightPupilDiameter, RightEyeOpenness, ActivatedModelIndex, CognitiveActivityRightPupilDiamter
# mean_value_zombie_pupil_size = input_data_copy.loc[(input_data_copy["ActivatedModelIndex"] == 13) & 
#                                                    (input_data_copy["LeftEyeOpenness"] > 0.8) & (input_data_copy["RightEyeOpenness"] > 0.8)]

input_data_copy["LeftPupilDiameter_scaled"] = MinMaxScaler().fit_transform(input_data_copy[["LeftPupilDiameter"]])
input_data_copy["RightPupilDiameter_scaled"] = MinMaxScaler().fit_transform(input_data_copy[["RightPupilDiameter"]])
input_data_copy["CognitiveActivityLeftPupilDiamter_scaled"] = MinMaxScaler().fit_transform(input_data_copy[["CognitiveActivityLeftPupilDiamter"]])
input_data_copy["CognitiveActivityRightPupilDiamter_scaled"] = MinMaxScaler().fit_transform(input_data_copy[["CognitiveActivityRightPupilDiamter"]])
print(input_data_copy.head(1))

# sys.exit()

l_pupil_d_scaled = []
r_pupil_d_scaled = []
l_c_a_pupil_d_scaled = []
r_c_a_pupil_d_scaled = []

avg_l_pupil_d = []
max_l_pupil_d = []
avg_r_pupil_d = []
max_r_pupil_d = []
avg_l_c_a_pupil_d = []
max_l_c_a_pupil_d = []
avg_r_c_a_pupil_d = []
max_r_c_a_pupil_d = []

for id in range(16):
    print("--------------------------------- id: ", id)
    mean_value_zombie_pupil_size = input_data_copy.loc[(input_data_copy["ActivatedModelIndex"] == id) & (input_data_copy["Conscientious"] == 1) & 
                                                       (input_data_copy["LeftEyeOpenness"] > 0.8) & (input_data_copy["RightEyeOpenness"] > 0.8)]

    l_pupil_d_scaled.append(mean_value_zombie_pupil_size["LeftPupilDiameter_scaled"].values.reshape(1,-1)[0].sum())
    
    print("AVG pupil size left: ", mean_value_zombie_pupil_size[["LeftPupilDiameter_scaled"]].mean())
    avg_l_pupil_d.append(mean_value_zombie_pupil_size[["LeftPupilDiameter_scaled"]].mean())
    print("MAX pupil size left: ", mean_value_zombie_pupil_size[["LeftPupilDiameter_scaled"]].max())
    max_l_pupil_d.append(mean_value_zombie_pupil_size[["LeftPupilDiameter_scaled"]].max())

    r_pupil_d_scaled.append(mean_value_zombie_pupil_size["RightPupilDiameter_scaled"].values.reshape(1,-1)[0].sum())

    print("AVG pupil size right: ", mean_value_zombie_pupil_size[["RightPupilDiameter_scaled"]].mean())
    avg_r_pupil_d.append(mean_value_zombie_pupil_size[["RightPupilDiameter_scaled"]].mean())
    print("MAX pupil size right: ", mean_value_zombie_pupil_size[["RightPupilDiameter_scaled"]].max())
    max_r_pupil_d.append(mean_value_zombie_pupil_size[["RightPupilDiameter_scaled"]].max())
    
    l_c_a_pupil_d_scaled.append(mean_value_zombie_pupil_size["CognitiveActivityLeftPupilDiamter_scaled"].values.reshape(1,-1)[0])
    #l_c_a_pupil_d_scaled.append([mean_value_zombie_pupil_size["CognitiveActivityLeftPupilDiamter_scaled"].values.reshape(1,-1)[0].sum()])

    print("CA-AVG pupil size left: ", mean_value_zombie_pupil_size[["CognitiveActivityLeftPupilDiamter_scaled"]].mean())
    avg_l_c_a_pupil_d.append(mean_value_zombie_pupil_size[["CognitiveActivityLeftPupilDiamter_scaled"]].mean())
    print("CA-MAX pupil size left: ", mean_value_zombie_pupil_size[["CognitiveActivityLeftPupilDiamter_scaled"]].max())
    max_l_c_a_pupil_d.append(mean_value_zombie_pupil_size[["CognitiveActivityLeftPupilDiamter_scaled"]].max())

    print("CA-AVG pupil size right: ",mean_value_zombie_pupil_size[["CognitiveActivityRightPupilDiamter_scaled"]].mean())
    avg_r_c_a_pupil_d.append(mean_value_zombie_pupil_size[["CognitiveActivityRightPupilDiamter_scaled"]].mean())
    print("CA-MAX pupil size right: ",mean_value_zombie_pupil_size[["CognitiveActivityRightPupilDiamter_scaled"]].max())
    max_r_c_a_pupil_d.append(mean_value_zombie_pupil_size[["CognitiveActivityRightPupilDiamter_scaled"]].max())

plt.figure(figsize=(15,10))
plt.scatter(range(16),avg_l_pupil_d, c="b", label="avg lefet pupil diameter")
plt.scatter(range(16),max_l_pupil_d, c="g", label="max lefet pupil diameter")
plt.scatter(range(16),avg_r_pupil_d, c="r", label="avg right pupil diameter")
plt.scatter(range(16),avg_r_pupil_d, c="c", label="max right pupil diameter")
plt.scatter(range(16),avg_l_c_a_pupil_d, c="m", label="avg lefet cognitive activity")
plt.scatter(range(16),max_l_c_a_pupil_d, c="burlywood", label="max lefet cognitive activity")
plt.scatter(range(16),avg_r_c_a_pupil_d, c="k", label="avg right cognitive activity")
plt.scatter(range(16),max_r_c_a_pupil_d, c="lightcoral", label="max right cognitive activity")
plt.title("CL EYE Sensor with pupilometry ", fontsize=16)
plt.legend()
#plt.show()
plt.close()

print(l_pupil_d_scaled)

data_1 = np.random.normal(100, 10, 200)
data_2 = np.random.normal(90, 20, 200)
data_3 = np.random.normal(80, 30, 200)
data_4 = np.random.normal(70, 40, 200)
data = [data_1, data_2, data_3, data_4]
#print(data)
plt.figure(figsize=(15,10))
#plt.boxplot(l_pupil_d_scaled)
#plt.boxplot(r_pupil_d_scaled)
plt.boxplot(l_c_a_pupil_d_scaled)
plt.show()
plt.close()
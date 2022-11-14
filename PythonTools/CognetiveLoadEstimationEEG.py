from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.matlib
import sys
import os
from os.path import exists

print("Create output directory")
# --- create dir
mode = 0o666
if not exists("./output"):
    os.mkdir("./output", mode)
path = "./output/CogenetiveLoadEstimation"
if not exists(path):
    os.mkdir(path, mode)
path_eeg = "{}/EEG".format(path)
if not exists(path_eeg):
    os.mkdir(path_eeg, mode)

# read data (baseline and task)
input_stage_0 = pd.read_csv("All_Participents_Stage0_DataFrame.csv", sep=";", decimal=',')
input_stage_1 = pd.read_csv("All_Participents_Stage1_DataFrame.csv", sep=";", decimal=',')

# EEG bandpower with alpha wave for each sensor on headset:  Event related synchronization/event related desynchronization, 
# congnitive load index = ((baseline interval band power - test interval band power) / baseline interval band power) * 100

input_stage_0["theta"] = (input_stage_0["AF3.theta"] + input_stage_0["F7.theta"]  + input_stage_0["F3.theta"] + 
                          input_stage_0["FC5.theta"] + input_stage_0["T7.theta"]  + input_stage_0["P7.theta"] + 
                          input_stage_0["O1.theta"]  + input_stage_0["O2.theta"]  + input_stage_0["P8.theta"] + 
                          input_stage_0["T8.theta"]  + input_stage_0["AF4.theta"] + input_stage_0["F8.theta"] + 
                          input_stage_0["F4.theta"]  + input_stage_0["FC6.theta"])
input_stage_0["alpha"] = (input_stage_0["AF3.alpha"] + input_stage_0["F7.alpha"]  + input_stage_0["F3.alpha"] + 
                          input_stage_0["FC5.alpha"] + input_stage_0["T7.alpha"]  + input_stage_0["P7.alpha"] + 
                          input_stage_0["O1.alpha"]  + input_stage_0["O2.alpha"]  + input_stage_0["P8.alpha"] + 
                          input_stage_0["T8.alpha"]  + input_stage_0["AF4.alpha"] + input_stage_0["F8.alpha"] + 
                          input_stage_0["F4.alpha"]  + input_stage_0["FC6.alpha"]) 

#input_stage_0["theta_scaled"] = StandardScaler().fit_transform(input_stage_0[["theta"]])
#input_stage_0["alpha_scaled"] = StandardScaler().fit_transform(input_stage_0[["alpha"]])
input_stage_0["theta_scaled"] = input_stage_0[["theta"]]
input_stage_0["alpha_scaled"] = input_stage_0[["alpha"]]


input_stage_1["theta"] = (input_stage_1["AF3.theta"] + input_stage_1["F7.theta"]  + input_stage_1["F3.theta"] + 
                          input_stage_1["FC5.theta"] + input_stage_1["T7.theta"]  + input_stage_1["P7.theta"] + 
                          input_stage_1["O1.theta"]  + input_stage_1["O2.theta"]  + input_stage_1["P8.theta"] + 
                          input_stage_1["T8.theta"]  + input_stage_1["AF4.theta"] + input_stage_1["F8.theta"] + 
                          input_stage_1["F4.theta"]  + input_stage_1["FC6.theta"])
input_stage_1["alpha"] = (input_stage_1["AF3.alpha"] + input_stage_1["F7.alpha"]  + input_stage_1["F3.alpha"] + 
                          input_stage_1["FC5.alpha"] + input_stage_1["T7.alpha"]  + input_stage_1["P7.alpha"] + 
                          input_stage_1["O1.alpha"]  + input_stage_1["O2.alpha"]  + input_stage_1["P8.alpha"] + 
                          input_stage_1["T8.alpha"]  + input_stage_1["AF4.alpha"] + input_stage_1["F8.alpha"] + 
                          input_stage_1["F4.alpha"]  + input_stage_1["FC6.alpha"])

#input_stage_1["theta_scaled"] = StandardScaler().fit_transform(input_stage_1[["theta"]])
#input_stage_1["alpha_scaled"] = StandardScaler().fit_transform(input_stage_1[["alpha"]])
input_stage_1["theta_scaled"] = input_stage_1[["theta"]]
input_stage_1["alpha_scaled"] = input_stage_1[["alpha"]]

cognitive_load_index_theta_waves = []
cognitive_load_index_alpha_waves = []

for id in range(16):
    print("--------------------------------- id: ", id)
    model_input_stage_1 = input_stage_1.loc[(input_stage_1["ActivatedModelIndex"] == id)]
    cognitive_load_index_theta = ((input_stage_0["theta_scaled"].mean() - model_input_stage_1["theta_scaled"].mean()) / input_stage_0["theta_scaled"].mean()) * 100
    cognitive_load_index_theta_waves.append([cognitive_load_index_theta])

    cognitive_load_index_alpha = ((input_stage_0["alpha_scaled"].mean() - model_input_stage_1["alpha_scaled"].mean()) / input_stage_0["alpha_scaled"].mean()) * 100
    cognitive_load_index_alpha_waves.append([cognitive_load_index_alpha])

file_name = '{}/EEG_alpha_theta_boxplot.png'.format(path_eeg)
plt.figure(figsize=(15,10))
plt.title("EEG and Cognitive Load Effect", fontsize=18)
width = 0.3
plt.boxplot(input_stage_0["theta_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="green"), positions=[0], widths=width)
plt.boxplot(input_stage_0["alpha_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[1], widths=width)
plt.boxplot(input_stage_1["theta_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[2], widths=width)
plt.boxplot(input_stage_1["alpha_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="red"), positions=[3], widths=width)
labels_list = ['theta (stage 0)', 'alpha (stage 0)', 'theta (stage 1)', 'alpha (stage 1)']
plt.xticks(range(4), labels=labels_list, rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Theta and alpha values", fontsize=16)
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
plt.tight_layout()
plt.savefig(file_name)
plt.close()

file_name = '{}/EEG_alpha_theta_scatterplot.png'.format(path_eeg)
plt.figure(figsize=(15,10))
plt.title("EEG and Cognitive Load Effect", fontsize=18)
plt.plot(range(16), cognitive_load_index_theta_waves, color="blue", alpha=0.5, marker="s", zorder=2)
plt.scatter(range(16), cognitive_load_index_theta_waves, color="blue", label="theta", alpha=0.5, marker="s", zorder=1)
plt.plot(range(16), cognitive_load_index_alpha_waves, alpha=0.5, marker="o", zorder=2)
plt.scatter(range(16), cognitive_load_index_alpha_waves, label="alpha", alpha=0.5, marker="o", zorder=1)
labels_list = ['Gettie', 'Eyebot','Turret','JRRobo','Lloid','Atlas','Ribbot','Katie','Alice','Freddy','MedicBot','link','Duchess','Zombie','MixamoGirl', 'Remy']
plt.xticks(range(16), labels=labels_list, rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Cognitive Load Index of theta and alpha values", fontsize=16)
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
plt.tight_layout()
plt.savefig(file_name)
plt.close()

# ------------------------------------------------------------------------------ parietal lob and frontal lob
# read data (baseline and task)
input_stage_0 = pd.read_csv("All_Participents_Stage0_DataFrame.csv", sep=";", decimal=',')
input_stage_1 = pd.read_csv("All_Participents_Stage1_DataFrame.csv", sep=";", decimal=',')

# EEG bandpower with alpha wave for each sensor on headset:  Event related synchronization/event related desynchronization, 
# congnitive load index = ((baseline interval band power - test interval band power) / baseline interval band power) * 100
# alpha at the parietal areas and theta at the frontal
# T and O to delete
# Neuroscience and CSR: Using EEG for Assessing the Effectiveness of Branded Videos Related to Environmental Issues
# https://emotiv.gitbook.io/epoc-x-user-manual/introduction/introduction-to-epoc-x/coverage
# REPORT OF THE COMMITTEE ON METHODS OF CLINICAL EXAMINATION IN ELECTROENCEPHALOGRAPHY
input_stage_0["theta"] = (input_stage_0["AF3.theta"] + input_stage_0["F7.theta"] + input_stage_0["F3.theta"] + 
                          input_stage_0["FC5.theta"] + 
                          input_stage_0["AF4.theta"] + input_stage_0["F8.theta"] + 
                          input_stage_0["F4.theta"]  + input_stage_0["FC6.theta"])

input_stage_0["alpha"] = (input_stage_0["P7.alpha"] + input_stage_0["P8.alpha"]) 

#input_stage_0["theta_scaled"] = StandardScaler().fit_transform(input_stage_0[["theta"]])
#input_stage_0["alpha_scaled"] = StandardScaler().fit_transform(input_stage_0[["alpha"]])
input_stage_0["theta_scaled"] = input_stage_0[["theta"]]
input_stage_0["alpha_scaled"] = input_stage_0[["alpha"]]


input_stage_1["theta"] = (input_stage_1["AF3.theta"] + input_stage_1["F7.theta"]  + input_stage_1["F3.theta"] + 
                          input_stage_1["FC5.theta"] + 
                          input_stage_1["AF4.theta"] + input_stage_1["F8.theta"] + 
                          input_stage_1["F4.theta"]  + input_stage_1["FC6.theta"])

input_stage_1["alpha"] = (input_stage_1["P7.alpha"] + input_stage_1["P8.alpha"])

#input_stage_1["theta_scaled"] = StandardScaler().fit_transform(input_stage_1[["theta"]])
#input_stage_1["alpha_scaled"] = StandardScaler().fit_transform(input_stage_1[["alpha"]])
input_stage_1["theta_scaled"] = input_stage_1[["theta"]]
input_stage_1["alpha_scaled"] = input_stage_1[["alpha"]]


cognitive_load_index_theta_waves = []
cognitive_load_index_alpha_waves = []
cognitive_load_index_sum = []
for id in range(16):
    print("--------------------------------- id: ", id)
    model_input_stage_1 = input_stage_1.loc[(input_stage_1["ActivatedModelIndex"] == id)]
    cognitive_load_index_theta = ((input_stage_0["theta_scaled"].mean() - model_input_stage_1["theta_scaled"].mean()) / input_stage_0["theta_scaled"].mean()) * 100
    cognitive_load_index_theta_waves.append([cognitive_load_index_theta])

    cognitive_load_index_alpha = ((input_stage_0["alpha_scaled"].mean() - model_input_stage_1["alpha_scaled"].mean()) / input_stage_0["alpha_scaled"].mean()) * 100
    cognitive_load_index_alpha_waves.append([cognitive_load_index_alpha])
    #cognitive_load_index_sum.append([cognitive_load_index_theta + cognitive_load_index_alpha])

file_name = '{}/EEG_alpha_theta_parietal_frontal_boxplot.png'.format(path_eeg)
plt.figure(figsize=(15,10))
plt.title("EEG parietal and frontal lob for Cognitive Load Effect estimation", fontsize=18)
width = 0.3
plt.boxplot(input_stage_0["theta_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="green"), positions=[0], widths=width)
plt.boxplot(input_stage_0["alpha_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[1], widths=width)
plt.boxplot(input_stage_1["theta_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[2], widths=width)
plt.boxplot(input_stage_1["alpha_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="red"), positions=[3], widths=width)
labels_list = ['theta (stage 0)', 'alpha (stage 0)', 'theta (stage 1)', 'alpha (stage 1)']
plt.xticks(range(4), labels=labels_list, rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Theta and alpha values", fontsize=16)
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
plt.tight_layout()
plt.savefig(file_name)
plt.close()

file_name = '{}/EEG_alpha_theta_parietal_frontal_scatterplot.png'.format(path_eeg)
plt.figure(figsize=(15,10))
plt.title("EEG parietal and frontal lob for Cognitive Load Effect estimation", fontsize=18)
plt.plot(range(16), cognitive_load_index_theta_waves, color="blue", alpha=0.5, marker="s", zorder=2)
plt.scatter(range(16), cognitive_load_index_theta_waves, color="blue", label="theta", alpha=0.5, marker="s", zorder=1)
plt.plot(range(16), cognitive_load_index_alpha_waves, alpha=0.5, marker="o", zorder=2)
plt.scatter(range(16), cognitive_load_index_alpha_waves, label="alpha", alpha=0.5, marker="o", zorder=1)
labels_list = ['Gettie', 'Eyebot','Turret','JRRobo','Lloid','Atlas','Ribbot','Katie','Alice','Freddy','MedicBot','link','Duchess','Zombie','MixamoGirl', 'Remy']
plt.xticks(range(16), labels=labels_list, rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Cognitive Load Index of theta and alpha values", fontsize=16)
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
#plt.ticklabel_format(useOffset=False)
plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
plt.tight_layout()
plt.savefig(file_name)
plt.close()
    

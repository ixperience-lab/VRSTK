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
path = "./output/BehaviorValidityScoreAbstraction"
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

input_stage_0["betaL"] = (input_stage_0["AF3.betaL"] + input_stage_0["F7.betaL"]  + input_stage_0["F3.betaL"] + 
                          input_stage_0["FC5.betaL"] + input_stage_0["T7.betaL"]  + input_stage_0["P7.betaL"] + 
                          input_stage_0["O1.betaL"]  + input_stage_0["O2.betaL"]  + input_stage_0["P8.betaL"] + 
                          input_stage_0["T8.betaL"]  + input_stage_0["AF4.betaL"] + input_stage_0["F8.betaL"] + 
                          input_stage_0["F4.betaL"]  + input_stage_0["FC6.betaL"])
input_stage_0["betaH"] = (input_stage_0["AF3.betaH"] + input_stage_0["F7.betaH"]  + input_stage_0["F3.betaH"] + 
                          input_stage_0["FC5.betaH"] + input_stage_0["T7.betaH"]  + input_stage_0["P7.betaH"] + 
                          input_stage_0["O1.betaH"]  + input_stage_0["O2.betaH"]  + input_stage_0["P8.betaH"] + 
                          input_stage_0["T8.betaH"]  + input_stage_0["AF4.betaH"] + input_stage_0["F8.betaH"] + 
                          input_stage_0["F4.betaH"]  + input_stage_0["FC6.betaH"]) 
input_stage_0["gamma"] = (input_stage_0["AF3.gamma"] + input_stage_0["F7.gamma"]  + input_stage_0["F3.gamma"] + 
                          input_stage_0["FC5.gamma"] + input_stage_0["T7.gamma"]  + input_stage_0["P7.gamma"] + 
                          input_stage_0["O1.gamma"]  + input_stage_0["O2.gamma"]  + input_stage_0["P8.gamma"] + 
                          input_stage_0["T8.gamma"]  + input_stage_0["AF4.gamma"] + input_stage_0["F8.gamma"] + 
                          input_stage_0["F4.gamma"]  + input_stage_0["FC6.gamma"]) 

input_stage_0["betaL_scaled"] = StandardScaler().fit_transform(input_stage_0[["betaL"]])
input_stage_0["betaH_scaled"] = StandardScaler().fit_transform(input_stage_0[["betaH"]])
input_stage_0["gamma_scaled"] = StandardScaler().fit_transform(input_stage_0[["gamma"]])

input_stage_1["betaL"] = (input_stage_1["AF3.betaL"] + input_stage_1["F7.betaL"]  + input_stage_1["F3.betaL"] + 
                          input_stage_1["FC5.betaL"] + input_stage_1["T7.betaL"]  + input_stage_1["P7.betaL"] + 
                          input_stage_1["O1.betaL"]  + input_stage_1["O2.betaL"]  + input_stage_1["P8.betaL"] + 
                          input_stage_1["T8.betaL"]  + input_stage_1["AF4.betaL"] + input_stage_1["F8.betaL"] + 
                          input_stage_1["F4.betaL"]  + input_stage_1["FC6.betaL"])
input_stage_1["betaH"] = (input_stage_1["AF3.betaH"] + input_stage_1["F7.betaH"]  + input_stage_1["F3.betaH"] + 
                          input_stage_1["FC5.betaH"] + input_stage_1["T7.betaH"]  + input_stage_1["P7.betaH"] + 
                          input_stage_1["O1.betaH"]  + input_stage_1["O2.betaH"]  + input_stage_1["P8.betaH"] + 
                          input_stage_1["T8.betaH"]  + input_stage_1["AF4.betaH"] + input_stage_1["F8.betaH"] + 
                          input_stage_1["F4.betaH"]  + input_stage_1["FC6.betaH"]) 
input_stage_1["gamma"] = (input_stage_1["AF3.gamma"] + input_stage_1["F7.gamma"]  + input_stage_1["F3.gamma"] + 
                          input_stage_1["FC5.gamma"] + input_stage_1["T7.gamma"]  + input_stage_1["P7.gamma"] + 
                          input_stage_1["O1.gamma"]  + input_stage_1["O2.gamma"]  + input_stage_1["P8.gamma"] + 
                          input_stage_1["T8.gamma"]  + input_stage_1["AF4.gamma"] + input_stage_1["F8.gamma"] + 
                          input_stage_1["F4.gamma"]  + input_stage_1["FC6.gamma"]) 

input_stage_1["betaL_scaled"] = StandardScaler().fit_transform(input_stage_1[["betaL"]])
input_stage_1["betaH_scaled"] = StandardScaler().fit_transform(input_stage_1[["betaH"]])
input_stage_1["gamma_scaled"] = StandardScaler().fit_transform(input_stage_1[["gamma"]])

cluster_conscientious = [ 1, 2, 3, 4, 5, 6, 7, 10 ]
cluster_non_conscientious = [ 13, 14, 15, 16, 17, 18, 19, 20, 31, 34 ]
cluster_filtered_non_conscientious = [ 15, 16, 17, 18, 19, 20, 31, 34 ]

base_line_eeg_conscientious_stage_0 = input_stage_0.loc[ (input_stage_0['pId'].isin(cluster_conscientious))]
eeg_conscientious_stage_1 = input_stage_1.loc[ (input_stage_1['pId'].isin(cluster_conscientious)) ]

conscientious_event_related_index_betaL = []
conscientious_event_related_index_betaH = []
conscientious_event_related_index_gamma = []
for id in cluster_conscientious:
    base_line_conscientious_stage_0 = input_stage_0.loc[(input_stage_0["pId"] == id)]
    conscientious_stage_1 = input_stage_1.loc[(input_stage_1["pId"] == id)]
    conscientious_event_related_index_betaL.append([((base_line_conscientious_stage_0["betaL_scaled"].mean() - conscientious_stage_1["betaL_scaled"].mean()) / base_line_conscientious_stage_0["betaL_scaled"].mean()) * 100])
    conscientious_event_related_index_betaH.append([((base_line_conscientious_stage_0["betaH_scaled"].mean() - conscientious_stage_1["betaH_scaled"].mean()) / base_line_conscientious_stage_0["betaH_scaled"].mean()) * 100])
    conscientious_event_related_index_gamma.append([((base_line_conscientious_stage_0["gamma_scaled"].mean() - conscientious_stage_1["gamma_scaled"].mean()) / base_line_conscientious_stage_0["gamma_scaled"].mean()) * 100])

    # conscientious_event_related_index_betaL.append([conscientious_stage_1["betaL_scaled"].mean()])
    # conscientious_event_related_index_betaH.append([conscientious_stage_1["betaH_scaled"].mean()])
    # conscientious_event_related_index_gamma.append([conscientious_stage_1["gamma_scaled"].mean()])

base_line_eeg_non_conscientious_stage_0 = input_stage_0.loc[ (input_stage_0['pId'].isin(cluster_non_conscientious))]
eeg_non_conscientious_stage_1 = input_stage_1.loc[ (input_stage_1['pId'].isin(cluster_non_conscientious)) ]

non_conscientious_event_related_index_betaL = []
non_conscientious_event_related_index_betaH = []
non_conscientious_event_related_index_gamma = []
for id in cluster_filtered_non_conscientious:
    base_line_non_conscientious_stage_0 = input_stage_0.loc[(input_stage_0["pId"] == id)]
    non_conscientious_stage_1 = input_stage_1.loc[(input_stage_1["pId"] == id)]
    non_conscientious_event_related_index_betaL.append([((base_line_non_conscientious_stage_0["betaL_scaled"].mean() - non_conscientious_stage_1["betaL_scaled"].mean()) / base_line_non_conscientious_stage_0["betaL_scaled"].mean()) * 100])
    non_conscientious_event_related_index_betaH.append([((base_line_non_conscientious_stage_0["betaH_scaled"].mean() - non_conscientious_stage_1["betaH_scaled"].mean()) / base_line_non_conscientious_stage_0["betaH_scaled"].mean()) * 100])
    non_conscientious_event_related_index_gamma.append([((base_line_non_conscientious_stage_0["gamma_scaled"].mean() - non_conscientious_stage_1["gamma_scaled"].mean()) / base_line_non_conscientious_stage_0["gamma_scaled"].mean()) * 100])

    # non_conscientious_event_related_index_betaL.append([non_conscientious_stage_1["betaL_scaled"].mean()])
    # non_conscientious_event_related_index_betaH.append([non_conscientious_stage_1["betaH_scaled"].mean()])
    # non_conscientious_event_related_index_gamma.append([non_conscientious_stage_1["gamma_scaled"].mean()])

file_name = '{}/EEG_beta_gamma_boxplot.png'.format(path_eeg)
plt.figure(figsize=(15,10))
plt.title("EEG Behavior Validity Score Abstraction")
width = 0.2
plt.boxplot(base_line_eeg_conscientious_stage_0["betaL_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="green"), positions=[-0.75], widths=width)
plt.boxplot(base_line_eeg_conscientious_stage_0["betaH_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="green"), positions=[-0.5], widths=width)
plt.boxplot(base_line_eeg_conscientious_stage_0["gamma_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="green"), positions=[-0.25], widths=width)
plt.boxplot(eeg_conscientious_stage_1["betaL_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[0.25], widths=width)
plt.boxplot(eeg_conscientious_stage_1["betaH_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[0.5], widths=width)
plt.boxplot(eeg_conscientious_stage_1["gamma_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[0.75], widths=width)

plt.boxplot(base_line_eeg_non_conscientious_stage_0["betaL_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="red"), positions=[1.5], widths=width)
plt.boxplot(base_line_eeg_non_conscientious_stage_0["betaH_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="red"), positions=[1.75], widths=width)
plt.boxplot(base_line_eeg_non_conscientious_stage_0["gamma_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="red"), positions=[2], widths=width)
plt.boxplot(eeg_non_conscientious_stage_1["betaL_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[2.25], widths=width)
plt.boxplot(eeg_non_conscientious_stage_1["betaH_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[2.5], widths=width)
plt.boxplot(eeg_non_conscientious_stage_1["gamma_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[2.75], widths=width)
labels_list = ['0','1']
plt.xticks(range(2), labels=labels_list)
#plt.legend()
plt.savefig(file_name)
plt.close()

file_name = '{}/EEG_beta_gamma_scatterplot.png'.format(path_eeg)
plt.figure(figsize=(15,10))
plt.title("EEG Behavior Validity Score Abstraction")
plt.scatter(range(len(cluster_conscientious)), conscientious_event_related_index_betaL, color="darkblue", label="conscientious event related index betaL", alpha=0.5, marker="s", zorder=1)
plt.scatter(range(len(cluster_conscientious)), conscientious_event_related_index_betaH, color="blue", label="conscientious event related index betaH", alpha=0.5, marker="s", zorder=1)
plt.scatter(range(len(cluster_conscientious)), conscientious_event_related_index_gamma, color="green", label="conscientious event related index gamma", alpha=0.5, marker="s", zorder=1)

plt.scatter(range(len(cluster_filtered_non_conscientious)), non_conscientious_event_related_index_betaL, color="red", label="non-conscientious event related index betaL", alpha=0.5, marker="s", zorder=1)
plt.scatter(range(len(cluster_filtered_non_conscientious)), non_conscientious_event_related_index_betaH, color="orange", label="non-conscientious event related index betaH", alpha=0.5, marker="s", zorder=1)
plt.scatter(range(len(cluster_filtered_non_conscientious)), non_conscientious_event_related_index_gamma, color="yellow", label="non-conscientious event related index gamma", alpha=0.5, marker="s", zorder=1)
labels_list = ['1','2','3','4','5','6','7','8']
plt.xticks(range(len(cluster_conscientious)), labels=labels_list)
plt.legend()
plt.savefig(file_name)
plt.close()


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.matlib
import sys
import os
from os.path import exists

def plot_heart_rate_variability(base_line_data, base_line_color, base_line_label, task_data, task_color, task_label,
                                title, x_label, y_label,file_name, show=False, save=False):
    plt.figure(figsize=(15,10))
    plt.title(title, fontsize=18)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(all_particepants_ids, labels=[ "1", "2", "3", "4", "5", "6", "7", "10", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "31", "34" ], fontsize=14)
    plt.plot(all_particepants_ids, base_line_data,    color=base_line_color, alpha=0.5, zorder=2)
    plt.scatter(all_particepants_ids, base_line_data, color=base_line_color, label=base_line_label, alpha=0.5, marker="s", zorder=1)
    plt.plot(all_particepants_ids, task_data,         color=task_color, alpha=0.5, zorder=2)
    plt.scatter(all_particepants_ids, task_data,      color=task_color, label=task_label, alpha=0.5, marker="s", zorder=1)
    plt.grid(which="major", alpha=0.6)
    plt.grid(which="minor", alpha=0.6)
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(file_name)
    if show:
        plt.show()
    plt.close()
    plt.close()

print("Create output directory")
# --- create dir
mode = 0o666
if not exists("./output"):
    os.mkdir("./output", mode)
path = "./output/CogenetiveLoadEstimation"
if not exists(path):
    os.mkdir(path, mode)
path_hrv = "{}/HeartRateVariability".format(path)
if not exists(path_hrv):
    os.mkdir(path_hrv, mode)

# read data (baseline and task)
input_stage_0 = pd.read_csv("All_Participents_Stage0_DataFrame.csv", sep=";", decimal=',')
input_stage_1 = pd.read_csv("All_Participents_Stage1_DataFrame.csv", sep=";", decimal=',')

# hrv LF/HF ration: This ratio is already known to illustrate the cognitive activity. As the mental effort increases, this ratio will decrease.
# artical: Measuring Cognitive Load: Heart-rate Variability and Pupillometry Assessment

base_line_lf_hf_ratios = []
base_line_sd1_sd2_ratios = []
lf_hf_ratios = []
sd1_sd2_ratios = []

# all particepants hrv with all 3D models
# -----------------------------------------------------------------------------------------------------------------------------
all_particepants_ids = [ 1, 2, 3, 4, 5, 6, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 34 ]
for i in all_particepants_ids:
    pId = i
    # stage 0
    base_line_stage_0 = input_stage_0.loc[(input_stage_0["pId"] == pId)]
    base_line_lf_hf_ratios.append([base_line_stage_0["LFHFRatio"].mean()])
    base_line_sd1_sd2_ratios.append([base_line_stage_0["SD1SD2Ratio"].mean()])
    # stage 1
    base_line_stage_1 = input_stage_1.loc[(input_stage_1["pId"] == pId)]
    lf_hf_ratios.append([base_line_stage_1["LFHFRatio"].mean()])
    sd1_sd2_ratios.append([base_line_stage_1["SD1SD2Ratio"].mean()])

file_name = '{}/HeartRateVariability_lf_hf_ratio_scatterplot.png'.format(path_hrv)
plot_heart_rate_variability(base_line_data=base_line_lf_hf_ratios, base_line_color="orange", base_line_label="base line", 
                            task_data=lf_hf_ratios, task_color="red", task_label="task",
                            title="Hearte rate variability and cognitive load effect", 
                            x_label="Particepant-ID", y_label="LF\\HF ratio", file_name=file_name, save=True)

file_name = '{}/HeartRateVariability_sd1_sd2_ratio_scatterplot.png'.format(path_hrv)
plot_heart_rate_variability(base_line_data=base_line_sd1_sd2_ratios, base_line_color="blue", base_line_label="base line", 
                            task_data=sd1_sd2_ratios, task_color="green", task_label="task",
                            title="Hearte rate variability and cognitive load effect",
                            x_label="Particepant-ID", y_label="SD1\\SD2 ratio", file_name=file_name, save=True)

path_eda_all = "{}/SkinConductanceAll".format(path)
if not exists(path_eda_all):
    os.mkdir(path_eda_all, mode)

path_eda_seperate_cluster = "{}/SkinConductanceSeperateCluster".format(path)
if not exists(path_eda_seperate_cluster):
    os.mkdir(path_eda_seperate_cluster, mode)

path_eda_all_seperate = "{}/SkinConductanceAllSeperate".format(path)
if not exists(path_eda_all_seperate):
    os.mkdir(path_eda_all_seperate, mode)



# -----------------------------------------------------------------
# all particepants skin conductance with all 3D models
for i in [ 1, 2, 3, 4, 5, 6, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 31, 34, 21, 22, 23, 24, 25, 26, 27, 28, 29 ]:
    pId = i
    # stage 0
    base_line_filtered_eda_stage_0 = input_stage_0.loc[(input_stage_0["pId"] == pId)]
    base_line_filtered_eda_stage_0["FilteredValueInMicroSiemens_scaled"] = StandardScaler().fit_transform(base_line_filtered_eda_stage_0[["FilteredValueInMicroSiemens"]])
    # stage 1
    filtered_eda_stage_1 = input_stage_1.loc[(input_stage_1["pId"] == pId)]
    filtered_eda_stage_1["FilteredValueInMicroSiemens_scaled"] = StandardScaler().fit_transform(filtered_eda_stage_1[["FilteredValueInMicroSiemens"]])
    
    base_line_means = []
    task_values = []
    for id in range(16):
        filtered_eda_to_model_id = filtered_eda_stage_1.loc[(filtered_eda_stage_1["ActivatedModelIndex"] == id)]
        task_values.append([filtered_eda_to_model_id["FilteredValueInMicroSiemens_scaled"].mean()])
        base_line_means.append([base_line_filtered_eda_stage_0["FilteredValueInMicroSiemens_scaled"].mean()]) 
    
    file_name = '{}/Cognitive_Activity_skin_conductance_id_{}_boxplot.png'.format(path_eda_all_seperate, pId)
    plt.figure(figsize=(15,10))
    plt.title("SkinConductance and cognitive load effect", fontsize=18)
    width = 0.3
    plt.boxplot(base_line_filtered_eda_stage_0["FilteredValueInMicroSiemens_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="red"), positions=[0], widths=width)
    plt.boxplot(filtered_eda_stage_1["FilteredValueInMicroSiemens_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[1], widths=width)
    plt.tight_layout()
    #plt.legend()
    plt.savefig(file_name)
    plt.close()

    file_name = '{}/Cognitive_Activity_skin_conductance_id_{}_scatterplot.png'.format(path_eda_all_seperate, pId)
    plt.figure(figsize=(15,10))
    plt.title("SkinConductance and cognitive load effect", fontsize=18)
    plt.plot(range(16), base_line_means, color="blue", alpha=0.5, marker="o", zorder=2)
    plt.scatter(range(16), base_line_means, color="blue", label="base line", alpha=0.5, marker="o", zorder=1)
    plt.plot(range(16), task_values, alpha=0.5, marker="o", zorder=2)
    plt.scatter(range(16), task_values, label="task", alpha=0.5, marker="o", zorder=1)
    labels_list = ['Gettie', 'Eyebot','Turret','JRRobo','Lloid','Atlas','Ribbot','Katie','Alice','Freddy','MedicBot','link','Duchess','Zombie','MixamoGirl', 'Remy']
    plt.xticks(range(16), labels=labels_list, rotation=45, ha='right', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Filtered skin conductance mean values (micro Siemens)", fontsize=16)
    plt.grid(which="major", alpha=0.6)
    plt.grid(which="minor", alpha=0.6)
    plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

# all particepants skin conduction with all 3D models
# -----------------------------------------------------------------------------------------------------------------------------
base_line_filtered_eda_stage_0 = input_stage_0
base_line_filtered_eda_stage_0["FilteredValueInMicroSiemens_scaled"] = StandardScaler().fit_transform(base_line_filtered_eda_stage_0[["FilteredValueInMicroSiemens"]])

filtered_eda_stage_1 = input_stage_1
filtered_eda_stage_1["FilteredValueInMicroSiemens_scaled"] = StandardScaler().fit_transform(filtered_eda_stage_1[["FilteredValueInMicroSiemens"]])

plt.figure(figsize=(15,10))
plt.title("Skin conductance and Cognitive Load Effect", fontsize=18)
width       = 0.3
plt.boxplot(base_line_filtered_eda_stage_0["FilteredValueInMicroSiemens_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="red"), positions=[0], widths=width)
plt.boxplot(filtered_eda_stage_1["FilteredValueInMicroSiemens_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[1], widths=width)
#plt.legend()
file_name = '{}/SkinConductance_all_boxplot.png'.format(path_eda_all)
plt.tight_layout()
plt.savefig(file_name)
plt.close()

base_line_means = []
task_values = []
for id in range(16):
    filtered_eda_to_model_id = filtered_eda_stage_1.loc[(filtered_eda_stage_1["ActivatedModelIndex"] == id)]
    task_values.append([filtered_eda_to_model_id["FilteredValueInMicroSiemens_scaled"].mean()])
    base_line_means.append([base_line_filtered_eda_stage_0["FilteredValueInMicroSiemens_scaled"].mean()]) 

plt.figure(figsize=(15,10))
plt.title("Skin conductance and Cognitive Load Effect", fontsize=18)
#plt.plot(range(16), base_line_means, color="blue", alpha=0.5, marker="o", zorder=2)
#plt.scatter(range(16), base_line_means, color="blue", label="skin conductance mean values (base line)", alpha=0.5, marker="o", zorder=1)
plt.plot(range(16), task_values, alpha=0.5, marker="o", zorder=2)
plt.scatter(range(16), task_values, alpha=0.5, marker="o", zorder=1)
labels_list = ['Gettie', 'Eyebot','Turret','JRRobo','Lloid','Atlas','Ribbot','Katie','Alice','Freddy','MedicBot','link','Duchess','Zombie','MixamoGirl', 'Remy']
plt.xticks(range(16), labels=labels_list, rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Scaled filtered skin conductance mean values (micro Siemens)", fontsize=16)
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
#plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
file_name = '{}/SkinConductance_cluster_all_scatter.png'.format(path_eda_all)
plt.tight_layout()
plt.savefig(file_name)
plt.close()

# seperate scluster 
# ------------------------------------------------------
cluster_conscientious = [ 1, 2, 3, 4, 5, 6, 7, 10 ]
cluster_non_conscientious = [ 13, 14, 15, 16, 17, 18, 19, 20, 31, 34 ]
cluster_no_specifications = [ 21, 22, 23, 24, 25, 26, 27, 28, 29 ]

base_line_filtered_eda_stage_0 = input_stage_0.loc[ (input_stage_0['pId'].isin(cluster_conscientious))]
base_line_filtered_eda_stage_0["FilteredValueInMicroSiemens_scaled"] = StandardScaler().fit_transform(base_line_filtered_eda_stage_0[["FilteredValueInMicroSiemens"]])

filtered_eda_stage_1 = input_stage_1.loc[ (input_stage_1['pId'].isin(cluster_conscientious)) ]
filtered_eda_stage_1["FilteredValueInMicroSiemens_scaled"] = StandardScaler().fit_transform(filtered_eda_stage_1[["FilteredValueInMicroSiemens"]])

base_line_means = []
task_values = []
for id in range(16):
    filtered_eda_to_model_id = filtered_eda_stage_1.loc[(filtered_eda_stage_1["ActivatedModelIndex"] == id)]
    task_values.append([filtered_eda_to_model_id["FilteredValueInMicroSiemens_scaled"].mean()])
    base_line_means.append([base_line_filtered_eda_stage_0["FilteredValueInMicroSiemens_scaled"].mean()]) 

plt.figure(figsize=(15,10))
plt.title("Skin conductance and Cognitive Load Effect (conscientious)", fontsize=18)
width       = 0.3
plt.boxplot(base_line_filtered_eda_stage_0["FilteredValueInMicroSiemens_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="red"), positions=[0], widths=width)
plt.boxplot(filtered_eda_stage_1["FilteredValueInMicroSiemens_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[1], widths=width)
#plt.legend()
file_name = '{}/SkinConductance_cluster_conscientious_boxplot.png'.format(path_eda_seperate_cluster)
plt.tight_layout()
plt.savefig(file_name)
plt.close()

plt.figure(figsize=(15,10))
plt.title("Skin conductance and Cognitive Load Effect (conscientious)", fontsize=18)
plt.plot(range(16), task_values, alpha=0.5, marker="o", zorder=2)
plt.scatter(range(16), task_values, alpha=0.5, marker="o", zorder=1)
labels_list = ['Gettie', 'Eyebot','Turret','JRRobo','Lloid','Atlas','Ribbot','Katie','Alice','Freddy','MedicBot','link','Duchess','Zombie','MixamoGirl', 'Remy']
plt.xticks(range(16), labels=labels_list, rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Scaled filtered skin conductance mean values (micro Siemens)", fontsize=16)
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
#plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
plt.tight_layout()
file_name = '{}/SkinConductance_cluster_conscientious_scatter.png'.format(path_eda_seperate_cluster)
plt.savefig(file_name)
plt.close()

# cluster non-conscientious
# ------------------------------------------------------------------------------------------------------------
base_line_filtered_eda_stage_0 = input_stage_0.loc[ (input_stage_0['pId'].isin(cluster_non_conscientious))]
base_line_filtered_eda_stage_0["FilteredValueInMicroSiemens_scaled"] = StandardScaler().fit_transform(base_line_filtered_eda_stage_0[["FilteredValueInMicroSiemens"]])

filtered_eda_stage_1 = input_stage_1.loc[ (input_stage_1['pId'].isin(cluster_non_conscientious)) ]
filtered_eda_stage_1["FilteredValueInMicroSiemens_scaled"] = StandardScaler().fit_transform(filtered_eda_stage_1[["FilteredValueInMicroSiemens"]])

base_line_means = []
task_values = []
for id in range(16):
    filtered_eda_to_model_id = filtered_eda_stage_1.loc[(filtered_eda_stage_1["ActivatedModelIndex"] == id)]
    task_values.append([filtered_eda_to_model_id["FilteredValueInMicroSiemens_scaled"].mean()])

    base_line_means.append([base_line_filtered_eda_stage_0["FilteredValueInMicroSiemens_scaled"].mean()]) 

plt.figure(figsize=(15,10))
plt.title("Skin conductance and Cognitive Load Effect (non-conscientious)", fontsize=18)
width       = 0.3
plt.boxplot(base_line_filtered_eda_stage_0["FilteredValueInMicroSiemens_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="red"), positions=[0], widths=width)
plt.boxplot(filtered_eda_stage_1["FilteredValueInMicroSiemens_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[1], widths=width)
#plt.legend()
plt.tight_layout()
file_name = '{}/SkinConductance_cluster_non_conscientious_boxplot.png'.format(path_eda_seperate_cluster)
plt.savefig(file_name)
plt.close()

plt.figure(figsize=(15,10))
plt.title("Skin conductance and Cognitive Load Effect (non-conscientious)", fontsize=18)
plt.plot(range(16), task_values, alpha=0.5, marker="o", zorder=2)
plt.scatter(range(16), task_values, alpha=0.5, marker="o", zorder=1)
labels_list = ['Gettie', 'Eyebot','Turret','JRRobo','Lloid','Atlas','Ribbot','Katie','Alice','Freddy','MedicBot','link','Duchess','Zombie','MixamoGirl', 'Remy']
plt.xticks(range(16), labels=labels_list, rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Scaled filtered skin conductance mean values (micro Siemens)", fontsize=16)
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
#plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
plt.tight_layout()
file_name = '{}/SkinConductance_cluster_non_conscientious_scatter.png'.format(path_eda_seperate_cluster)
plt.savefig(file_name)
plt.close()

# cluster no-specifications
# -------------------------------------------------------------------------------------------------------------

base_line_filtered_eda_stage_0 = input_stage_0.loc[ (input_stage_0['pId'].isin(cluster_no_specifications))]
base_line_filtered_eda_stage_0["FilteredValueInMicroSiemens_scaled"] = StandardScaler().fit_transform(base_line_filtered_eda_stage_0[["FilteredValueInMicroSiemens"]])

filtered_eda_stage_1 = input_stage_1.loc[ (input_stage_1['pId'].isin(cluster_no_specifications)) ]
filtered_eda_stage_1["FilteredValueInMicroSiemens_scaled"] = StandardScaler().fit_transform(filtered_eda_stage_1[["FilteredValueInMicroSiemens"]])

base_line_means = []
task_values = []
for id in range(16):
    filtered_eda_to_model_id = filtered_eda_stage_1.loc[(filtered_eda_stage_1["ActivatedModelIndex"] == id)]
    task_values.append([filtered_eda_to_model_id["FilteredValueInMicroSiemens_scaled"].mean()])
    base_line_means.append([base_line_filtered_eda_stage_0["FilteredValueInMicroSiemens_scaled"].mean()]) 

plt.figure(figsize=(15,10))
plt.title("Skin conductance and Cognitive Load Effect (no-specifications)", fontsize=18)
width       = 0.3
plt.boxplot(base_line_filtered_eda_stage_0["FilteredValueInMicroSiemens_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="red"), positions=[0], widths=width)
plt.boxplot(filtered_eda_stage_1["FilteredValueInMicroSiemens_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[1], widths=width)
#plt.legend()
plt.tight_layout()
file_name = '{}/SkinConductance_cluster_no_specifications_boxplot.png'.format(path_eda_seperate_cluster)
plt.savefig(file_name)
plt.close()

plt.figure(figsize=(15,10))
plt.title("Skin conductance and Cognitive Load Effectt (no-specifications)", fontsize=18)
plt.plot(range(16), task_values, alpha=0.5, marker="o", zorder=2)
plt.scatter(range(16), task_values, alpha=0.5, marker="o", zorder=1)
labels_list = ['Gettie', 'Eyebot','Turret','JRRobo','Lloid','Atlas','Ribbot','Katie','Alice','Freddy','MedicBot','link','Duchess','Zombie','MixamoGirl', 'Remy']
plt.xticks(range(16), labels=labels_list, rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Scaled filtered skin conductance mean values (micro Siemens)", fontsize=16)
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
#plt.legend(bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
plt.tight_layout()
file_name = '{}/SkinConductance_cluster_no_specifications_scatter.png'.format(path_eda_seperate_cluster)
plt.savefig(file_name)
plt.close()
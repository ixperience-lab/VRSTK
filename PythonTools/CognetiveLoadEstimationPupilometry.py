from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.matlib
import sys
import os
from os.path import exists

def plot_cognetive_activity_as_boxplot(data_l, data_r, legend_label, title, file_name, show=False, save=False):
    plt.figure(figsize=(15,10))
    plt.title(title)
    xlocations  = range(len(data_l))
    width       = 0.3
    positions_group1 = [x-(width+0.01) for x in xlocations]
    positions_group2 = xlocations
    labels_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14', '15','16']
    plt.boxplot(data_l, medianprops=dict(color="red"), positions=positions_group1, widths=width)
    plt.boxplot(data_r, medianprops=dict(color="blue"), positions=positions_group2, widths=width)
    plt.xticks(xlocations, labels=labels_list)
    if save:
        plt.savefig(file_name)
    if show:
        plt.show()
    plt.close()

def plot_cognetive_activity_as_scatterplot(data_l_mean, data_l_max, data_r_mean, data_r_max, legend_label, title, file_name, show=False, save=False):
    plt.figure(figsize=(15,10))
    plt.title(title)
    plt.scatter(range(16), data_l_mean, color="orange", label="left mean percent change pupil dialtions", alpha=0.5, marker="s")
    plt.scatter(range(16), data_l_max,  color="red",    label="left max percent change pupil dialtions", alpha=0.5, marker="s")
    plt.scatter(range(16), data_r_mean, color="blue",   label="right mean percent change pupil dialtions", alpha=0.5, marker="o")
    plt.scatter(range(16), data_r_max,  color="green",  label="right max percent change pupil dialtions", alpha=0.5, marker="o")
    plt.legend()
    if save:
        plt.savefig(file_name)
    if show:
        plt.show()
    plt.close()

print("Create output directory")
# --- create dir
mode = 0o666
if not exists("./output"):
    os.mkdir("./output", mode)
path = "./output/CogenetiveLoadEstimation"
if not exists(path):
    os.mkdir(path, mode)
path_all = "{}/PupilometryAll".format(path)
if not exists(path_all):
    os.mkdir(path_all, mode)
path_all_seperate = "{}/PupilometryAllSeperate".format(path)
if not exists(path_all_seperate):
    os.mkdir(path_all_seperate, mode)
path_seperate_cluster = "{}/PupilometrySeperateCluster".format(path)
if not exists(path_seperate_cluster):
    os.mkdir(path_seperate_cluster, mode)

# read data (baseline and task)
# input_data_copy = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrameKopie.csv", sep=";", decimal=',')			
# input_data = pd.read_csv("All_Participents_Clusterd_WaveSum_DataFrame.csv", sep=";", decimal=',')

input_stage_0 = pd.read_csv("All_Participents_Stage0_DataFrame.csv", sep=";", decimal=',')
input_stage_1 = pd.read_csv("All_Participents_Stage1_DataFrame.csv", sep=";", decimal=',')

# figure id = 13
# features: pId, LeftPupilDiameter, LeftEyeOpenness, RightPupilDiameter, RightEyeOpenness, ActivatedModelIndex, CognitiveActivityRightPupilDiamter
# mean_value_zombie_pupil_size = input_data_copy.loc[(input_data_copy["ActivatedModelIndex"] == 13) & 
#                                                    (input_data_copy["LeftEyeOpenness"] > 0.8) & (input_data_copy["RightEyeOpenness"] > 0.8)]

# -----------------------------------------------------------------
# all particepants pupilometry with all 3D models
for i in [ 1, 2, 3, 4, 5, 6, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 31, 34, 21, 22, 23, 24, 25, 26, 27, 28, 29 ]:
    pId = i
    # stage 0
    base_line_pupil_diameter_stage_0 = input_stage_0.loc[(input_stage_0["pId"] == pId) & (input_stage_0["LeftEyeOpenness"] > 0.8) & (input_stage_0["RightEyeOpenness"] > 0.8) & 
                                                         (input_stage_0["CognitiveActivityLeftPupilDiamter"] > 0) & (input_stage_0["CognitiveActivityRightPupilDiamter"] > 0)]
    base_line_mean_l_pupil_diameter = base_line_pupil_diameter_stage_0["LeftPupilDiameter"].mean()
    base_line_mean_r_pupil_diameter = base_line_pupil_diameter_stage_0["RightPupilDiameter"].mean()

    pupil_size_pId_stage_1 = input_stage_1.loc[(input_stage_1["pId"] == pId) & (input_stage_1["LeftEyeOpenness"] > 0.8) & (input_stage_1["RightEyeOpenness"] > 0.8) & 
                                                (input_stage_1["CognitiveActivityLeftPupilDiamter"] > 0) & (input_stage_1["CognitiveActivityRightPupilDiamter"] > 0)]

    pupil_size_pId_stage_1["LeftPercentChangePupilDialtion"] = (pupil_size_pId_stage_1["LeftPupilDiameter"] - base_line_mean_l_pupil_diameter) / base_line_mean_l_pupil_diameter
    pupil_size_pId_stage_1["RightPercentChangePupilDialtion"] = (pupil_size_pId_stage_1["RightPupilDiameter"] - base_line_mean_l_pupil_diameter) / base_line_mean_l_pupil_diameter
    
    left_percent_change_pupil_dialtions = []
    left_mean_percent_change_pupil_dialtions = []
    left_max_percent_change_pupil_dialtions = []
    right_percent_change_pupil_dialtions = []
    right_mean_percent_change_pupil_dialtions = []
    right_max_percent_change_pupil_dialtions = []
    
    for id in range(16):
        print("--------------------------------- id: ", id)
        figure_percent_change_pupil_dialtions = pupil_size_pId_stage_1.loc[(pupil_size_pId_stage_1["ActivatedModelIndex"] == id)]
        left_percent_change_pupil_dialtions.append(figure_percent_change_pupil_dialtions["LeftPercentChangePupilDialtion"].values.reshape(1, -1)[0])
        left_mean_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["LeftPercentChangePupilDialtion"].mean()])
        left_max_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["LeftPercentChangePupilDialtion"].max()])

        right_percent_change_pupil_dialtions.append(figure_percent_change_pupil_dialtions["RightPercentChangePupilDialtion"].values.reshape(1, -1)[0])
        right_mean_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["RightPercentChangePupilDialtion"].mean()])
        right_max_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["RightPercentChangePupilDialtion"].max()])

    file_name = '{}/Cognitive_Activity_Pupilometry_id_{}_boxplot.png'.format(path_all_seperate, pId)
    plot_cognetive_activity_as_boxplot(data_l=left_percent_change_pupil_dialtions, data_r=right_percent_change_pupil_dialtions,
                                       legend_label="", title="Cognitive Activity Pupilometry boxplot" , file_name=file_name, save=True)
    
    file_name = '{}/Cognitive_Activity_Pupilometry_id_{}_scatterplot.png'.format(path_all_seperate, pId)
    plot_cognetive_activity_as_scatterplot(data_l_mean=left_mean_percent_change_pupil_dialtions, data_l_max=left_max_percent_change_pupil_dialtions, 
                                           data_r_mean=right_mean_percent_change_pupil_dialtions, data_r_max=right_max_percent_change_pupil_dialtions, 
                                           legend_label="", title="Cognitive Activity Pupilometry scatter", file_name=file_name, save=True)
    

# -----------------------------------------------------------------
# all cluster/groups pupilomentry with all 3D models
base_line_pupil_diameter_stage_0 = input_stage_0.loc[(input_stage_0["LeftEyeOpenness"] > 0.8) & (input_stage_0["RightEyeOpenness"] > 0.8) & 
                                                     (input_stage_0["CognitiveActivityLeftPupilDiamter"] > 0) & (input_stage_0["CognitiveActivityRightPupilDiamter"] > 0)]
base_line_pupil_diameter_stage_0["LeftPupilDiameter_scaled"] = MinMaxScaler().fit_transform(base_line_pupil_diameter_stage_0[["LeftPupilDiameter"]])
base_line_pupil_diameter_stage_0["RightPupilDiameter_scaled"] = MinMaxScaler().fit_transform(base_line_pupil_diameter_stage_0[["RightPupilDiameter"]])

base_line_mean_l_pupil_diameter = base_line_pupil_diameter_stage_0["LeftPupilDiameter_scaled"].mean()
base_line_mean_r_pupil_diameter = base_line_pupil_diameter_stage_0["RightPupilDiameter_scaled"].mean()

pupil_size_pId_stage_1 = input_stage_1.loc[(input_stage_1["LeftEyeOpenness"] > 0.8) & (input_stage_1["RightEyeOpenness"] > 0.8) & 
                                            (input_stage_1["CognitiveActivityLeftPupilDiamter"] > 0) & (input_stage_1["CognitiveActivityRightPupilDiamter"] > 0)]

pupil_size_pId_stage_1["LeftPupilDiameter_scaled"] = MinMaxScaler().fit_transform(pupil_size_pId_stage_1[["LeftPupilDiameter"]])
pupil_size_pId_stage_1["RightPupilDiameter_scaled"] = MinMaxScaler().fit_transform(pupil_size_pId_stage_1[["RightPupilDiameter"]])

pupil_size_pId_stage_1["LeftPercentChangePupilDialtion"] = (pupil_size_pId_stage_1["LeftPupilDiameter_scaled"] - base_line_mean_l_pupil_diameter) / base_line_mean_l_pupil_diameter
pupil_size_pId_stage_1["RightPercentChangePupilDialtion"] = (pupil_size_pId_stage_1["RightPupilDiameter_scaled"] - base_line_mean_l_pupil_diameter) / base_line_mean_l_pupil_diameter

left_percent_change_pupil_dialtions = []
left_mean_percent_change_pupil_dialtions = []
left_max_percent_change_pupil_dialtions = []
right_percent_change_pupil_dialtions = []
right_mean_percent_change_pupil_dialtions = []
right_max_percent_change_pupil_dialtions = []
for id in range(16):
    print("--------------------------------- id: ", id)
    figure_percent_change_pupil_dialtions = pupil_size_pId_stage_1.loc[(pupil_size_pId_stage_1["ActivatedModelIndex"] == id)]
    left_percent_change_pupil_dialtions.append(figure_percent_change_pupil_dialtions["LeftPercentChangePupilDialtion"].values.reshape(1, -1)[0])
    left_mean_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["LeftPercentChangePupilDialtion"].mean()])
    left_max_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["LeftPercentChangePupilDialtion"].max()])

    right_percent_change_pupil_dialtions.append(figure_percent_change_pupil_dialtions["RightPercentChangePupilDialtion"].values.reshape(1, -1)[0])
    right_mean_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["RightPercentChangePupilDialtion"].mean()])
    right_max_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["RightPercentChangePupilDialtion"].max()])

file_name = '{}/Cognitive_Activity_Pupilometry_boxplot_all_groups.png'.format(path_all)
plot_cognetive_activity_as_boxplot(data_l=left_percent_change_pupil_dialtions, data_r=right_percent_change_pupil_dialtions,
                                   legend_label="", title="Cognitive Activity Pupilometry all groups  boxplot" , file_name=file_name, save=True)

file_name = '{}/Cognitive_Activity_Pupilometry_scatterplot_all_groups.png'.format(path_all)
plot_cognetive_activity_as_scatterplot(data_l_mean=left_mean_percent_change_pupil_dialtions, data_l_max=left_max_percent_change_pupil_dialtions, 
                                       data_r_mean=right_mean_percent_change_pupil_dialtions, data_r_max=right_max_percent_change_pupil_dialtions, 
                                       legend_label="", title="Cognitive Activity Pupilometry all groups scatter", file_name=file_name, save=True)

# -----------------------------------------------------------------
# seperate cluster/groups pupilomentry with all 3D models
cluster_conscientious = [ 1, 2, 3, 4, 5, 6, 7, 10 ]
cluster_non_conscientious = [ 13, 14, 15, 16, 17, 18, 19, 20, 31, 34 ]
cluster_no_specifications = [ 21, 22, 23, 24, 25, 26, 27, 28, 29 ]

base_line_pupil_diameter_stage_0 = input_stage_0.loc[ (input_stage_0['pId'].isin(cluster_conscientious)) & (input_stage_0["LeftEyeOpenness"] > 0.8) & (input_stage_0["RightEyeOpenness"] > 0.8) & 
                                                     (input_stage_0["CognitiveActivityLeftPupilDiamter"] > 0) & (input_stage_0["CognitiveActivityRightPupilDiamter"] > 0)]
base_line_pupil_diameter_stage_0["LeftPupilDiameter_scaled"] = MinMaxScaler().fit_transform(base_line_pupil_diameter_stage_0[["LeftPupilDiameter"]])
base_line_pupil_diameter_stage_0["RightPupilDiameter_scaled"] = MinMaxScaler().fit_transform(base_line_pupil_diameter_stage_0[["RightPupilDiameter"]])

base_line_mean_l_pupil_diameter = base_line_pupil_diameter_stage_0["LeftPupilDiameter_scaled"].mean()
base_line_mean_r_pupil_diameter = base_line_pupil_diameter_stage_0["RightPupilDiameter_scaled"].mean()

pupil_size_pId_stage_1 = input_stage_1.loc[ (input_stage_1['pId'].isin(cluster_conscientious)) & (input_stage_1["LeftEyeOpenness"] > 0.8) & (input_stage_1["RightEyeOpenness"] > 0.8) & 
                                            (input_stage_1["CognitiveActivityLeftPupilDiamter"] > 0) & (input_stage_1["CognitiveActivityRightPupilDiamter"] > 0)]

pupil_size_pId_stage_1["LeftPupilDiameter_scaled"] = MinMaxScaler().fit_transform(pupil_size_pId_stage_1[["LeftPupilDiameter"]])
pupil_size_pId_stage_1["RightPupilDiameter_scaled"] = MinMaxScaler().fit_transform(pupil_size_pId_stage_1[["RightPupilDiameter"]])

pupil_size_pId_stage_1["LeftPercentChangePupilDialtion"] = (pupil_size_pId_stage_1["LeftPupilDiameter_scaled"] - base_line_mean_l_pupil_diameter) / base_line_mean_l_pupil_diameter
pupil_size_pId_stage_1["RightPercentChangePupilDialtion"] = (pupil_size_pId_stage_1["RightPupilDiameter_scaled"] - base_line_mean_l_pupil_diameter) / base_line_mean_l_pupil_diameter

left_percent_change_pupil_dialtions = []
left_mean_percent_change_pupil_dialtions = []
left_max_percent_change_pupil_dialtions = []
right_percent_change_pupil_dialtions = []
right_mean_percent_change_pupil_dialtions = []
right_max_percent_change_pupil_dialtions = []
for id in range(16):
    print("--------------------------------- id: ", id)
    figure_percent_change_pupil_dialtions = pupil_size_pId_stage_1.loc[(pupil_size_pId_stage_1["ActivatedModelIndex"] == id)]
    left_percent_change_pupil_dialtions.append(figure_percent_change_pupil_dialtions["LeftPercentChangePupilDialtion"].values.reshape(1, -1)[0])
    left_mean_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["LeftPercentChangePupilDialtion"].mean()])
    left_max_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["LeftPercentChangePupilDialtion"].max()])

    right_percent_change_pupil_dialtions.append(figure_percent_change_pupil_dialtions["RightPercentChangePupilDialtion"].values.reshape(1, -1)[0])
    right_mean_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["RightPercentChangePupilDialtion"].mean()])
    right_max_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["RightPercentChangePupilDialtion"].max()])

file_name = '{}/Cognitive_Activity_Pupilometry_boxplot_cluster_conscientious.png'.format(path_seperate_cluster)
plot_cognetive_activity_as_boxplot(data_l=left_percent_change_pupil_dialtions, data_r=right_percent_change_pupil_dialtions,
                                   legend_label="", title="Cognitive Activity Pupilometry conscientious boxplot" , file_name=file_name, save=True)

file_name = '{}/Cognitive_Activity_Pupilometry_scatterplot_cluster_conscientious.png'.format(path_seperate_cluster)
plot_cognetive_activity_as_scatterplot(data_l_mean=left_mean_percent_change_pupil_dialtions, data_l_max=left_max_percent_change_pupil_dialtions, 
                                       data_r_mean=right_mean_percent_change_pupil_dialtions, data_r_max=right_max_percent_change_pupil_dialtions, 
                                       legend_label="", title="Cognitive Activity Pupilometry conscientious scatter", file_name=file_name, save=True)


base_line_pupil_diameter_stage_0 = input_stage_0.loc[ (input_stage_0['pId'].isin(cluster_non_conscientious)) & (input_stage_0["LeftEyeOpenness"] > 0.8) & (input_stage_0["RightEyeOpenness"] > 0.8) & 
                                                     (input_stage_0["CognitiveActivityLeftPupilDiamter"] > 0) & (input_stage_0["CognitiveActivityRightPupilDiamter"] > 0)]
base_line_pupil_diameter_stage_0["LeftPupilDiameter_scaled"] = MinMaxScaler().fit_transform(base_line_pupil_diameter_stage_0[["LeftPupilDiameter"]])
base_line_pupil_diameter_stage_0["RightPupilDiameter_scaled"] = MinMaxScaler().fit_transform(base_line_pupil_diameter_stage_0[["RightPupilDiameter"]])

base_line_mean_l_pupil_diameter = base_line_pupil_diameter_stage_0["LeftPupilDiameter_scaled"].mean()
base_line_mean_r_pupil_diameter = base_line_pupil_diameter_stage_0["RightPupilDiameter_scaled"].mean()

pupil_size_pId_stage_1 = input_stage_1.loc[ (input_stage_1['pId'].isin(cluster_non_conscientious)) & (input_stage_1["LeftEyeOpenness"] > 0.8) & (input_stage_1["RightEyeOpenness"] > 0.8) & 
                                            (input_stage_1["CognitiveActivityLeftPupilDiamter"] > 0) & (input_stage_1["CognitiveActivityRightPupilDiamter"] > 0)]

pupil_size_pId_stage_1["LeftPupilDiameter_scaled"] = MinMaxScaler().fit_transform(pupil_size_pId_stage_1[["LeftPupilDiameter"]])
pupil_size_pId_stage_1["RightPupilDiameter_scaled"] = MinMaxScaler().fit_transform(pupil_size_pId_stage_1[["RightPupilDiameter"]])

pupil_size_pId_stage_1["LeftPercentChangePupilDialtion"] = (pupil_size_pId_stage_1["LeftPupilDiameter_scaled"] - base_line_mean_l_pupil_diameter) / base_line_mean_l_pupil_diameter
pupil_size_pId_stage_1["RightPercentChangePupilDialtion"] = (pupil_size_pId_stage_1["RightPupilDiameter_scaled"] - base_line_mean_l_pupil_diameter) / base_line_mean_l_pupil_diameter

left_percent_change_pupil_dialtions = []
left_mean_percent_change_pupil_dialtions = []
left_max_percent_change_pupil_dialtions = []
right_percent_change_pupil_dialtions = []
right_mean_percent_change_pupil_dialtions = []
right_max_percent_change_pupil_dialtions = []
for id in range(16):
    print("--------------------------------- id: ", id)
    figure_percent_change_pupil_dialtions = pupil_size_pId_stage_1.loc[(pupil_size_pId_stage_1["ActivatedModelIndex"] == id)]
    left_percent_change_pupil_dialtions.append(figure_percent_change_pupil_dialtions["LeftPercentChangePupilDialtion"].values.reshape(1, -1)[0])
    left_mean_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["LeftPercentChangePupilDialtion"].mean()])
    left_max_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["LeftPercentChangePupilDialtion"].max()])

    right_percent_change_pupil_dialtions.append(figure_percent_change_pupil_dialtions["RightPercentChangePupilDialtion"].values.reshape(1, -1)[0])
    right_mean_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["RightPercentChangePupilDialtion"].mean()])
    right_max_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["RightPercentChangePupilDialtion"].max()])

file_name = '{}/Cognitive_Activity_Pupilometry_boxplot_cluster_non_conscientious.png'.format(path_seperate_cluster)
plot_cognetive_activity_as_boxplot(data_l=left_percent_change_pupil_dialtions, data_r=right_percent_change_pupil_dialtions,
                                   legend_label="", title="Cognitive Activity Pupilometry non-conscientious boxplot" , file_name=file_name, save=True)

file_name = '{}/Cognitive_Activity_Pupilometry_scatterplot_cluster_non_conscientious.png'.format(path_seperate_cluster)
plot_cognetive_activity_as_scatterplot(data_l_mean=left_mean_percent_change_pupil_dialtions, data_l_max=left_max_percent_change_pupil_dialtions, 
                                       data_r_mean=right_mean_percent_change_pupil_dialtions, data_r_max=right_max_percent_change_pupil_dialtions, 
                                       legend_label="", title="Cognitive Activity Pupilometry non-conscientious scatter", file_name=file_name, save=True)


base_line_pupil_diameter_stage_0 = input_stage_0.loc[ (input_stage_0['pId'].isin(cluster_no_specifications)) & (input_stage_0["LeftEyeOpenness"] > 0.8) & (input_stage_0["RightEyeOpenness"] > 0.8) & 
                                                     (input_stage_0["CognitiveActivityLeftPupilDiamter"] > 0) & (input_stage_0["CognitiveActivityRightPupilDiamter"] > 0)]
base_line_pupil_diameter_stage_0["LeftPupilDiameter_scaled"] = MinMaxScaler().fit_transform(base_line_pupil_diameter_stage_0[["LeftPupilDiameter"]])
base_line_pupil_diameter_stage_0["RightPupilDiameter_scaled"] = MinMaxScaler().fit_transform(base_line_pupil_diameter_stage_0[["RightPupilDiameter"]])

base_line_mean_l_pupil_diameter = base_line_pupil_diameter_stage_0["LeftPupilDiameter_scaled"].mean()
base_line_mean_r_pupil_diameter = base_line_pupil_diameter_stage_0["RightPupilDiameter_scaled"].mean()

pupil_size_pId_stage_1 = input_stage_1.loc[ (input_stage_1['pId'].isin(cluster_no_specifications)) & (input_stage_1["LeftEyeOpenness"] > 0.8) & (input_stage_1["RightEyeOpenness"] > 0.8) & 
                                            (input_stage_1["CognitiveActivityLeftPupilDiamter"] > 0) & (input_stage_1["CognitiveActivityRightPupilDiamter"] > 0)]

pupil_size_pId_stage_1["LeftPupilDiameter_scaled"] = MinMaxScaler().fit_transform(pupil_size_pId_stage_1[["LeftPupilDiameter"]])
pupil_size_pId_stage_1["RightPupilDiameter_scaled"] = MinMaxScaler().fit_transform(pupil_size_pId_stage_1[["RightPupilDiameter"]])

pupil_size_pId_stage_1["LeftPercentChangePupilDialtion"] = (pupil_size_pId_stage_1["LeftPupilDiameter_scaled"] - base_line_mean_l_pupil_diameter) / base_line_mean_l_pupil_diameter
pupil_size_pId_stage_1["RightPercentChangePupilDialtion"] = (pupil_size_pId_stage_1["RightPupilDiameter_scaled"] - base_line_mean_l_pupil_diameter) / base_line_mean_l_pupil_diameter

left_percent_change_pupil_dialtions = []
left_mean_percent_change_pupil_dialtions = []
left_max_percent_change_pupil_dialtions = []
right_percent_change_pupil_dialtions = []
right_mean_percent_change_pupil_dialtions = []
right_max_percent_change_pupil_dialtions = []
for id in range(16):
    print("--------------------------------- id: ", id)
    figure_percent_change_pupil_dialtions = pupil_size_pId_stage_1.loc[(pupil_size_pId_stage_1["ActivatedModelIndex"] == id)]
    left_percent_change_pupil_dialtions.append(figure_percent_change_pupil_dialtions["LeftPercentChangePupilDialtion"].values.reshape(1, -1)[0])
    left_mean_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["LeftPercentChangePupilDialtion"].mean()])
    left_max_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["LeftPercentChangePupilDialtion"].max()])

    right_percent_change_pupil_dialtions.append(figure_percent_change_pupil_dialtions["RightPercentChangePupilDialtion"].values.reshape(1, -1)[0])
    right_mean_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["RightPercentChangePupilDialtion"].mean()])
    right_max_percent_change_pupil_dialtions.append([figure_percent_change_pupil_dialtions["RightPercentChangePupilDialtion"].max()])

file_name = '{}/Cognitive_Activity_Pupilometry_boxplot_cluster_no_specifications.png'.format(path_seperate_cluster)
plot_cognetive_activity_as_boxplot(data_l=left_percent_change_pupil_dialtions, data_r=right_percent_change_pupil_dialtions,
                                   legend_label="", title="Cognitive Activity Pupilometry no specifications boxplot" , file_name=file_name, save=True)

file_name = '{}/Cognitive_Activity_Pupilometry_scatterplot_cluster_no_specifications.png'.format(path_seperate_cluster)
plot_cognetive_activity_as_scatterplot(data_l_mean=left_mean_percent_change_pupil_dialtions, data_l_max=left_max_percent_change_pupil_dialtions, 
                                       data_r_mean=right_mean_percent_change_pupil_dialtions, data_r_max=right_max_percent_change_pupil_dialtions, 
                                       legend_label="", title="Cognitive Activity Pupilometry no specifications scatter", file_name=file_name, save=True)

#sys.exit()
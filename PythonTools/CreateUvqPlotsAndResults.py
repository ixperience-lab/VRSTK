import pandas as pd
import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# org_indexes_from_paper: All different characters used for evaluation. 1. Eyebot, 2. Turret, 3. JRRobo, 4. Lloyd, 5. Atlas, 6. 
#                         Ribbot, 7. Katie, 8. Alice, 9. Freddy, 10. Medic, 11. Link, 12. Dutchess, 13. Zombie, 14. MixamoGirl, 15. Remy
# Index_numbers:  Gettie (not include in the massure of UV) = 0; Eyebot = 1; Turret = 2; minitrileglv1galaxy(JRRobo) = 3; Lloid = 4; Atlas = 5; 
#                 ACPC_Ribbot = 6; Katie = 7; ACPC_Alice = 8; Freddy = 9; MedicBot = 10; link = 11; Duchess = 12; Pose_Zombiegirl = 13; Pose_MixamoGirl = 14; Pose_remy = 15;
# participants ids = [ 1,2,3,4,5,6,7,10,13,14,15,16,17,18,19,20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31,34]

def plot_uncanny_valley_results(title, xlabel, ylabel, x, y, original_y, original_color, color, original_legend_lebel, legend_label, file_path, save=False, show=False, invert_y_axis=False):
    major_ticks_top=np.linspace(0,20,41)
    
    plt.figure(figsize=(15,10))
    plt.rcParams["figure.autolayout"] = True
    plt.tick_params(axis='x', pad=60)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(major_ticks_top, labels="0 1 2 3 4 5 6 7", fontsize=14)
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.plot(x, y, color=color, zorder=1)
    plt.scatter(x, y, color=color, zorder=3, label=legend_label) 
    plt.plot(x, original_y, color=original_color, zorder=1)
    plt.scatter(x, original_y, color=original_color, zorder=3, label=original_legend_lebel) 
    plt.grid(which="major", alpha=0.6)
    plt.grid(which="minor", alpha=0.6)
    plt.legend(title="Device: ", title_fontsize=18, bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
    plt.tight_layout()    

    if invert_y_axis:
        plt.gca().invert_yaxis()
    if save:
        plt.savefig(file_path)
    if show:
        plt.show()

    plt.close()

def compute_numpy_arrays_for_plot(data_frame, questionId, models_names_alias_index_mean_array):
    avgArrayOfAllModelsHumanLikeness=[]
    nameArrayOfAllModelsHumanLikeness=[]
    
    for _, row in data_frame.iterrows():
        question = str(row[0])
        if questionId in question:
            avgArrayOfAllModelsHumanLikeness.append(float(str(row[3]).replace(",", ".")))
            nameArrayOfAllModelsHumanLikeness.append(str(row[0]).replace(questionId, ""))

    for i, model_name in  enumerate(nameArrayOfAllModelsHumanLikeness):
        for j in range(len(models_names_alias_index_mean_array)):
            if models_names_alias_index_mean_array[j][0] == model_name:
                models_names_alias_index_mean_array[j][3] = float(avgArrayOfAllModelsHumanLikeness[i])
                break

    models_names_alias_index_mean_array_np = np.array(models_names_alias_index_mean_array) 
    return np.array(models_names_alias_index_mean_array_np[:][:,1]), np.array(models_names_alias_index_mean_array_np[:][:,3]).astype('float32') 


condition = "A"
selected_condition = "Condition " + condition
folder_path = "../RTools/{}/RResults/Questionnaires".format(selected_condition)

# Create array with names and aliases for preprocess the data frame
models_names_alias_index_mean_array = [["Eyebot", "Eyebot", 1, 0.0], ["Turret", "Turret", 2, 0.0], ["minitrileglv1galaxy", "JRRobo", 3, 0.0], 
                                       ["Lloid", "Lloid", 4, 0.0], ["Atlas", "Atlas", 5, 0.0], ["ACPC_Ribbot", "Ribbot", 6, 0.0], 
                                       ["Katie", "Katie", 7, 0.0], ["ACPC_Alice", "Alice", 8, 0.0], ["Freddy", "Freddy", 9, 0.0], 
                                       ["MedicBot", "MedicBot", 10, 0.0], ["link", "link", 11, 0.0], ["Duchess", "Duchess", 12, 0.0], 
                                       ["Pose_Zombiegirl", "Zombie", 13, 0.0], ["Pose_MixamoGirl", "MixamoGirl", 14, 0.0], ["Pose_remy", "Remy", 15, 0.0]]

# original eerniness mean values from paper 
original_mean_human_likeness_array_np = np.array([1.57, 2.43, 2.14, 3.36, 4.5, 4.93, 5.29, 5.57, 5.79, 6.36, 7.07, 6.79, 8.14, 8.36, 8.64])
original_mean_human_likeness_array_np = (original_mean_human_likeness_array_np / 9) * 7
print(original_mean_human_likeness_array_np)

# original eerniness mean values from paper 
original_mean_eerniness_array_np = np.array([4.79, 3.93, 3.5, 2.79, 3.71, 3.43, 2.29, 2.71, 5.14, 5.07, 5.57, 5.64, 6.14, 1.14, 2.57])

# original likability mean values from paper 
original_mean_likability_array_np = np.array([2.43 , 3.14, 3.86, 5.43, 3.79, 4.71, 5.29, 4.5, 2.64, 2.79, 3.36, 2.07, 1.64, 6.07, 4.0])

# -----------------------------------------------------------------------------------------------------------------------
# Uncanny Valley of one conditions statistic results file
# -----------------------------------------------------------------------------------------------------------------------

uncanny_valley_of_one_conditions_file_name = "{}/AllUncannyValleyConditionStatisticResults_DataFrame.csv".format(folder_path)
print(uncanny_valley_of_one_conditions_file_name)
loaded_data_frame = pd.read_csv(uncanny_valley_of_one_conditions_file_name, sep=";")

# ---------------------------------------------------
# Perceived human likeness
# -----------------------------------------------------------------------------------------------------------------------

nameArray_np, avgArray_np = compute_numpy_arrays_for_plot(loaded_data_frame, "q1_", models_names_alias_index_mean_array)

plot_file_name_to_save = "{}/AllUncannyValleyCondition-PerceivedHumanLikeness.png".format(folder_path)
plot_uncanny_valley_results("Perceived human likeness of all evaluated 3D-Models of {} participant Group".format(selected_condition), 
                            "3D-Models", "Avg. human likeness", 
                            nameArray_np, avgArray_np, original_mean_human_likeness_array_np, "red", "blue", 
                            "HMD paper version", "HMD {}".format(selected_condition), plot_file_name_to_save, save=True, show=True)
# ---------------------------------------------------

# ---------------------------------------------------
# Perceived eeriness
# -----------------------------------------------------------------------------------------------------------------------

nameArray_np, avgArray_np = compute_numpy_arrays_for_plot(loaded_data_frame, "q2_", models_names_alias_index_mean_array)

plot_file_name_to_save = "{}/AllUncannyValleyCondition-PerceivedEeriness.png".format(folder_path)
plot_uncanny_valley_results("Perceived eeriness of all evaluated 3D-Models of {} participant Group".format(selected_condition),
                            "3D-Models", "Avg. eeriness", 
                            nameArray_np, avgArray_np, original_mean_eerniness_array_np, "red", "blue", 
                            "HMD paper version", "HMD {}".format(selected_condition), plot_file_name_to_save, save=True, show=False, invert_y_axis=True)
# ---------------------------------------------------

# ---------------------------------------------------
# Perceived likability
# -----------------------------------------------------------------------------------------------------------------------

nameArray_np, avgArray_np = compute_numpy_arrays_for_plot(loaded_data_frame, "q3_", models_names_alias_index_mean_array)

plot_file_name_to_save = "{}/AllUncannyValleyCondition-Likability.png".format(folder_path)
plot_uncanny_valley_results("Perceived likability of all evaluated 3D-Models of {} participant Group".format(selected_condition),
                            "3D-Models", "Avg. likability", 
                            nameArray_np, avgArray_np, original_mean_likability_array_np, "red", "blue", 
                            "HMD paper version", "HMD {}".format(selected_condition), plot_file_name_to_save, save=True, show=False)
# ---------------------------------------------------

#sys.exit()

# -----------------------------------------------------------------------------------------------------------------------
# Uncanny Valley of All Conditions statistic results file
# -----------------------------------------------------------------------------------------------------------------------

uncanny_valley_of_one_conditions_file_name = "{}/AllUncannyValleyStatisticResults_DataFrame.csv".format(folder_path)
print(uncanny_valley_of_one_conditions_file_name)
loaded_data_frame = pd.read_csv(uncanny_valley_of_one_conditions_file_name, sep=";")

# ---------------------------------------------------
# Perceived human likeness
# -----------------------------------------------------------------------------------------------------------------------

nameArray_np, avgArray_np = compute_numpy_arrays_for_plot(loaded_data_frame, "q1_", models_names_alias_index_mean_array)

plot_file_name_to_save = "{}/AllUncannyValley-PerceivedHumanLikeness.png".format(folder_path)
plot_uncanny_valley_results("Perceived human likeness of all evaluated 3D-Models of all Condition participant Group", 
                            "3D-Models", "Avg. human likeness", 
                            nameArray_np, avgArray_np, original_mean_human_likeness_array_np, "red", "blue", 
                            "HMD paper version", "HMD {}".format("all Conditions"), plot_file_name_to_save, save=True, show=False)
# ---------------------------------------------------

# ---------------------------------------------------
# Perceived eeriness
# -----------------------------------------------------------------------------------------------------------------------

nameArray_np, avgArray_np = compute_numpy_arrays_for_plot(loaded_data_frame, "q2_", models_names_alias_index_mean_array)

plot_file_name_to_save = "{}/AllUncannyValley-PerceivedEeriness.png".format(folder_path)
plot_uncanny_valley_results("Perceived eeriness of all evaluated 3D-Models of all Condition participant Group", 
                            "3D-Models", "Avg. eeriness", 
                            nameArray_np, avgArray_np, original_mean_eerniness_array_np, "red", "blue", 
                            "HMD paper version", "HMD {}".format("all Conditions"), plot_file_name_to_save, save=True, show=False, invert_y_axis=True)
# ---------------------------------------------------

# ---------------------------------------------------
# Perceived likability
# -----------------------------------------------------------------------------------------------------------------------

nameArray_np, avgArray_np = compute_numpy_arrays_for_plot(loaded_data_frame, "q3_", models_names_alias_index_mean_array)

plot_file_name_to_save = "{}/AllUncannyValley-Likability.png".format(folder_path)
plot_uncanny_valley_results("Perceived likability of all evaluated 3D-Models of all Condition participant Group", 
                            "3D-Models", "Avg. likability", 
                            nameArray_np, avgArray_np, original_mean_likability_array_np, "red", "blue", 
                            "HMD paper version", "HMD {}".format("all Conditions"), plot_file_name_to_save, save=True, show=False)
# ---------------------------------------------------

#sys.exit()
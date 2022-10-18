import pandas as pd
import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


condition = "A"
selected_condition = "Condition " + condition
folder_path = "../RTools/{}/RResults/Questionnaires".format(selected_condition)

# -----------------------------------------------------------------------------------------------------------------------
# Motion sickness questionnairy statistic results file
# -----------------------------------------------------------------------------------------------------------------------
# Ratingskala-1:    täglich = 0; mehrmals pro Woche = 1; einmal pro Woche = 2; mehrmals pro Monat = 3; seltener = 4;
# --------------
# Ratingskala-2:    nicht zutreffend = 0; nie krank gefühlt = 1; selten krank gefühlt = 2; manchmal krank gefühlt = 3; öfters krank gefühlt = 4;
if "A" in condition:
    motion_sickness_questionnairy_file_name = "{}/AllMSSQ_Pure_StatisticResults_DataFrame.csv".format(folder_path)
    print(motion_sickness_questionnairy_file_name)
    loaded_motion_sickness_data_frame = pd.read_csv(motion_sickness_questionnairy_file_name, sep=";", decimal=",")
    print(loaded_motion_sickness_data_frame.head(5))

    titel = "Wie oft haben Sie sich krank gefühlt oder Übelkeit verspürt?"
    ratingscale_info = 'Ratingskala:\n' + 'nicht zutreffend = 0\n' + 'nie krank gefühlt = 1\n' + 'selten krank gefühlt = 2\n' + 'manchmal krank gefühlt = 3\n' + 'öfters krank gefühlt = 4'
    questions = ['Auto', 'Bus/Reisebus', 'Zug', 'Flugzeug', 'klein Bus', 'Schiff', 'Schauckel', 'Karussell', 'Achterbahn' ]
    avg_motion_sickness = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    questions_as_sticks = np.arange(9)

    data_young = loaded_motion_sickness_data_frame.loc[6:14, 'mean'].to_numpy(dtype='float32')
    data_older = loaded_motion_sickness_data_frame.loc[15:23, 'mean'].to_numpy(dtype='float32')

    plot_file_name_to_save = "{}/MSSQ_median_plot.png".format(folder_path)
    major_ticks_top=np.linspace(0,20,41)
    plt.figure(figsize=(17,10))
    plt.xticks(questions_as_sticks, questions)
    plt.ylim(top=4)
    plt.yticks(avg_motion_sickness, "0 1 2 3 4") 
    plt.xlabel("Modalitäten", fontsize=12)
    plt.ylabel("Mittelwert der Ratings", fontsize=12)
    plt.bar(questions_as_sticks - 0.10, data_young, color = 'b', width = 0.20, label="Als Kind, jünger als 12 Jahre", zorder=2)
    plt.bar(questions_as_sticks + 0.10, data_older, color = 'r', width = 0.20, label="Über die letzten 10 Jahre", zorder=2)
    plt.grid(which="major", alpha=0.4)
    plt.grid(which="minor", alpha=0.4)
    plt.legend(title=ratingscale_info, bbox_to_anchor=(1, 0.5), loc='center left')
    plt.title(titel)
    plt.tight_layout()
    plt.savefig(plot_file_name_to_save)
    #plt.show()
    plt.close()

    titel = "Wie oft werden die einzelnen Modalitäten verwendet?"
    ratingscale_info = 'Ratingskala:\n' + 'täglich = 0\n' + 'mehrmals pro Woche = 1\n' + 'einmal pro Woche = 2\n' + 'mehrmals pro Monat = 3\n' + 'seltener = 4'
    questions = ['Eintauchen in VR', 'Auto fahren', 'Rennsimulationen spielen']
    avg_motion_sickness = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    questions_as_sticks = np.arange(3)

    data = loaded_motion_sickness_data_frame.loc[3:5, 'mean'].to_numpy(dtype='float32')
    
    plot_file_name_to_save = "{}/MSSQ_used_modelity_median_plot.png".format(folder_path)
    major_ticks_top=np.linspace(0,20,41)
    plt.figure(figsize=(17,10))
    plt.xticks(questions_as_sticks, questions)
    plt.ylim(top=4)
    plt.yticks(avg_motion_sickness, "0 1 2 3 4") 
    plt.xlabel("Modalitäten", fontsize=12)
    plt.ylabel("Mittelwert der Ratings", fontsize=12)
    plt.bar(questions_as_sticks, data, color = 'b', width = 0.20, zorder=2)
    plt.grid(which="major", alpha=0.4)
    plt.grid(which="minor", alpha=0.4)
    plt.legend(title=ratingscale_info, bbox_to_anchor=(1, 0.5), loc='center left')
    plt.title(titel)
    plt.tight_layout()
    plt.savefig(plot_file_name_to_save)
    #plt.show()
    plt.close()

# -----------------------------------------------------------------------------------------------------------------------
# Simulation Sickness Questionnairy statistic results file
# -----------------------------------------------------------------------------------------------------------------------
# Ratingskala: Keine = 0, Leicht = 1, Moderat = 2, Stark = 3

simulation_sickness_questionnairy_file_name = "{}/AllSSQStatisticResults_DataFrame.csv".format(folder_path)
print(simulation_sickness_questionnairy_file_name)
loaded_simulation_sickness_data_frame = pd.read_csv(simulation_sickness_questionnairy_file_name, sep=";", decimal=",")
print(loaded_simulation_sickness_data_frame.head(5))

titel = "Simulation-Sickness-Questionnairy anworten"
ratingscale_info = 'Ratingskala (empfinden eines Symptoms):\n' + 'Keine = 0\n' + 'Leicht = 1\n' + 'Moderat = 2\n' + 'Stark = 3'
questions = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16']
avg_motion_sickness = [0, 0.5, 1, 1.5, 2, 2.5, 3]
questions_as_sticks = np.arange(16)

data = loaded_simulation_sickness_data_frame.loc[:, 'mean'].to_numpy(dtype='float32')

print(len(data))
print(len(questions_as_sticks))

plot_file_name_to_save = "{}/SSQ_median_plot.png".format(folder_path)
major_ticks_top=np.linspace(0,20,41)
plt.figure(figsize=(17,10))
plt.xticks(questions_as_sticks, questions)
plt.ylim(top=4)
plt.yticks(avg_motion_sickness, "0 1 2 3") 
plt.xlabel("Fragen", fontsize=12)
plt.ylabel("Mittelwert der Ratings", fontsize=12)
plt.bar(questions_as_sticks, data, color = 'b', width = 0.20, zorder=2)
plt.grid(which="major", alpha=0.4)
plt.grid(which="minor", alpha=0.4)
plt.legend(title=ratingscale_info, bbox_to_anchor=(1, 0.5), loc='center left')
plt.title(titel)
plt.tight_layout()
plt.savefig(plot_file_name_to_save)
#plt.show()
plt.close()

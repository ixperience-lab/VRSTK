import pandas as pd
import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
# einfaktorielle anaylse anova
from scipy.stats import f_oneway
from scipy.stats import anderson
from scipy.stats import mannwhitneyu

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
    loaded_motion_sickness_data_frame = pd.read_csv(motion_sickness_questionnairy_file_name, sep=";", decimal=",")#, encoding='ANSI')
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
    plt.figure(figsize=(15,10))
    plt.xticks(questions_as_sticks, questions, fontsize=14)
    plt.ylim(top=4)
    plt.yticks(avg_motion_sickness, "0 1 2 3 4", fontsize=14) 
    plt.xlabel("Modalitäten", fontsize=16)
    plt.ylabel("Mittelwert der Ratings", fontsize=16)
    plt.bar(questions_as_sticks - 0.10, data_young, color = 'blue', width = 0.20, label="< 12 Jahre (Kind)", zorder=2)
    plt.bar(questions_as_sticks + 0.10, data_older, color = 'red', width = 0.20, label="Letzten 10 Jahre", zorder=2)
    plt.grid(which="major", alpha=0.4)
    plt.grid(which="minor", alpha=0.4)
    plt.legend(title=ratingscale_info, title_fontsize=18, bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
    plt.title(titel, fontsize=18)
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
    plt.figure(figsize=(15,10))
    plt.xticks(questions_as_sticks, questions, fontsize=14)
    plt.ylim(top=4)
    plt.yticks(avg_motion_sickness, "0 1 2 3 4", fontsize=14) 
    plt.xlabel("Modalitäten", fontsize=16)
    plt.ylabel("Mittelwert der Ratings", fontsize=16)
    plt.bar(questions_as_sticks, data, color = 'blue', width = 0.20, zorder=2)
    plt.grid(which="major", alpha=0.4)
    plt.grid(which="minor", alpha=0.4)
    plt.legend(title=ratingscale_info, title_fontsize=18, bbox_to_anchor=(1, 0.5), loc='center left', fontsize=16)
    plt.title(titel, fontsize=18)
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

titel = "Simulation-Sickness-Questionnairy Anworten"
ratingscale_info = 'Ratingskala (Symptome):\n' + 'Keine = 0\n' + 'Leicht = 1\n' + 'Moderat = 2\n' + 'Stark = 3'
questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16']
avg_motion_sickness = [0, 0.5, 1, 1.5, 2, 2.5, 3]
questions_as_sticks = np.arange(16)

data = loaded_simulation_sickness_data_frame.loc[:, 'mean'].to_numpy(dtype='float32')

print(len(data))
print(len(questions_as_sticks))

plot_file_name_to_save = "{}/SSQ_median_plot.png".format(folder_path)
major_ticks_top=np.linspace(0,20,41)
plt.figure(figsize=(15,10))
plt.xticks(questions_as_sticks, questions, fontsize=14)
plt.ylim(top=4)
plt.yticks(avg_motion_sickness, "0 1 2 3", fontsize=14) 
plt.xlabel("Fragen", fontsize=16)
plt.ylabel("Mittelwert der Ratings", fontsize=16)
plt.bar(questions_as_sticks, data, color = 'blue', width = 0.20, zorder=2)
plt.grid(which="major", alpha=0.4)
plt.grid(which="minor", alpha=0.4)
plt.legend(title=ratingscale_info, title_fontsize=18, bbox_to_anchor=(1, 0.5), loc='center left', fontsize=14)
plt.title(titel, fontsize=18)
plt.tight_layout()
plt.savefig(plot_file_name_to_save)
#plt.show()
plt.close()



# -----------------------------------------------------------------------------------------------------------------------
# Simulation Sickness Questionnairy statistic results file anova test
# -----------------------------------------------------------------------------------------------------------------------
# Ratingskala: Keine = 0, Leicht = 1, Moderat = 2, Stark = 3

# Condition A
folder_path = "../RTools/Condition A/RResults/Questionnaires".format(selected_condition)
simulation_sickness_questionnairy_file_name_a = "{}/AllSSQConditionStatisticResults_DataFrame.csv".format(folder_path)
print(simulation_sickness_questionnairy_file_name_a)
loaded_simulation_sickness_data_frame_a = pd.read_csv(simulation_sickness_questionnairy_file_name_a, sep=";", decimal=",")
print(loaded_simulation_sickness_data_frame_a.head(5))

# questionnaire_ID_SSQ_ALL_Condition_A_SSQAnswers = "../RTools/Condition A/Questionnaires/Answers/questionnaireID_SSQ_ALL_Condition-A_SSQAnswers.csv".format(folder_path)
# data_Condition_A = pd.read_csv(questionnaire_ID_SSQ_ALL_Condition_A_SSQAnswers, sep=";", decimal=",")
# print(data_Condition_A.head(5))

# data_Condition_A["weights"] = (data_Condition_A['Answer_Participant_id-2_condition_Condition A'].values + 
#            data_Condition_A['Answer_Participant_id-1_condition_Condition A'].values +
#            data_Condition_A['Answer_Participant_id-3_condition_Condition A'].values +
#            data_Condition_A['Answer_Participant_id-5_condition_Condition A'].values +
#            data_Condition_A['Answer_Participant_id-6_condition_Condition A'].values +
#            data_Condition_A['Answer_Participant_id-7_condition_Condition A'].values +
#            data_Condition_A['Answer_Participant_id-4_condition_Condition A'].values +
#            data_Condition_A['Answer_Participant_id-10_condition_Condition A'].values)

# plt.hist(x=[0, 1, 2, 3, 4, 5, 6 ,7 , 8 , 9 , 10, 11 , 12, 13, 14, 15], bins=[0, 1, 2, 3, 4, 5, 6 ,7 , 8 , 9 , 10, 11 , 12, 13, 14, 15], weights=data_Condition_A["weights"])
# plt.ylabel('Probability')
# plt.xlabel('Data')
# plt.show()

# questionnaire_ID_SSQ_ALL_Condition_B_SSQAnswers = "../RTools/Condition B/Questionnaires/Answers/questionnaireID_SSQ_ALL_Condition-B_SSQAnswers.csv".format(folder_path)
# data_Condition_B = pd.read_csv(questionnaire_ID_SSQ_ALL_Condition_B_SSQAnswers, sep=";", decimal=",")
# print(data_Condition_B.head(5))

# data_Condition_B["weights"] = (data_Condition_B['Answer_Participant_id-13_condition_Condition B'].values + 
#            data_Condition_B['Answer_Participant_id-14_condition_Condition B'].values +
#            data_Condition_B['Answer_Participant_id-15_condition_Condition B'].values +
#            data_Condition_B['Answer_Participant_id-16_condition_Condition B'].values +
#            data_Condition_B['Answer_Participant_id-17b_condition_Condition B'].values +
#            data_Condition_B['Answer_Participant_id-18_condition_Condition B'].values +
#            data_Condition_B['Answer_Participant_id-19_condition_Condition B'].values +
#            data_Condition_B['Answer_Participant_id-20_condition_Condition B'].values +
#            data_Condition_B['Answer_Participant_id-31_condition_Condition B'].values +
#            data_Condition_B['Answer_Participant_id-34_condition_Condition B'].values)

# plt.hist(x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], weights=data_Condition_B["weights"])
# #plt.hist(data_Condition_A, density=True, bins=30)  # density=False would make counts
# plt.ylabel('Probability')
# plt.xlabel('Data')
# plt.show()



# Condition B
folder_path = "../RTools/Condition B/RResults/Questionnaires".format(selected_condition)
simulation_sickness_questionnairy_file_name_b = "{}/AllSSQConditionStatisticResults_DataFrame.csv".format(folder_path)
print(simulation_sickness_questionnairy_file_name_b)
loaded_simulation_sickness_data_frame_b = pd.read_csv(simulation_sickness_questionnairy_file_name_b, sep=";", decimal=",")
print(loaded_simulation_sickness_data_frame_b.head(5))

folder_path = "../RTools/Condition C/RResults/Questionnaires".format(selected_condition)
simulation_sickness_questionnairy_file_name_c = "{}/AllSSQConditionStatisticResults_DataFrame.csv".format(folder_path)
print(simulation_sickness_questionnairy_file_name_c)
loaded_simulation_sickness_data_frame_c = pd.read_csv(simulation_sickness_questionnairy_file_name_c, sep=";", decimal=",")
print(loaded_simulation_sickness_data_frame_c.head(5))

data_a = loaded_simulation_sickness_data_frame_a.loc[:, 'mean'].to_numpy(dtype='float32')
data_b = loaded_simulation_sickness_data_frame_b.loc[:, 'mean'].to_numpy(dtype='float32')
data_c = loaded_simulation_sickness_data_frame_c.loc[:, 'mean'].to_numpy(dtype='float32')

# Anderson-Normality-Test
print("Anderson-Test for check data to normal distribution\n------------------------------------------------------------\n")
anova_test_content = "Anderson-Test for check data to normal distribution\n------------------------------------------------------------\n"
statistic = anderson(data_a, dist='norm')
anova_test_content = "{}Condition Group A :\n statistic: {} \n".format(anova_test_content, statistic)
print(anova_test_content)
statistic = anderson(data_b, dist='norm')
anova_test_content = "{}Condition Group B :\n statistic: {} \n".format(anova_test_content, statistic)
print(anova_test_content)
statistic = anderson(data_c, dist='norm')
anova_test_content = "{}Condition Group C :\n statistic: {} \n".format(anova_test_content, statistic)
print(anova_test_content)
print("\n------------------------------------------------------------\n")
anova_test_content = "{}\n------------------------------------------------------------\n".format(anova_test_content)

# ANOVA test
# ------------
anova_test_content = "{}ANOVA SSQ-Gruppen-Test    (F-Value)   (p-Value)   conditions-based\n------------------------------------------------------------\n".format(anova_test_content)
print("ANOVA SSQ-Gruppen-Test")
print("------------------------------------------------------------ ")

fvalue, pvalue = f_oneway(data_a, data_b)
anova_test_content = "{}On A and B: & \({}\)      & \({}\)\n".format(anova_test_content, fvalue, pvalue)
print("On A and B")
print("F-Value: {}  p-Value: {}".format(fvalue, pvalue))

fvalue, pvalue = f_oneway(data_a, data_c)
anova_test_content = "{}On A and C: & \({}\)      & \({}\)\n".format(anova_test_content, fvalue, pvalue)
print("On A and C")
print("F-Value: {}  p-Value: {}".format(fvalue, pvalue))

fvalue, pvalue = f_oneway(data_b, data_c)
anova_test_content = "{}On B and C: & \({}\)      & \({}\)\n".format(anova_test_content, fvalue, pvalue)
print("On B and C")
print("F-Value: {}  p-Value: {}".format(fvalue, pvalue))

fvalue, pvalue = f_oneway(data_a, data_b, data_c)
anova_test_content = "{}On A and B and C: & \({}\)      & \({}\)\n".format(anova_test_content,fvalue, pvalue)
print("On A and B and C")
print("F-Value: {}  p-Value: {}".format(fvalue, pvalue))
anova_test_content = "{}\n------------------------------------------------------------\n".format(anova_test_content)

# Mann-Whitney U test
# ------------
anova_test_content = "{}Mann-Whitney U test SSQ-Gruppen-Test    (W-statistic)   (p-Value)   conditions-based\n------------------------------------------------------------\n".format(anova_test_content)
print("Mann-Whitney U test SSQ-Gruppen-Test")
print("------------------------------------------------------------ ")
statistic, pvalue = mannwhitneyu(data_a, data_b, method="exact")
anova_test_content = "{}On A and B: & \({}\)      & \({}\)\n".format(anova_test_content, statistic, pvalue)
print("On A and B")
print("statistic: {}  p-Value: {}".format(fvalue, pvalue))

statistic, pvalue = mannwhitneyu(data_a, data_c)
anova_test_content = "{}On A and C: & \({}\)      & \({}\)\n".format(anova_test_content, statistic, pvalue)
print("On A and C")
print("statistic: {}  p-Value: {}".format(fvalue, pvalue))

statistic, pvalue = mannwhitneyu(data_b, data_c, method="exact")
anova_test_content = "{}On B and C: & \({}\)      & \({}\)\n".format(anova_test_content, statistic, pvalue)
print("On B and C")
print("statistic: {}  p-Value: {}".format(fvalue, pvalue))
anova_test_content = "{}\n------------------------------------------------------------\n".format(anova_test_content)

file_name = "../RTools/SSQ_ANOVA_Results.txt"
file = open(file_name, "w")
file.write(anova_test_content)
file.close()
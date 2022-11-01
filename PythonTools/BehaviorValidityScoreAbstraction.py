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

# EEG bandpower waves for each sensor on headset
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

input_stage_0["theta_scaled"] = StandardScaler().fit_transform(input_stage_0[["theta"]])
input_stage_0["alpha_scaled"] = StandardScaler().fit_transform(input_stage_0[["alpha"]])
input_stage_0["betaL_scaled"] = StandardScaler().fit_transform(input_stage_0[["betaL"]])
input_stage_0["betaH_scaled"] = StandardScaler().fit_transform(input_stage_0[["betaH"]])
input_stage_0["gamma_scaled"] = StandardScaler().fit_transform(input_stage_0[["gamma"]])

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

input_stage_1["theta_scaled"] = StandardScaler().fit_transform(input_stage_1[["theta"]])
input_stage_1["alpha_scaled"] = StandardScaler().fit_transform(input_stage_1[["alpha"]])
input_stage_1["betaL_scaled"] = StandardScaler().fit_transform(input_stage_1[["betaL"]])
input_stage_1["betaH_scaled"] = StandardScaler().fit_transform(input_stage_1[["betaH"]])
input_stage_1["gamma_scaled"] = StandardScaler().fit_transform(input_stage_1[["gamma"]])

eeg_low_score_stage_1 = input_stage_1.loc[(input_stage_1["EvaluatedGlobalTIMERSICalc"] == 0) & (input_stage_1["DegTimeLowQuality"] == 0)]

eeg_high_score_stage_1 = input_stage_1.loc[(input_stage_1["EvaluatedGlobalTIMERSICalc"] > 0) & (input_stage_1["DegTimeLowQuality"] > 0)]

file_name = '{}/EEG_beta_gamma_boxplot.png'.format(path_eeg)
plt.figure(figsize=(15,10))
plt.rcParams["figure.autolayout"] = True
plt.title("EEG Behavior Validity Score Abstraction", fontsize=16)
width = 0.2
plt.boxplot(eeg_low_score_stage_1["theta_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[-0.25], widths=width)
plt.boxplot(eeg_low_score_stage_1["alpha_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[0], widths=width)
plt.boxplot(eeg_low_score_stage_1["betaL_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[0.25], widths=width)
plt.boxplot(eeg_low_score_stage_1["betaH_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[0.5], widths=width)
plt.boxplot(eeg_low_score_stage_1["gamma_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[0.75], widths=width)

plt.boxplot(eeg_high_score_stage_1["theta_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[1.75], widths=width)
plt.boxplot(eeg_high_score_stage_1["alpha_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[2], widths=width)
plt.boxplot(eeg_high_score_stage_1["betaL_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[2.25], widths=width)
plt.boxplot(eeg_high_score_stage_1["betaH_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[2.5], widths=width)
plt.boxplot(eeg_high_score_stage_1["gamma_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[2.75], widths=width)
labels_list = ['theta (low-score)', 'alpha (low-score)', 'betaL (low-score)', 'betaH (low-score)', 'gamma (low-score)', 
               'theta (high-score)', 'alpha (high-score)', 'betaL (high-score)', 'betaH (high-score)', 'gamma (high-score)']
plt.xticks([-0.25, 0, 0.25, 0.5, 0.75, 1.75, 2, 2.25, 2.5, 2.75], labels=labels_list, rotation=45, ha='right')
plt.ylabel("Scaled sum EEG Bandpower values", fontsize=12)
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
plt.savefig(file_name)
plt.close()

#sys.exit()

# HRV with LFHFRatio, SD1SD2Ratio and HeartRate
# -----------------------------------------------------
path_hrv = "{}/HeartRateVariability".format(path)
if not exists(path_hrv):
    os.mkdir(path_hrv, mode)

hrv_low_score_stage_1 = input_stage_1.loc[(input_stage_1["EvaluatedGlobalTIMERSICalc"] == 0) & (input_stage_1["DegTimeLowQuality"] == 0)]
hrv_high_score_stage_1 = input_stage_1.loc[(input_stage_1["EvaluatedGlobalTIMERSICalc"] > 0) & (input_stage_1["DegTimeLowQuality"] > 0)]

file_name = '{}/HRV_LFHFRatio_HeartRate_boxplot.png'.format(path_hrv)
plt.figure(figsize=(15,10))
plt.title("HRV Behavior Validity Score Abstraction", fontsize=16)
width = 0.2
plt.boxplot(hrv_low_score_stage_1["LFHFRatio"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[-0.25], widths=width)
plt.boxplot(hrv_low_score_stage_1["SD1SD2Ratio"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[0.25], widths=width)
plt.boxplot(hrv_high_score_stage_1["LFHFRatio"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[1.75], widths=width)
plt.boxplot(hrv_high_score_stage_1["SD1SD2Ratio"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[2.25], widths=width)
labels_list = ['LF\HF Ratio (low-score)', 'SD1\SD2 Ratio (low-score)', 'LF\HF Ratio (high-score)', 'SD1\SD2 Ratio (low-score)']
plt.xticks([-0.25, 0.25, 1.75, 2.25], labels=labels_list, rotation=45, ha='right')
plt.ylabel("HRV ratio values", fontsize=12)
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
plt.savefig(file_name)
plt.close()

# Skin Conductance FilteredValueInMicroSiemens
# -----------------------------------------------------
path_eda = "{}/SkinConductance".format(path)
if not exists(path_eda):
    os.mkdir(path_eda, mode)

input_stage_1["FilteredValueInMicroSiemens_scaled"] = StandardScaler().fit_transform(input_stage_1[["FilteredValueInMicroSiemens"]])

eda_low_score_stage_1 = input_stage_1.loc[(input_stage_1["EvaluatedGlobalTIMERSICalc"] == 0) & (input_stage_1["DegTimeLowQuality"] == 0)]
eda_high_score_stage_1 = input_stage_1.loc[(input_stage_1["EvaluatedGlobalTIMERSICalc"] > 0) & (input_stage_1["DegTimeLowQuality"] > 0)]

file_name = '{}/EDA_FilteredValueInMicroSiemens_boxplot.png'.format(path_eda)
plt.figure(figsize=(15,10))
plt.title("EDA Behavior Validity Score Abstraction", fontsize=16)
width = 0.2
plt.boxplot(eda_low_score_stage_1["FilteredValueInMicroSiemens_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[0], widths=width)
plt.boxplot(eda_high_score_stage_1["FilteredValueInMicroSiemens_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[1], widths=width)
labels_list = ['Filtered EDA Value (low-score)', 'Filtered EDA Value (high-score)']
plt.xticks([0, 1], labels=labels_list, rotation=45, ha='right')
plt.ylabel("Scaled EDA values in (Micro Siemens)", fontsize=12)
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
plt.savefig(file_name)
plt.close()

#sys.exit()

# Eye Tracking with pupillomentry
# -----------------------------------------------------
path_eye = "{}/EyeTracking".format(path)
if not exists(path_eye):
    os.mkdir(path_eye, mode)

input_stage_1["LeftPercentChangePupilDialtion"] = 0.0
input_stage_1["RightPercentChangePupilDialtion"] = 0.0

for i in [ 1, 2, 3, 4, 5, 6, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 31, 34, 21, 22, 23, 24, 25, 26, 27, 28, 29 ]:
    pId = i
    # stage 0
    base_line_pupil_diameter_stage_0 = input_stage_0.loc[(input_stage_0["pId"] == pId)]
    base_line_mean_l_pupil_diameter = base_line_pupil_diameter_stage_0["LeftPupilDiameter"].mean()
    base_line_mean_r_pupil_diameter = base_line_pupil_diameter_stage_0["RightPupilDiameter"].mean()

    pupil_size_pId_stage_1 = input_stage_1.loc[(input_stage_1["pId"] == pId)]

    input_stage_1.loc[(input_stage_1["pId"] == pId), ["LeftPercentChangePupilDialtion"]] = (
                       pupil_size_pId_stage_1["LeftPupilDiameter"] - base_line_mean_l_pupil_diameter) / base_line_mean_l_pupil_diameter

    input_stage_1.loc[(input_stage_1["pId"] == pId), ["RightPercentChangePupilDialtion"]] = (
                       pupil_size_pId_stage_1["RightPupilDiameter"] - base_line_mean_r_pupil_diameter) / base_line_mean_r_pupil_diameter

input_stage_1["LeftPercentChangePupilDialtion_scaled"] = StandardScaler().fit_transform(input_stage_1[["LeftPercentChangePupilDialtion"]])
input_stage_1["RightPercentChangePupilDialtion_scaled"] = StandardScaler().fit_transform(input_stage_1[["RightPercentChangePupilDialtion"]])

input_stage_1["TotalFixationCounter_scaled"] = StandardScaler().fit_transform(input_stage_1[["TotalFixationCounter"]])
input_stage_1["SaccadeCounter_scaled"] = StandardScaler().fit_transform(input_stage_1[["SaccadeCounter"]])

eye_low_score_stage_1 = input_stage_1.loc[(input_stage_1["EvaluatedGlobalTIMERSICalc"] == 0) & (input_stage_1["DegTimeLowQuality"] == 0)]
eye_high_score_stage_1 = input_stage_1.loc[(input_stage_1["EvaluatedGlobalTIMERSICalc"] > 0) & (input_stage_1["DegTimeLowQuality"] > 0)]

file_name = '{}/Eye_pupillometry_boxplot.png'.format(path_eye)
plt.figure(figsize=(15,10))
plt.title("Eye-Tracking (Pupillometry) Behavior Validity Score Abstraction", fontsize=16)
width = 0.2
plt.boxplot(eye_low_score_stage_1["LeftPercentChangePupilDialtion_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[0], widths=width)
plt.boxplot(eye_low_score_stage_1["RightPercentChangePupilDialtion_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[1], widths=width)

plt.boxplot(eye_high_score_stage_1["LeftPercentChangePupilDialtion_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[2], widths=width)
plt.boxplot(eye_high_score_stage_1["RightPercentChangePupilDialtion_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[3], widths=width)
labels_list = ['Left pupil dialtion (low-score)', 'Right pupil dialtion (low-score)', 'Left pupil dialtion (high-score)', 'Right pupil dialtion (high-score)']
plt.xticks([0, 1, 2, 3], labels=labels_list, rotation=45, ha='right')
plt.ylabel("Scaled pupil dialtion values", fontsize=12)
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
plt.savefig(file_name)
plt.close()

file_name = '{}/Eye_saccads_fixation_boxplot.png'.format(path_eye)
plt.figure(figsize=(15,10))
plt.title("Eye-Tracking (Fixation and Saccades) Behavior Validity Score Abstraction", fontsize=16)
width = 0.2
plt.boxplot(eye_low_score_stage_1["TotalFixationCounter_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[0], widths=width)
plt.boxplot(eye_low_score_stage_1["SaccadeCounter_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="blue"), positions=[1], widths=width)

plt.boxplot(eye_high_score_stage_1["TotalFixationCounter_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[2], widths=width)
plt.boxplot(eye_high_score_stage_1["SaccadeCounter_scaled"].values.reshape(1, -1)[0], medianprops=dict(color="orange"), positions=[3], widths=width)
labels_list = ['Total fixation number (low-score)', 'Total saccades number (low-score)', 'Total fixation number (high-score)', 'Total saccades number (high-score)']
plt.xticks([0, 1, 2, 3], labels=labels_list, rotation=45, ha='right')
plt.ylabel("Scaled total numbers of fixation and saccades", fontsize=12)
plt.grid(which="major", alpha=0.6)
plt.grid(which="minor", alpha=0.6)
plt.savefig(file_name)
plt.close()

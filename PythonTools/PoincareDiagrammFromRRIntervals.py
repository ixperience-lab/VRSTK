# Reference-Link: https://xn--hrv-herzratenvariabilitt-dcc.de/2017/07/berechnung-des-poincare-diagramms-aus-rr-intervallen/
# Standard from: https://www.ahajournals.org/doi/10.1161/01.CIR.93.5.1043
# de: Quadratwurzel des Mittelwerts der Summe aller quadrierten Differenzen zwischen benachbarten RR-Intervallen
# en: square root of the mean of the sum of the squares of differences between adjacent NN intervals
#
#

from cProfile import label
import os
from stat import FILE_ATTRIBUTE_NORMAL
import sys
import math
from turtle import width
import matplotlib.pyplot as plt
from os.path import exists
from matplotlib.patches import Ellipse
import numpy as np
from numpy.linalg import norm
from math import pi, cos, sin
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nl
from matplotlib.legend import Legend

def get_files(search_path):
     for (dirpath, _, filenames) in os.walk(search_path):
         for filename in filenames:
             yield os.path.join(dirpath, filename)

#def angle_between_two_vectors (v1, v2):
#    dot = np.dot(v1,v2)
#    x_modulus = np.sqrt((v1*v1).sum())
#    y_modulus = np.sqrt((v2*v2).sum())
#    cos_angle = dot / x_modulus / y_modulus 
#    angle = np.arccos(cos_angle) # rad
#    angle = (angle / np.pi) * 180 # deg
#    return angle

condition = "C"
selected_condition = "Condition " + condition
selected_condition_list = []
if "Condition A" in selected_condition:
    # Conditions Ids:
    # A -> id-1, id-2, id-3, id-4, id-5, id-6, id-7, id-10, (id-42 eeg-quality 0%) => 8 (9) Participents
    selected_condition_list = ["id-1", "id-2", "id-3", "id-4", "id-5", "id-6", "id-7", "id-10"]     
if "Condition B" in selected_condition:
    # B -> id-13, id-14, id-15, id-16, id-17b, id-18, id-19, id-20, id-31, id-34, (id-25) 0 => optional participent because subjective it was a none-conscientious
    selected_condition_list = ["id-13", "id-14", "id-15", "id-16", "id-17b", "id-18", "id-19", "id-20", "id-31", "id-34"]
    #selected_condition_list = ["id-13", "id-17b", "id-18", "id-19", "id-20", "id-31", "id-34"]
if "Condition C" in selected_condition:
    # C -> id-21, id-22, id-22, id-23, id-24, id-25, id-26, id-27, id-28, id-29 
    selected_condition_list = ["id-21", "id-22", "id-23", "id-24", "id-25", "id-26", "id-27", "id-28", "id-29"]

print(selected_condition)
print(selected_condition_list) 

for cId in selected_condition_list:
    file_name_0 = ""
    file_name_1 = ""
    file_name_2 = ""
    #folder_path = "./input/"
    folder_path = "../../../RTools/{}/Biosppy/{}".format(selected_condition, cId)
    filenames = []
    list_files = get_files(folder_path)
    for filename in list_files:
        if "_id-" in filename and "ECG_HearRateResults" in filename:
            filenames.append(filename)

    print(filenames)

    file_name_0 = filenames[0]
    file_name_1 = filenames[1]
    file_name_2 = filenames[2]

    # stage 0
    #--------------------------------------------------------------------------------------------
    file_path = file_name_0
    #file_path = folder_path + file_name_0
    raw_data_file = open(file_path, 'r')

    rpeak_time_input = []

    # read file
    for line in raw_data_file:
        rpeak_time_input.append(float(line.split(";")[2])) # form s to ms
        
    # Closing the file after reading
    raw_data_file.close()

    rr_intervals = []
    for index in range(len(rpeak_time_input) - 1):
        rr_intervals.append(int(rpeak_time_input[index + 1] - rpeak_time_input[index]))

    rr_min = sys.maxsize
    rr_max = 0
    rr_mean = 0

    for index in range(len(rr_intervals)):
        if rr_min > rr_intervals[index]:
            rr_min = rr_intervals[index]
        if rr_max < rr_intervals[index]:
            rr_max = rr_intervals[index]
        rr_mean += rr_intervals[index]

    rr_mean /= (len(rr_intervals))

    #rr_mean = np.mean(rr_intervals)

    rr_2d=[]

    for index in range(len(rr_intervals) - 1):
        rr_2d.append([int(rr_intervals[index]), int(rr_intervals[index + 1])])

    rr_2d_np = np.array(rr_2d)

    # Reference-Link : https://xn--hrv-herzratenvariabilitt-dcc.de/2019/01/berechnung-des-hrv-werts-sdnn/
    # sdnn calculation
    rr_intervals_sum = 0
    for index in range(len(rr_intervals)):
        if index >= 2:
            rr_intervals_sum += int(rr_intervals[index])

    rr_intervals_mean = rr_intervals_sum / (len(rr_intervals) - 1)

    rr_intervals_sum_mean = 0
    for index in range(len(rr_intervals)):
        if index >= 2:
            rr_intervals_sum_mean += (rr_intervals[index] - rr_intervals_mean)**2

    rr_intervals_sum_mean_mean = rr_intervals_sum_mean / (len(rr_intervals) - 1)
    sdnn = math.sqrt(rr_intervals_sum_mean_mean)

    # Reference-Link : https://xn--hrv-herzratenvariabilitt-dcc.de/2017/10/berechnung-des-hrv-werts-rmssd/
    # rmssd calculation
    rr_time_squered_sum_sum = 0

    for index in range(len(rr_intervals) - 1):
        expression = int(rr_intervals[index + 1] - rr_intervals[index])**2
        rr_time_squered_sum_sum += expression

    rr_time_squered_sum_sum_mean = rr_time_squered_sum_sum / (len(rr_intervals) - 1)
    rmssd = math.sqrt(rr_time_squered_sum_sum_mean)

    # Reference-Link : https://ieeexplore.ieee.org/abstract/document/959330
    # sdsd calculation
    sdsd = 0
    sdsd_sum = 0
    for index in range(len(rr_intervals) - 1):
        sdsd_sum += ( int(rr_intervals[index]) - int(rr_intervals[index + 1]) )**2

    sdsd_sum_mean = sdsd_sum / (len(rr_intervals) - 1)
    sdsd = math.sqrt(sdsd_sum_mean)

    # Reference-Link : https://onlinelibrary.wiley.com/doi/10.1002/mus.25573
    # sd1 calculation
    sd1 = 0
    sd1_sum = 0
    for index in range(len(rr_intervals) - 1):
        sd1_sum += ( ((1/math.sqrt(2)) * int(rr_intervals[index])) - ((1/math.sqrt(2)) * int(rr_intervals[index + 1])) )**2

    sd1_sum_mean = sd1_sum / (len(rr_intervals) - 1)

    sd1 = math.sqrt(sd1_sum_mean)

    # Reference-Link : https://pyhrv.readthedocs.io/en/latest/_pages/api/nonlinear.html
    # sd2 calculation
    sd2 = math.sqrt(((2 * sdnn**2) - (0.5 * sdsd**2)))

    # Reference-Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/
    # sd1 over sd2 ratio = sd1 / sd2
    sd1_over_sd2_ratio = sd1 / sd2
    
    # Reference-Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/
    # S as Ellipse erea
    s_ellipse_area = pi * sd1 * sd2

    angle =  pi / 4 # rad
    angle = (angle / np.pi) * 180 # deg
    print (angle)
    
    sd1_point = [rr_mean + (-sd1 * np.sqrt(2) / 2), rr_mean + (sd1 * np.sqrt(2) / 2)]
    rr_mean_point = [rr_mean, rr_mean]
    sd2_point = [rr_mean + (sd2) * np.cos(np.deg2rad(45)), rr_mean + (sd2) * np.sin(np.deg2rad(45))]

    # set plot parameters
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    # set plot title and axis labels
    fig.suptitle('Poincaré-Plot')
    plt.xlabel("RRi [ms]")
    plt.ylabel("RRi+1 [ms]")
    # create ellipse plot object
    ellipse = Ellipse((rr_mean, rr_mean), 2*sd2, 2*sd1, angle=angle, linewidth=1, fill=False, color="k", zorder=3)
    area = Ellipse((rr_mean, rr_mean), 2*sd2, 2*sd1, angle=angle, alpha=0.2, linewidth=1, fill=True, color="yellow", zorder=2)
    ax = plt.gca()
    ax.add_patch(ellipse)
    ax.add_patch(area)
    # set limits    
    plt.xlim([rr_min, rr_max])
    plt.ylim([rr_min, rr_max])
    # set axes identity lines
    x_values = [rr_min, rr_max]
    y_values = [rr_min, rr_max]
    plt.plot(x_values, y_values, color='green', zorder=1, linewidth=1, linestyle="--", label="x-Axe identity line")
    x_values = [rr_max, rr_min]
    y_values = [rr_min, rr_max]
    plt.plot(x_values, y_values, color='blue', zorder=1, linewidth=1, linestyle="--", label="y-Axe identity line")
    # sd1 point and vector
    plt.scatter(sd1_point[0], sd1_point[1], color='blue', zorder=4, label="SD1: " + str(sd1))
    plt.arrow(rr_mean_point[0], rr_mean_point[1], (-sd1) * np.cos(np.deg2rad(angle)), (sd1) * np.sin(np.deg2rad(angle)), color="blue", linewidth=1, zorder=3)
    # rr_intervals mean point
    plt.scatter(rr_mean_point[0], rr_mean_point[1], color='red', zorder=4, label="RR-Intervals-Mean: " + str(rr_mean))
    # sd2 point and vector
    plt.scatter(sd2_point[0], sd2_point[1], color='green', zorder=4, label="SD2: " + str(sd2))
    plt.arrow(rr_mean_point[0], rr_mean_point[1], (sd2) * np.cos(np.deg2rad(angle)), (sd2) * np.sin(np.deg2rad(angle)), color="green", linewidth=1, zorder=3)
    # rr_intervals points and lines
    plt.plot(rr_2d_np[:,0], rr_2d_np[:,1], color='gray', zorder=1)
    plt.scatter(rr_2d_np[:,0], rr_2d_np[:,1], zorder=2, label="RR-Intervals-Points")
    # Create empty plot with blank marker containing the extra label
    plt.plot([], [], ' ', label="S (Ellipse area): " + str(s_ellipse_area))
    # activate labels as legend
    plt.legend()
    # activate plot grid
    plt.grid()
    # make layout tight
    fig.tight_layout()
    # save to file
    generatet_file_name = "Bitalinoi-Proband-Stage-0_{}_{}{}".format(cId, selected_condition.replace(' ','-'), "_ECG_HearRateVariabilityResults.png")
    path_to_heart_rate_variability_fig_file = "{}/{}".format(folder_path, generatet_file_name)
    fig.savefig(path_to_heart_rate_variability_fig_file, dpi=600, bbox_inches='tight')
    # show only for debug
    #plt.show()
    plt.close()

    # Reference-Link: https://pyhrv.readthedocs.io/en/latest/_pages/api/frequency.html
    # simple solution/std (welch-fft) solution for HRV frequence domain
    # VLF: [0.00Hz - 0.04Hz]
    # LF: [0.04Hz - 0.15Hz]
    # HF: [0.15Hz - 0.40Hz]
    result = fd.welch_psd(rpeaks=rpeak_time_input,  show=False, show_param=False, legend=False)
    
    vlf_peak =  result['fft_peak'][0]
    lf_peak  =  result['fft_peak'][1]
    hf_peak  =  result['fft_peak'][2]

    vlf_abs =  result['fft_abs'][0]
    lf_abs  =  result['fft_abs'][1]
    hf_abs  =  result['fft_abs'][2]
    
    vlf_log =  result['fft_log'][0]
    lf_log  =  result['fft_log'][1]
    hf_log  =  result['fft_log'][2]

    lf_norm  =  result['fft_norm'][0]
    hf_norm  =  result['fft_norm'][1]
    
    lf_hf_ratio = result['fft_ratio']

    f_total = result['fft_total']
    
    '''
    fft_peak (tuple): Peak frequencies of all frequency bands [Hz]
    fft_abs (tuple): Absolute powers of all frequency bands [ms^2]
    fft_rel (tuple): Relative powers of all frequency bands [%]
    fft_log (tuple): Logarithmic powers of all frequency bands [log]
    fft_norm (tuple): Normalized powers of the LF and HF frequency bands [-]
    fft_ratio (float): LF/HF ratio [-]
    fft_total (float): Total power over all frequency bands [ms^2]
    fft_interpolation (str): Interpolation method used for NNI interpolation (hard-coded to ‘cubic’)
    fft_resampling_frequency (int): Resampling frequency used for NNI interpolation [Hz] (hard-coded to 4Hz as recommended by the HRV Guidelines)
    fft_window (str): Spectral window used for PSD estimation of the Welch’s method
    fft_plot (matplotlib figure object): PSD plot figure object
    '''

    heart_rate_variability_results_str = (str(0) + " ; " + str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " + str(0) + " ; " +
                                          str(0) + " ; " + str(0) + " ; " + str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " + str(0) + " ; " + 
                                          str(0) + " ; " + str(0) + " ; " +  str(0) + "\n")
    for index, data_element in enumerate(rr_intervals):
        heart_rate_variability_results_str += (str(data_element) + " ; " + str(rr_min) + " ; " +  str(rr_mean) + " ; " +  str(rr_max) + " ; " +  str(sdsd) + " ; " +  str(sd1) + " ; " +  str(sd2) + " ; " +  str(sdnn) + " ; " +  str(rmssd) + " ; " + str(sd1_over_sd2_ratio) + " ; " + 
                                               str(s_ellipse_area) + " ; " + str(vlf_peak) + " ; " + str(lf_peak) + " ; " +  str(hf_peak) + " ; " +  str(vlf_abs) + " ; " +  str(lf_abs) + " ; " +  str(hf_abs) + " ; " +  str(vlf_log) + " ; " +  str(lf_log) + " ; " +  str(hf_log) + " ; " + 
                                               str(lf_norm) + " ; " + str(hf_norm) + " ; " +  str(lf_hf_ratio) + " ; " +  str(f_total) + "\n")

    generatet_file_name = "Bitalinoi-Proband-Stage-0_{}_{}{}".format(cId, selected_condition.replace(' ','-'), "_ECG_HearRateVariabilityResults.txt")
    path_to_heart_rate_variability_results_file = "{}/{}".format(folder_path, generatet_file_name)
    if exists(path_to_heart_rate_variability_results_file):
        os.remove(path_to_heart_rate_variability_results_file)
    
    with open(path_to_heart_rate_variability_results_file, 'w', encoding='utf-8') as f:
        f.writelines(heart_rate_variability_results_str)

    print("Stage 0")
    print("--------------------------")
    print("rr_min: " + str(rr_min))
    print("rr_mean: " + str(rr_mean))
    print("rr_max: " + str(rr_max))
    print("--------")
    print("sdsd:" + str(sdsd))
    print("sd1: " + str(sd1))
    print("sd2: " + str(sd2))
    print("sd1/sd2: " + str(sd1_over_sd2_ratio))
    print("sdnn: " + str(sdnn))
    print("rmssd: " + str(rmssd))
    print("--------------------------")

    #results = nl.poincare(rr_intervals) # only for debug and to compare own plot and algorithemen

    # stage 1 
    #--------------------------------------------------------------------------------------------
    file_path = file_name_1
    #file_path = folder_path + file_name_1
    raw_data_file = open(file_path, 'r')

    rpeak_time_input = []

    # read file
    for line in raw_data_file:
        rpeak_time_input.append(float(line.split(";")[2])) # form s to ms
        
    # Closing the file after reading
    raw_data_file.close()

    rr_intervals = []
    for index in range(len(rpeak_time_input) - 1):
        rr_intervals.append(int(rpeak_time_input[index + 1] - rpeak_time_input[index]))

    rr_min = sys.maxsize
    rr_max = 0
    rr_mean = 0

    for index in range(len(rr_intervals)):
        if rr_min > rr_intervals[index]:
            rr_min = rr_intervals[index]
        if rr_max < rr_intervals[index]:
            rr_max = rr_intervals[index]
        rr_mean += rr_intervals[index]

    rr_mean /= (len(rr_intervals))

    rr_2d=[]

    for index in range(len(rr_intervals) - 1):
        rr_2d.append([int(rr_intervals[index]), int(rr_intervals[index + 1])])

    rr_2d_np = np.array(rr_2d)

    # Reference-Link : https://xn--hrv-herzratenvariabilitt-dcc.de/2019/01/berechnung-des-hrv-werts-sdnn/
    # sdnn calculation
    rr_intervals_sum = 0
    for index in range(len(rr_intervals)):
        if index >= 2:
            rr_intervals_sum += int(rr_intervals[index])

    rr_intervals_mean = rr_intervals_sum / (len(rr_intervals) - 1)

    rr_intervals_sum_mean = 0
    for index in range(len(rr_intervals)):
        if index >= 2:
            rr_intervals_sum_mean += (rr_intervals[index] - rr_intervals_mean)**2

    rr_intervals_sum_mean_mean = rr_intervals_sum_mean / (len(rr_intervals) - 1)
    sdnn = math.sqrt(rr_intervals_sum_mean_mean)

    # Reference-Link : https://xn--hrv-herzratenvariabilitt-dcc.de/2017/10/berechnung-des-hrv-werts-rmssd/
    # rmssd calculation
    rr_time_squered_sum_sum = 0

    for index in range(len(rr_intervals) - 1):
        expression = int(rr_intervals[index + 1] - rr_intervals[index])**2
        rr_time_squered_sum_sum += expression

    rr_time_squered_sum_sum_mean = rr_time_squered_sum_sum / (len(rr_intervals) - 1)
    rmssd = math.sqrt(rr_time_squered_sum_sum_mean)

    # Reference-Link : https://ieeexplore.ieee.org/abstract/document/959330
    # sdsd calculation
    sdsd = 0
    sdsd_sum = 0
    for index in range(len(rr_intervals) - 1):
        sdsd_sum += ( int(rr_intervals[index]) - int(rr_intervals[index + 1]) )**2

    sdsd_sum_mean = sdsd_sum / (len(rr_intervals) - 1)
    sdsd = math.sqrt(sdsd_sum_mean)

    # Reference-Link : https://onlinelibrary.wiley.com/doi/10.1002/mus.25573
    # sd1 calculation
    sd1 = 0
    sd1_sum = 0
    for index in range(len(rr_intervals) - 1):
        sd1_sum += ( ((1/math.sqrt(2)) * int(rr_intervals[index])) - ((1/math.sqrt(2)) * int(rr_intervals[index + 1])) )**2

    sd1_sum_mean = sd1_sum / (len(rr_intervals) - 1)

    sd1 = math.sqrt(sd1_sum_mean)

    # Reference-Link : https://pyhrv.readthedocs.io/en/latest/_pages/api/nonlinear.html
    # sd2 calculation
    sd2 = math.sqrt(((2 * sdnn**2) - (0.5 * sdsd**2)))

    # Reference-Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/
    # sd1 over sd2 ratio = sd1 / sd2
    sd1_over_sd2_ratio = sd1 / sd2
    
    # Reference-Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/
    # S as Ellipse erea
    s_ellipse_area = pi * sd1 * sd2

    angle =  pi / 4 # rad
    angle = (angle / np.pi) * 180 # deg
    print (angle)
    
    sd1_point = [rr_mean + (-sd1 * np.sqrt(2) / 2), rr_mean + (sd1 * np.sqrt(2) / 2)]
    rr_mean_point = [rr_mean, rr_mean]
    sd2_point = [rr_mean + (sd2) * np.cos(np.deg2rad(45)), rr_mean + (sd2) * np.sin(np.deg2rad(45))]

    # set plot parameters
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    # set plot title and axis labels
    fig.suptitle('Poincaré-Plot')
    plt.xlabel("RRi [ms]")
    plt.ylabel("RRi+1 [ms]")
    # create ellipse plot object
    ellipse = Ellipse((rr_mean, rr_mean), 2*sd2, 2*sd1, angle=angle, linewidth=1, fill=False, color="k", zorder=3)
    area = Ellipse((rr_mean, rr_mean), 2*sd2, 2*sd1, angle=angle, alpha=0.2, linewidth=1, fill=True, color="yellow", zorder=2)
    ax = plt.gca()
    ax.add_patch(ellipse)
    ax.add_patch(area)
    # set limits    
    plt.xlim([rr_min, rr_max])
    plt.ylim([rr_min, rr_max])
    # set axes identity lines
    x_values = [rr_min, rr_max]
    y_values = [rr_min, rr_max]
    plt.plot(x_values, y_values, color='green', zorder=1, linewidth=1, linestyle="--", label="x-Axe identity line")
    x_values = [rr_max, rr_min]
    y_values = [rr_min, rr_max]
    plt.plot(x_values, y_values, color='blue', zorder=1, linewidth=1, linestyle="--", label="y-Axe identity line")
    # sd1 point and vector
    plt.scatter(sd1_point[0], sd1_point[1], color='blue', zorder=4, label="SD1: " + str(sd1))
    plt.arrow(rr_mean_point[0], rr_mean_point[1], (-sd1) * np.cos(np.deg2rad(angle)), (sd1) * np.sin(np.deg2rad(angle)), color="blue", linewidth=1, zorder=3)
    # rr_intervals mean point
    plt.scatter(rr_mean_point[0], rr_mean_point[1], color='red', zorder=4, label="RR-Intervals-Mean: " + str(rr_mean))
    # sd2 point and vector
    plt.scatter(sd2_point[0], sd2_point[1], color='green', zorder=4, label="SD2: " + str(sd2))
    plt.arrow(rr_mean_point[0], rr_mean_point[1], (sd2) * np.cos(np.deg2rad(angle)), (sd2) * np.sin(np.deg2rad(angle)), color="green", linewidth=1, zorder=3)
    # rr_intervals points and lines
    plt.plot(rr_2d_np[:,0], rr_2d_np[:,1], color='gray', zorder=1)
    plt.scatter(rr_2d_np[:,0], rr_2d_np[:,1], zorder=2, label="RR-Intervals-Points")
    # Create empty plot with blank marker containing the extra label
    plt.plot([], [], ' ', label="S (Ellipse area): " + str(s_ellipse_area))
    # activate labels as legend
    plt.legend()
    # activate plot grid
    plt.grid()
    # make layout tight
    fig.tight_layout()
    # save to file
    generatet_file_name = "Bitalinoi-Proband-Stage-1_{}_{}{}".format(cId, selected_condition.replace(' ','-'), "_ECG_HearRateVariabilityResults.png")
    path_to_heart_rate_variability_fig_file = "{}/{}".format(folder_path, generatet_file_name)
    fig.savefig(path_to_heart_rate_variability_fig_file, dpi=600, bbox_inches='tight')
    # show only for debug
    #plt.show()
    plt.close()

    # Reference-Link: https://pyhrv.readthedocs.io/en/latest/_pages/api/frequency.html
    # simple solution/std (welch-fft) solution for HRV frequence domain
    # VLF: [0.00Hz - 0.04Hz]
    # LF: [0.04Hz - 0.15Hz]
    # HF: [0.15Hz - 0.40Hz]
    result = fd.welch_psd(rpeaks=rpeak_time_input,  show=False, show_param=False, legend=False)
    
    vlf_peak =  result['fft_peak'][0]
    lf_peak  =  result['fft_peak'][1]
    hf_peak  =  result['fft_peak'][2]

    vlf_abs =  result['fft_abs'][0]
    lf_abs  =  result['fft_abs'][1]
    hf_abs  =  result['fft_abs'][2]
    
    vlf_log =  result['fft_log'][0]
    lf_log  =  result['fft_log'][1]
    hf_log  =  result['fft_log'][2]

    lf_norm  =  result['fft_norm'][0]
    hf_norm  =  result['fft_norm'][1]
    
    lf_hf_ratio = result['fft_ratio']

    f_total = result['fft_total']
        
    heart_rate_variability_results_str = (str(0) + " ; " + str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " + str(0) + " ; " +
                                          str(0) + " ; " + str(0) + " ; " + str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " + str(0) + " ; " + 
                                          str(0) + " ; " + str(0) + " ; " +  str(0) + "\n")
    for index, data_element in enumerate(rr_intervals):
        heart_rate_variability_results_str += (str(data_element) + " ; " + str(rr_min) + " ; " +  str(rr_mean) + " ; " +  str(rr_max) + " ; " +  str(sdsd) + " ; " +  str(sd1) + " ; " +  str(sd2) + " ; " +  str(sdnn) + " ; " +  str(rmssd) + " ; " + str(sd1_over_sd2_ratio) + " ; " + 
                                               str(s_ellipse_area) + " ; " + str(vlf_peak) + " ; " + str(lf_peak) + " ; " +  str(hf_peak) + " ; " +  str(vlf_abs) + " ; " +  str(lf_abs) + " ; " +  str(hf_abs) + " ; " +  str(vlf_log) + " ; " +  str(lf_log) + " ; " +  str(hf_log) + " ; " + 
                                               str(lf_norm) + " ; " + str(hf_norm) + " ; " +  str(lf_hf_ratio) + " ; " +  str(f_total) + "\n")

    generatet_file_name = "Bitalinoi-Proband-Stage-1_{}_{}{}".format(cId, selected_condition.replace(' ','-'), "_ECG_HearRateVariabilityResults.txt")
    path_to_heart_rate_variability_results_file = "{}/{}".format(folder_path, generatet_file_name)
    if exists(path_to_heart_rate_variability_results_file):
        os.remove(path_to_heart_rate_variability_results_file)
    
    with open(path_to_heart_rate_variability_results_file, 'w', encoding='utf-8') as f:
        f.writelines(heart_rate_variability_results_str)

    print("Stage 1")
    print("--------------------------")
    print("rr_min: " + str(rr_min))
    print("rr_mean: " + str(rr_mean))
    print("rr_max: " + str(rr_max))
    print("--------")
    print("sdsd:" + str(sdsd))
    print("sd1: " + str(sd1))
    print("sd2: " + str(sd2))
    print("sd1/sd2: " + str(sd1_over_sd2_ratio))
    print("sdnn: " + str(sdnn))
    print("rmssd: " + str(rmssd))
    print("--------------------------")

    # stage 2 
    #--------------------------------------------------------------------------------------------
    file_path = file_name_2
    #file_path = folder_path + file_name_2
    raw_data_file = open(file_path, 'r')

    rpeak_time_input = []

    # read file
    for line in raw_data_file:
        rpeak_time_input.append(float(line.split(";")[2])) # form s to ms
        
    # Closing the file after reading
    raw_data_file.close()

    rr_intervals = []
    for index in range(len(rpeak_time_input) - 1):
        rr_intervals.append(int(rpeak_time_input[index + 1] - rpeak_time_input[index]))

    rr_min = sys.maxsize
    rr_max = 0
    rr_mean = 0

    for index in range(len(rr_intervals)):
        if rr_min > rr_intervals[index]:
            rr_min = rr_intervals[index]
        if rr_max < rr_intervals[index]:
            rr_max = rr_intervals[index]
        rr_mean += rr_intervals[index]

    rr_mean /= (len(rr_intervals))

    rr_2d=[]

    for index in range(len(rr_intervals) - 1):
        rr_2d.append([int(rr_intervals[index]), int(rr_intervals[index + 1])])

    rr_2d_np = np.array(rr_2d)

    # Reference-Link : https://xn--hrv-herzratenvariabilitt-dcc.de/2019/01/berechnung-des-hrv-werts-sdnn/
    # sdnn calculation
    rr_intervals_sum = 0
    for index in range(len(rr_intervals)):
        if index >= 2:
            rr_intervals_sum += int(rr_intervals[index])

    rr_intervals_mean = rr_intervals_sum / (len(rr_intervals) - 1)

    rr_intervals_sum_mean = 0
    for index in range(len(rr_intervals)):
        if index >= 2:
            rr_intervals_sum_mean += (rr_intervals[index] - rr_intervals_mean)**2

    rr_intervals_sum_mean_mean = rr_intervals_sum_mean / (len(rr_intervals) - 1)
    sdnn = math.sqrt(rr_intervals_sum_mean_mean)

    # Reference-Link : https://xn--hrv-herzratenvariabilitt-dcc.de/2017/10/berechnung-des-hrv-werts-rmssd/
    # rmssd calculation
    rr_time_squered_sum_sum = 0

    for index in range(len(rr_intervals) - 1):
        expression = int(rr_intervals[index + 1] - rr_intervals[index])**2
        rr_time_squered_sum_sum += expression

    rr_time_squered_sum_sum_mean = rr_time_squered_sum_sum / (len(rr_intervals) - 1)
    rmssd = math.sqrt(rr_time_squered_sum_sum_mean)

    # Reference-Link : https://ieeexplore.ieee.org/abstract/document/959330
    # sdsd calculation
    sdsd = 0
    sdsd_sum = 0
    for index in range(len(rr_intervals) - 1):
        sdsd_sum += ( int(rr_intervals[index]) - int(rr_intervals[index + 1]) )**2

    sdsd_sum_mean = sdsd_sum / (len(rr_intervals) - 1)
    sdsd = math.sqrt(sdsd_sum_mean)

    # Reference-Link : https://onlinelibrary.wiley.com/doi/10.1002/mus.25573
    # sd1 calculation
    sd1 = 0
    sd1_sum = 0
    for index in range(len(rr_intervals) - 1):
        sd1_sum += ( ((1/math.sqrt(2)) * int(rr_intervals[index])) - ((1/math.sqrt(2)) * int(rr_intervals[index + 1])) )**2

    sd1_sum_mean = sd1_sum / (len(rr_intervals) - 1)

    sd1 = math.sqrt(sd1_sum_mean)

    # Reference-Link : https://pyhrv.readthedocs.io/en/latest/_pages/api/nonlinear.html
    # sd2 calculation
    sd2 = math.sqrt(((2 * sdnn**2) - (0.5 * sdsd**2)))

    # Reference-Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/
    # sd1 over sd2 ratio = sd1 / sd2
    sd1_over_sd2_ratio = sd1 / sd2
    
    # Reference-Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5624990/
    # S as Ellipse erea
    s_ellipse_area = pi * sd1 * sd2

    angle =  pi / 4 # rad
    angle = (angle / np.pi) * 180 # deg
    print (angle)
    
    sd1_point = [rr_mean + (-sd1 * np.sqrt(2) / 2), rr_mean + (sd1 * np.sqrt(2) / 2)]
    rr_mean_point = [rr_mean, rr_mean]
    sd2_point = [rr_mean + (sd2) * np.cos(np.deg2rad(45)), rr_mean + (sd2) * np.sin(np.deg2rad(45))]

    # set plot parameters
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    # set plot title and axis labels
    fig.suptitle('Poincaré-Plot')
    plt.xlabel("RRi [ms]")
    plt.ylabel("RRi+1 [ms]")
    # create ellipse plot object
    ellipse = Ellipse((rr_mean, rr_mean), 2*sd2, 2*sd1, angle=angle, linewidth=1, fill=False, color="k", zorder=3)
    area = Ellipse((rr_mean, rr_mean), 2*sd2, 2*sd1, angle=angle, alpha=0.2, linewidth=1, fill=True, color="yellow", zorder=2)
    ax = plt.gca()
    ax.add_patch(ellipse)
    ax.add_patch(area)
    # set limits    
    plt.xlim([rr_min, rr_max])
    plt.ylim([rr_min, rr_max])
    # set axes identity lines
    x_values = [rr_min, rr_max]
    y_values = [rr_min, rr_max]
    plt.plot(x_values, y_values, color='green', zorder=1, linewidth=1, linestyle="--", label="x-Axe identity line")
    x_values = [rr_max, rr_min]
    y_values = [rr_min, rr_max]
    plt.plot(x_values, y_values, color='blue', zorder=1, linewidth=1, linestyle="--", label="y-Axe identity line")
    # sd1 point and vector
    plt.scatter(sd1_point[0], sd1_point[1], color='blue', zorder=4, label="SD1: " + str(sd1))
    plt.arrow(rr_mean_point[0], rr_mean_point[1], (-sd1) * np.cos(np.deg2rad(angle)), (sd1) * np.sin(np.deg2rad(angle)), color="blue", linewidth=1, zorder=3)
    # rr_intervals mean point
    plt.scatter(rr_mean_point[0], rr_mean_point[1], color='red', zorder=4, label="RR-Intervals-Mean: " + str(rr_mean))
    # sd2 point and vector
    plt.scatter(sd2_point[0], sd2_point[1], color='green', zorder=4, label="SD2: " + str(sd2))
    plt.arrow(rr_mean_point[0], rr_mean_point[1], (sd2) * np.cos(np.deg2rad(angle)), (sd2) * np.sin(np.deg2rad(angle)), color="green", linewidth=1, zorder=3)
    # rr_intervals points and lines
    plt.plot(rr_2d_np[:,0], rr_2d_np[:,1], color='gray', zorder=1)
    plt.scatter(rr_2d_np[:,0], rr_2d_np[:,1], zorder=2, label="RR-Intervals-Points")
    # Create empty plot with blank marker containing the extra label
    plt.plot([], [], ' ', label="S (Ellipse area): " + str(s_ellipse_area))
    # activate labels as legend
    plt.legend()
    # activate plot grid
    plt.grid()
    # make layout tight
    fig.tight_layout()
    # save to file
    generatet_file_name = "Bitalinoi-Proband-Stage-2_{}_{}{}".format(cId, selected_condition.replace(' ','-'), "_ECG_HearRateVariabilityResults.png")
    path_to_heart_rate_variability_fig_file = "{}/{}".format(folder_path, generatet_file_name)
    fig.savefig(path_to_heart_rate_variability_fig_file, dpi=600, bbox_inches='tight')
    # show only for debug
    #plt.show()
    plt.close()

    # Reference-Link: https://pyhrv.readthedocs.io/en/latest/_pages/api/frequency.html
    # simple solution/std (welch-fft) solution for HRV frequence domain
    # VLF: [0.00Hz - 0.04Hz]
    # LF: [0.04Hz - 0.15Hz]
    # HF: [0.15Hz - 0.40Hz]
    result = fd.welch_psd(rpeaks=rpeak_time_input,  show=False, show_param=False, legend=False)
    
    vlf_peak =  result['fft_peak'][0]
    lf_peak  =  result['fft_peak'][1]
    hf_peak  =  result['fft_peak'][2]

    vlf_abs =  result['fft_abs'][0]
    lf_abs  =  result['fft_abs'][1]
    hf_abs  =  result['fft_abs'][2]
    
    vlf_log =  result['fft_log'][0]
    lf_log  =  result['fft_log'][1]
    hf_log  =  result['fft_log'][2]

    lf_norm  =  result['fft_norm'][0]
    hf_norm  =  result['fft_norm'][1]
    
    lf_hf_ratio = result['fft_ratio']

    f_total = result['fft_total']
        
    heart_rate_variability_results_str = (str(0) + " ; " + str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " + str(0) + " ; " +
                                          str(0) + " ; " + str(0) + " ; " + str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " +  str(0) + " ; " + str(0) + " ; " + 
                                          str(0) + " ; " + str(0) + " ; " +  str(0) + "\n")
    for index, data_element in enumerate(rr_intervals):
        heart_rate_variability_results_str += (str(data_element) + " ; " + str(rr_min) + " ; " +  str(rr_mean) + " ; " +  str(rr_max) + " ; " +  str(sdsd) + " ; " +  str(sd1) + " ; " +  str(sd2) + " ; " +  str(sdnn) + " ; " +  str(rmssd) + " ; " + str(sd1_over_sd2_ratio) + " ; " + 
                                               str(s_ellipse_area) + " ; " + str(vlf_peak) + " ; " + str(lf_peak) + " ; " +  str(hf_peak) + " ; " +  str(vlf_abs) + " ; " +  str(lf_abs) + " ; " +  str(hf_abs) + " ; " +  str(vlf_log) + " ; " +  str(lf_log) + " ; " +  str(hf_log) + " ; " + 
                                               str(lf_norm) + " ; " + str(hf_norm) + " ; " +  str(lf_hf_ratio) + " ; " +  str(f_total) + "\n")

    generatet_file_name = "Bitalinoi-Proband-Stage-2_{}_{}{}".format(cId, selected_condition.replace(' ','-'), "_ECG_HearRateVariabilityResults.txt")
    path_to_heart_rate_variability_results_file = "{}/{}".format(folder_path, generatet_file_name)
    if exists(path_to_heart_rate_variability_results_file):
        os.remove(path_to_heart_rate_variability_results_file)
    
    with open(path_to_heart_rate_variability_results_file, 'w', encoding='utf-8') as f:
        f.writelines(heart_rate_variability_results_str)

    print("Stage 2")
    print("--------------------------")
    print("rr_min: " + str(rr_min))
    print("rr_mean: " + str(rr_mean))
    print("rr_max: " + str(rr_max))
    print("--------")
    print("sdsd:" + str(sdsd))
    print("sd1: " + str(sd1))
    print("sd2: " + str(sd2))
    print("sd1/sd2: " + str(sd1_over_sd2_ratio))
    print("sdnn: " + str(sdnn))
    print("rmssd: " + str(rmssd))
    print("--------------------------")
# Qelle: https://xn--hrv-herzratenvariabilitt-dcc.de/2019/01/berechnung-des-hrv-werts-sdnn/
# Standard von: https://www.ahajournals.org/doi/10.1161/01.CIR.93.5.1043
# de: Quadratwurzel des Mittelwerts der Summe aller quadrierten Differenzen zwischen benachbarten RR-Intervallen
# en: square root of the mean of the sum of the squares of differences between adjacent NN intervals
#
# 
# Path = "./Input/Bitalinoi-Proband-Stage-0_id-17b-Condition-B_ECG_HearRateResults.txt"

import os
import math


file_name_0 = "Bitalinoi-Proband-Stage-0-id-1-Condition-A-ECG_HearRateResults.txt"
file_name_1 = "Bitalinoi-Proband-Stage-1-id-1-Condition-A-ECG_HearRateResults.txt"
file_name_2 = "Bitalinoi-Proband-Stage-2-id-1-Condition-A-ECG_HearRateResults.txt"
#file_name_3 = "RPeaksResults.txt"
folder_path = "./Input/"

# stage 0
file_path = folder_path + file_name_0
raw_data_file = open(file_path, 'r')

rpeak_time_input = []

# read file
for line in raw_data_file:
    rpeak_time_input.append(float(line.split(";")[2])) # form s to ms
    
# Closing the file after reading
raw_data_file.close()

#print(rpeak_time_input)

rr_intervals = []
for index in range(len(rpeak_time_input) - 1):
    rr_intervals.append(int(rpeak_time_input[index + 1] - rpeak_time_input[index]))

rr_intervals_sum = 0
for index in range(len(rr_intervals)):
    if index >= 2:
        rr_intervals_sum += int(rr_intervals[index])

#print(rr_intervals_sum)

rr_intervals_mean = rr_intervals_sum / (len(rr_intervals) - 1)

#print(rr_intervals_mean)

rr_intervals_sum_mean = 0
for index in range(len(rr_intervals)):
    if index >= 2:
        rr_intervals_sum_mean += (rr_intervals[index] - rr_intervals_mean)**2

#print(rr_intervals_sum_mean)

rr_intervals_sum_mean_mean = rr_intervals_sum_mean / (len(rr_intervals) - 1)

#print(rr_intervals_sum_mean_mean)

rr_intervals_squered_sum_mean_mean_root = math.sqrt(rr_intervals_sum_mean_mean)

print("Stage 0")
print(str(rr_intervals_squered_sum_mean_mean_root) + " ms")
print(str(rr_intervals_squered_sum_mean_mean_root / 1000) + " s")

# stage 1 

file_path = folder_path + file_name_1
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

rr_intervals_sum = 0
for index in range(len(rr_intervals)):
    if index >= 2:
        rr_intervals_sum += int(rr_intervals[index])

#print(rr_intervals_sum)

rr_intervals_mean = rr_intervals_sum / (len(rr_intervals) - 1)

#print(rr_intervals_mean)

rr_intervals_sum_mean = 0
for index in range(len(rr_intervals)):
    if index >= 2:
        rr_intervals_sum_mean += (rr_intervals[index] - rr_intervals_mean)**2

#print(rr_intervals_sum_mean)

rr_intervals_sum_mean_mean = rr_intervals_sum_mean / (len(rr_intervals) - 1)

#print(rr_intervals_sum_mean_mean)

rr_intervals_squered_sum_mean_mean_root = math.sqrt(rr_intervals_sum_mean_mean)

print("Stage 1")
print(str(rr_intervals_squered_sum_mean_mean_root) + " ms")
print(str(rr_intervals_squered_sum_mean_mean_root / 1000) + " s")


# stage 2 

file_path = folder_path + file_name_2
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

rr_intervals_sum = 0
for index in range(len(rr_intervals)):
    if index >= 2:
        rr_intervals_sum += int(rr_intervals[index])

#print(rr_intervals_sum)

rr_intervals_mean = rr_intervals_sum / (len(rr_intervals) - 1)

#print(rr_intervals_mean)

rr_intervals_sum_mean = 0
for index in range(len(rr_intervals)):
    if index >= 2:
        rr_intervals_sum_mean += (rr_intervals[index] - rr_intervals_mean)**2

#print(rr_intervals_sum_mean)

rr_intervals_sum_mean_mean = rr_intervals_sum_mean / (len(rr_intervals) - 1)

#print(rr_intervals_sum_mean_mean)

rr_intervals_squered_sum_mean_mean_root = math.sqrt(rr_intervals_sum_mean_mean)

print("Stage 2")
print(str(rr_intervals_squered_sum_mean_mean_root) + " ms")
print(str(rr_intervals_squered_sum_mean_mean_root / 1000) + " s")

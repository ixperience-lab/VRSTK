# Qelle: https://xn--hrv-herzratenvariabilitt-dcc.de/2017/10/berechnung-des-hrv-werts-rmssd/
# Standard von: https://www.ahajournals.org/doi/10.1161/01.CIR.93.5.1043
# de: Quadratwurzel des Mittelwerts der Summe aller quadrierten Differenzen zwischen benachbarten RR-Intervallen
# en: square root of the mean of the sum of the squares of differences between adjacent NN intervals
#
# 
# Path = "./Input/Bitalinoi-Proband-Stage-0_id-17b-Condition-B_ECG_HearRateResults.txt"

import os
import math


file_name_0 = "Bitalinoi-Proband-Stage-0_id-18-Condition-B_ECG_HearRateResults.txt"
file_name_1 = "Bitalinoi-Proband-Stage-1_id-18-Condition-B_ECG_HearRateResults.txt"
file_name_2 = "Bitalinoi-Proband-Stage-2_id-18-Condition-B_ECG_HearRateResults.txt"
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

rr_Intervals = []
for index in range(len(rpeak_time_input) - 1):
    rr_Intervals.append(int(rpeak_time_input[index + 1] - rpeak_time_input[index]))

rr_time_squered_sum_sum = 0

for index in range(len(rr_Intervals) - 1):
    expression = int(rr_Intervals[index + 1] - rr_Intervals[index])**2
    rr_time_squered_sum_sum += expression

#print(rr_time_squered_sum_sum)

rr_time_squered_sum_sum_mean = rr_time_squered_sum_sum / (len(rr_Intervals) - 1)

#print(rr_time_squered_sum_sum_mean)

rr_time_squered_sum_sum_mean_root = math.sqrt(rr_time_squered_sum_sum_mean)

print("Stage 0")
print(str(rr_time_squered_sum_sum_mean_root) + " ms")
print(str(rr_time_squered_sum_sum_mean_root / 1000) + " s")

# stage 1 

file_path = folder_path + file_name_1
raw_data_file = open(file_path, 'r')

rpeak_time_input = []

# read file
for line in raw_data_file:
    rpeak_time_input.append(float(line.split(";")[2])) # form s to ms
    
# Closing the file after reading
raw_data_file.close()

rr_Intervals = []
for index in range(len(rpeak_time_input) - 1):
    rr_Intervals.append(int(rpeak_time_input[index + 1] - rpeak_time_input[index]))

rr_time_squered_sum_sum = 0

for index in range(len(rr_Intervals) - 1):
    expression = int(rr_Intervals[index + 1] - rr_Intervals[index])**2
    rr_time_squered_sum_sum += expression

rr_time_squered_sum_sum_mean = rr_time_squered_sum_sum / (len(rr_Intervals) - 1)

rr_time_squered_sum_sum_mean_root = math.sqrt(rr_time_squered_sum_sum_mean)

print("Stage 1")
print(str(rr_time_squered_sum_sum_mean_root) + " ms")
print(str(rr_time_squered_sum_sum_mean_root / 1000) + " s")

# stage 2 

file_path = folder_path + file_name_2
raw_data_file = open(file_path, 'r')

rpeak_time_input = []

# read file
for line in raw_data_file:
    rpeak_time_input.append(float(line.split(";")[2])) # form s to ms
    
# Closing the file after reading
raw_data_file.close()

rr_Intervals = []
for index in range(len(rpeak_time_input) - 1):
    rr_Intervals.append(int(rpeak_time_input[index + 1] - rpeak_time_input[index]))

rr_time_squered_sum_sum = 0

for index in range(len(rr_Intervals) - 1):
    expression = int(rr_Intervals[index + 1] - rr_Intervals[index])**2
    rr_time_squered_sum_sum += expression

rr_time_squered_sum_sum_mean = rr_time_squered_sum_sum / (len(rr_Intervals) - 1)

rr_time_squered_sum_sum_mean_root = math.sqrt(rr_time_squered_sum_sum_mean)

print("Stage 2")
print(str(rr_time_squered_sum_sum_mean_root) + " ms")
print(str(rr_time_squered_sum_sum_mean_root / 1000) + " s")
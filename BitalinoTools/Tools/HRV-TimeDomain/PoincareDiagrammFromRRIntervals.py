# Qelle: https://xn--hrv-herzratenvariabilitt-dcc.de/2017/07/berechnung-des-poincare-diagramms-aus-rr-intervallen/
# Standard von: https://www.ahajournals.org/doi/10.1161/01.CIR.93.5.1043
# de: Quadratwurzel des Mittelwerts der Summe aller quadrierten Differenzen zwischen benachbarten RR-Intervallen
# en: square root of the mean of the sum of the squares of differences between adjacent NN intervals
#
# 
# Path = "./Input/Bitalinoi-Proband-Stage-0_id-17b-Condition-B_ECG_HearRateResults.txt"

import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np

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


plt.title("Poincaré-Plot")
plt.xlabel("RRi [ms]")
plt.ylabel("RRi+1 [ms]")

plt.plot(rr_2d_np[:,0], rr_2d_np[:,1], color='gray', zorder=1)
plt.scatter(rr_mean, rr_mean, color='red', zorder=3) 
plt.scatter(rr_2d_np[:,0], rr_2d_np[:,1], zorder=2)
plt.show()

#x = []
#y = []
#for index in range(len(rr_2d)):
#    x.append(rr_2d[index][0])
#    y.append(rr_2d[index][1])

print("Stage 0")
print(rr_min)
print(rr_mean)
print(rr_max)

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


plt.title("Poincaré-Plot")
plt.xlabel("RRi [ms]")
plt.ylabel("RRi+1 [ms]")

plt.plot(rr_2d_np[:,0], rr_2d_np[:,1], color='gray', zorder=1)
plt.scatter(rr_mean, rr_mean, color='red', zorder=3) 
plt.scatter(rr_2d_np[:,0], rr_2d_np[:,1], zorder=2)
plt.show()

#x = []
#y = []
#for index in range(len(rr_2d)):
#    x.append(rr_2d[index][0])
#    y.append(rr_2d[index][1])

print("Stage 1")
print(rr_min)
print(rr_mean)
print(rr_max)

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

plt.title("Poincaré-Plot")
plt.xlabel("RRi [ms]")
plt.ylabel("RRi+1 [ms]")

plt.plot(rr_2d_np[:,0], rr_2d_np[:,1], color='gray', zorder=1)
plt.scatter(rr_mean, rr_mean, color='red', zorder=3) 
plt.scatter(rr_2d_np[:,0], rr_2d_np[:,1], zorder=2)
plt.show()

#x = []
#y = []
#for index in range(len(rr_2d)):
#    x.append(rr_2d[index][0])
#    y.append(rr_2d[index][1])

print("Stage 2")
print(rr_min)
print(rr_mean)
print(rr_max)

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

file_name_0 = "Bitalinoi-Proband-Stage-0_id-18-Condition-B_ECG_HearRateResults.txt"
file_name_1 = "Bitalinoi-Proband-Stage-1_id-18-Condition-B_ECG_HearRateResults.txt"
file_name_2 = "Bitalinoi-Proband-Stage-2_id-18-Condition-B_ECG_HearRateResults.txt"
folder_path = "./Input/"

# stage 0
#--------------------------------------------------------------------------------------------
file_path = folder_path + file_name_0
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

# Quelle : https://xn--hrv-herzratenvariabilitt-dcc.de/2019/01/berechnung-des-hrv-werts-sdnn/
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

# Quelle : https://xn--hrv-herzratenvariabilitt-dcc.de/2017/10/berechnung-des-hrv-werts-rmssd/
# rmssd calculation
rr_time_squered_sum_sum = 0

for index in range(len(rr_intervals) - 1):
    expression = int(rr_intervals[index + 1] - rr_intervals[index])**2
    rr_time_squered_sum_sum += expression

rr_time_squered_sum_sum_mean = rr_time_squered_sum_sum / (len(rr_intervals) - 1)
rmssd = math.sqrt(rr_time_squered_sum_sum_mean)

# Quelle : https://ieeexplore.ieee.org/abstract/document/959330
# sdsd calculation
sdsd = 0
sdsd_sum = 0
for index in range(len(rr_intervals) - 1):
    sdsd_sum += ( int(rr_intervals[index]) - int(rr_intervals[index + 1]) )**2

sdsd_sum_mean = sdsd_sum / (len(rr_intervals) - 1)
sdsd = math.sqrt(sdsd_sum_mean)

# Quelle : https://onlinelibrary.wiley.com/doi/10.1002/mus.25573
# sd1 calculation
sd1 = 0
sd1_sum = 0
for index in range(len(rr_intervals) - 1):
    sd1_sum += ( ((1/math.sqrt(2)) * int(rr_intervals[index])) - ((1/math.sqrt(2)) * int(rr_intervals[index + 1])) )**2

sd1_sum_mean = sd1_sum / (len(rr_intervals) - 1)

sd1 = math.sqrt(sd1_sum_mean)

# Quelle : https://pyhrv.readthedocs.io/en/latest/_pages/api/nonlinear.html
# sd2 calculation
sd2 = math.sqrt(((2 * sdnn**2) - (0.5 * sdsd**2)))

plt.title("Poincaré-Plot")
plt.xlabel("RRi [ms]")
plt.ylabel("RRi+1 [ms]")

sd1_point = [rr_mean - sd1, rr_mean + sd1]
rr_mean_point = [rr_mean, rr_mean]
sd2_point = [rr_mean + sd2, rr_mean + sd2]

sd1_line_np = np.array([rr_mean_point, sd1_point])
sd2_line_np = np.array([rr_mean_point, sd2_point])

plt.plot(rr_2d_np[:,0], rr_2d_np[:,1], color='gray', zorder=1)
plt.scatter(sd1_point[0], sd1_point[1], color='red', zorder=3)
plt.plot(sd1_line_np[:,0], sd1_line_np[:,1], color='black', zorder=2)
plt.scatter(rr_mean_point[0], rr_mean_point[1], color='red', zorder=3)
plt.plot(sd2_line_np[:,0], sd2_line_np[:,1], color='black', zorder=2)
plt.scatter(sd2_point[0], sd2_point[1], color='red', zorder=3)
plt.scatter(rr_2d_np[:,0], rr_2d_np[:,1], zorder=2)
plt.show()

#x = []
#y = []
#for index in range(len(rr_2d)):
#    x.append(rr_2d[index][0])
#    y.append(rr_2d[index][1])

print("Stage 0")
print("--------------------------")
print("rr_min: " + str(rr_min))
print("rr_mean: " + str(rr_mean))
print("rr_max: " + str(rr_max))
print("--------")
print("sdsd:" + str(sdsd))
print("sd1: " + str(sd1))
print("sd2: " + str(sd2))
print("sdnn:" + str(sdnn))
print("rmssd: " + str(rmssd))
print("--------------------------")

# stage 1 
#--------------------------------------------------------------------------------------------
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

# Quelle : https://xn--hrv-herzratenvariabilitt-dcc.de/2019/01/berechnung-des-hrv-werts-sdnn/
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

# Quelle : https://xn--hrv-herzratenvariabilitt-dcc.de/2017/10/berechnung-des-hrv-werts-rmssd/
# rmssd calculation
rr_time_squered_sum_sum = 0

for index in range(len(rr_intervals) - 1):
    expression = int(rr_intervals[index + 1] - rr_intervals[index])**2
    rr_time_squered_sum_sum += expression

rr_time_squered_sum_sum_mean = rr_time_squered_sum_sum / (len(rr_intervals) - 1)
rmssd = math.sqrt(rr_time_squered_sum_sum_mean)

# Quelle : https://ieeexplore.ieee.org/abstract/document/959330
# sdsd calculation
sdsd = 0
sdsd_sum = 0
for index in range(len(rr_intervals) - 1):
    sdsd_sum += ( int(rr_intervals[index]) - int(rr_intervals[index + 1]) )**2

sdsd_sum_mean = sdsd_sum / (len(rr_intervals) - 1)
sdsd = math.sqrt(sdsd_sum_mean)

# Quelle : https://onlinelibrary.wiley.com/doi/10.1002/mus.25573
# sd1 calculation
sd1 = 0
sd1_sum = 0
for index in range(len(rr_intervals) - 1):
    sd1_sum += ( ((1/math.sqrt(2)) * int(rr_intervals[index])) - ((1/math.sqrt(2)) * int(rr_intervals[index + 1])) )**2

sd1_sum_mean = sd1_sum / (len(rr_intervals) - 1)

sd1 = math.sqrt(sd1_sum_mean)

# Quelle : https://pyhrv.readthedocs.io/en/latest/_pages/api/nonlinear.html
# sd2 calculation
sd2 = math.sqrt(((2 * sdnn**2) - (0.5 * sdsd**2)))


plt.title("Poincaré-Plot")
plt.xlabel("RRi [ms]")
plt.ylabel("RRi+1 [ms]")

sd1_point = [rr_mean - sd1, rr_mean + sd1]
rr_mean_point = [rr_mean, rr_mean]
sd2_point = [rr_mean + sd2, rr_mean + sd2]

sd1_line_np = np.array([rr_mean_point, sd1_point])
sd2_line_np = np.array([rr_mean_point, sd2_point])

plt.plot(rr_2d_np[:,0], rr_2d_np[:,1], color='gray', zorder=1)
plt.scatter(sd1_point[0], sd1_point[1], color='red', zorder=3)
plt.plot(sd1_line_np[:,0], sd1_line_np[:,1], color='black', zorder=2)
plt.scatter(rr_mean_point[0], rr_mean_point[1], color='red', zorder=3)
plt.plot(sd2_line_np[:,0], sd2_line_np[:,1], color='black', zorder=2)
plt.scatter(sd2_point[0], sd2_point[1], color='red', zorder=3)
plt.scatter(rr_2d_np[:,0], rr_2d_np[:,1], zorder=2)
plt.show()

print("Stage 1")
print("--------------------------")
print("rr_min: " + str(rr_min))
print("rr_mean: " + str(rr_mean))
print("rr_max: " + str(rr_max))
print("--------")
print("sdsd:" + str(sdsd))
print("sd1: " + str(sd1))
print("sd2: " + str(sd2))
print("sdnn:" + str(sdnn))
print("rmssd: " + str(rmssd))
print("--------------------------")

# stage 2 
#--------------------------------------------------------------------------------------------
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

# Quelle : https://xn--hrv-herzratenvariabilitt-dcc.de/2019/01/berechnung-des-hrv-werts-sdnn/
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

# Quelle : https://xn--hrv-herzratenvariabilitt-dcc.de/2017/10/berechnung-des-hrv-werts-rmssd/
# rmssd calculation
rr_time_squered_sum_sum = 0

for index in range(len(rr_intervals) - 1):
    expression = int(rr_intervals[index + 1] - rr_intervals[index])**2
    rr_time_squered_sum_sum += expression

rr_time_squered_sum_sum_mean = rr_time_squered_sum_sum / (len(rr_intervals) - 1)
rmssd = math.sqrt(rr_time_squered_sum_sum_mean)

# Quelle : https://ieeexplore.ieee.org/abstract/document/959330
# sdsd calculation
sdsd = 0
sdsd_sum = 0
for index in range(len(rr_intervals) - 1):
    sdsd_sum += ( int(rr_intervals[index]) - int(rr_intervals[index + 1]) )**2

sdsd_sum_mean = sdsd_sum / (len(rr_intervals) - 1)
sdsd = math.sqrt(sdsd_sum_mean)

# Quelle : https://onlinelibrary.wiley.com/doi/10.1002/mus.25573
# sd1 calculation
sd1 = 0
sd1_sum = 0
for index in range(len(rr_intervals) - 1):
    sd1_sum += ( ((1/math.sqrt(2)) * int(rr_intervals[index])) - ((1/math.sqrt(2)) * int(rr_intervals[index + 1])) )**2

sd1_sum_mean = sd1_sum / (len(rr_intervals) - 1)

sd1 = math.sqrt(sd1_sum_mean)

# Quelle : https://pyhrv.readthedocs.io/en/latest/_pages/api/nonlinear.html
# sd2 calculation
sd2 = math.sqrt(((2 * sdnn**2) - (0.5 * sdsd**2)))


plt.title("Poincaré-Plot")
plt.xlabel("RRi [ms]")
plt.ylabel("RRi+1 [ms]")

sd1_point = [rr_mean - sd1, rr_mean + sd1]
rr_mean_point = [rr_mean, rr_mean]
sd2_point = [rr_mean + sd2, rr_mean + sd2]

sd1_line_np = np.array([rr_mean_point, sd1_point])
sd2_line_np = np.array([rr_mean_point, sd2_point])

plt.plot(rr_2d_np[:,0], rr_2d_np[:,1], color='gray', zorder=1)
plt.scatter(sd1_point[0], sd1_point[1], color='red', zorder=3)
plt.plot(sd1_line_np[:,0], sd1_line_np[:,1], color='black', zorder=2)
plt.scatter(rr_mean_point[0], rr_mean_point[1], color='red', zorder=3)
plt.plot(sd2_line_np[:,0], sd2_line_np[:,1], color='black', zorder=2)
plt.scatter(sd2_point[0], sd2_point[1], color='red', zorder=3)
plt.scatter(rr_2d_np[:,0], rr_2d_np[:,1], zorder=2)
plt.show()

print("Stage 2")
print("--------------------------")
print("rr_min: " + str(rr_min))
print("rr_mean: " + str(rr_mean))
print("rr_max: " + str(rr_max))
print("--------")
print("sdsd:" + str(sdsd))
print("sd1: " + str(sd1))
print("sd2: " + str(sd2))
print("sdnn:" + str(sdnn))
print("rmssd: " + str(rmssd))
print("--------------------------")


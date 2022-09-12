import pandas as pd
import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np


# AllUncannyValleyConditionStatisticResults_DataFrame
# ---------------------------------------------------
# Perceived human likeness
# -----------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('AllUncannyValleyConditionStatisticResults_DataFrame.csv', sep=";")

avgArrayOfAllModelsHumanLikeness=[]
nameArrayOfAllModelsHumanLikeness=[]

for index, row in df.iterrows():
    #print(index)
    if (index % 3) == 0:
        avgArrayOfAllModelsHumanLikeness.append(float(str(row[3]).replace(",", ".")))
        nameArrayOfAllModelsHumanLikeness.append(str(row[0]).replace("q1_", ""))

# sorting the list:
# Eyebot, Turret, JRRobo, Lloyd, Atlas, Ribbot, Katie, Alice, Freddy, Medic, Link, Dutchess, Zombie, MixamoGirl, Remy

tempArrayAvg = []
tempArrayNames = []

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[12])
tempArrayNames.append("Eyebot")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[11])
tempArrayNames.append("Turret")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[4])
tempArrayNames.append("JRRobo")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[6])
tempArrayNames.append("Lloyd")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[15])
tempArrayNames.append("Atlas")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[13])
tempArrayNames.append("Ribbot")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[10])
tempArrayNames.append("Katie")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[9])
tempArrayNames.append("Alice")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[0])
tempArrayNames.append("Freddy")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[7])
tempArrayNames.append("Medic")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[1])
tempArrayNames.append("Link")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[8])
tempArrayNames.append("Dutchess")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[3])
tempArrayNames.append("Zombie")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[14])
tempArrayNames.append("MixamoGirl")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[5])
tempArrayNames.append("Remy")

avgArrayOfAllModelsHumanLikeness_np = np.array(tempArrayAvg)
nameArrayOfAllModelsHumanLikeness_np = np.array(tempArrayNames)

#print(avgArrayOfAllModelsHumanLikeness_np)
#print(nameArrayOfAllModelsHumanLikeness_np)

plt.title("Perceived human likeness of all evaluated 3D-Models")
plt.xlabel("3D-Models")
plt.ylabel("Avg. human likeness")

plt.plot(nameArrayOfAllModelsHumanLikeness_np, avgArrayOfAllModelsHumanLikeness_np, color='gray', zorder=1)
plt.scatter(nameArrayOfAllModelsHumanLikeness_np, avgArrayOfAllModelsHumanLikeness_np, color='red', zorder=3) 
plt.show()
# ---------------------------------------------------


# ---------------------------------------------------
# Perceived eeriness
# -----------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('AllUncannyValleyConditionStatisticResults_DataFrame.csv', sep=";")

avgArrayOfAllModelsHumanLikeness=[]
nameArrayOfAllModelsHumanLikeness=[]
questionId = "q2_"

for index, row in df.iterrows():
    question = str(row[0])
    if questionId in question:
        avgArrayOfAllModelsHumanLikeness.append(float(str(row[3]).replace(",", ".")))
        nameArrayOfAllModelsHumanLikeness.append(question.replace(questionId, ""))

#print(avgArrayOfAllModelsHumanLikeness)
#print(nameArrayOfAllModelsHumanLikeness)

# sorting the list:
# Eyebot, Turret, JRRobo, Lloyd, Atlas, Ribbot, Katie, Alice, Freddy, Medic, Link, Dutchess, Zombie, MixamoGirl, Remy

tempArrayAvg = []
tempArrayNames = []

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[12])
tempArrayNames.append("Eyebot")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[11])
tempArrayNames.append("Turret")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[4])
tempArrayNames.append("JRRobo")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[6])
tempArrayNames.append("Lloyd")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[15])
tempArrayNames.append("Atlas")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[13])
tempArrayNames.append("Ribbot")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[10])
tempArrayNames.append("Katie")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[9])
tempArrayNames.append("Alice")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[0])
tempArrayNames.append("Freddy")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[7])
tempArrayNames.append("Medic")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[1])
tempArrayNames.append("Link")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[8])
tempArrayNames.append("Dutchess")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[3])
tempArrayNames.append("Zombie")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[14])
tempArrayNames.append("MixamoGirl")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[5])
tempArrayNames.append("Remy")

avgArrayOfAllModelsHumanLikeness_np = np.array(tempArrayAvg)
nameArrayOfAllModelsHumanLikeness_np = np.array(tempArrayNames)

#print(avgArrayOfAllModelsHumanLikeness_np)
#print(nameArrayOfAllModelsHumanLikeness_np)

plt.title("Perceived eeriness of all evaluated 3D-Models")
plt.xlabel("3D-Models")
plt.ylabel("Avg. eeriness")

plt.plot(nameArrayOfAllModelsHumanLikeness_np, avgArrayOfAllModelsHumanLikeness_np, color='gray', zorder=1)
plt.scatter(nameArrayOfAllModelsHumanLikeness_np, avgArrayOfAllModelsHumanLikeness_np, color='red', zorder=3) 
plt.show()
# ---------------------------------------------------


# ---------------------------------------------------
# Perceived likability
# -----------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('AllUncannyValleyConditionStatisticResults_DataFrame.csv', sep=";")

avgArrayOfAllModelsHumanLikeness=[]
nameArrayOfAllModelsHumanLikeness=[]
questionId = "q3_"

for index, row in df.iterrows():
    question = str(row[0])
    if questionId in question:
        avgArrayOfAllModelsHumanLikeness.append(float(str(row[3]).replace(",", ".")))
        nameArrayOfAllModelsHumanLikeness.append(question.replace(questionId, ""))

#print(avgArrayOfAllModelsHumanLikeness)
#print(nameArrayOfAllModelsHumanLikeness)

# sorting the list:
# Eyebot, Turret, JRRobo, Lloyd, Atlas, Ribbot, Katie, Alice, Freddy, Medic, Link, Dutchess, Zombie, MixamoGirl, Remy

tempArrayAvg = []
tempArrayNames = []

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[12])
tempArrayNames.append("Eyebot")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[11])
tempArrayNames.append("Turret")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[4])
tempArrayNames.append("JRRobo")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[6])
tempArrayNames.append("Lloyd")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[15])
tempArrayNames.append("Atlas")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[13])
tempArrayNames.append("Ribbot")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[10])
tempArrayNames.append("Katie")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[9])
tempArrayNames.append("Alice")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[0])
tempArrayNames.append("Freddy")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[7])
tempArrayNames.append("Medic")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[1])
tempArrayNames.append("Link")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[8])
tempArrayNames.append("Dutchess")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[3])
tempArrayNames.append("Zombie")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[14])
tempArrayNames.append("MixamoGirl")

tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[5])
tempArrayNames.append("Remy")

avgArrayOfAllModelsHumanLikeness_np = np.array(tempArrayAvg)
nameArrayOfAllModelsHumanLikeness_np = np.array(tempArrayNames)

#print(avgArrayOfAllModelsHumanLikeness_np)
#print(nameArrayOfAllModelsHumanLikeness_np)

plt.title("Perceived likability of all evaluated 3D-Models")
plt.xlabel("3D-Models")
plt.ylabel("Avg. likability")

plt.plot(nameArrayOfAllModelsHumanLikeness_np, avgArrayOfAllModelsHumanLikeness_np, color='gray', zorder=1)
plt.scatter(nameArrayOfAllModelsHumanLikeness_np, avgArrayOfAllModelsHumanLikeness_np, color='red', zorder=3) 
plt.show()
# ---------------------------------------------------


# AllUncannyValleyStatisticResults_DataFrame
# ---------------------------------------------------
# Perceived human likeness
# -----------------------------------------------------------------------------------------------------------------------
""" df = pd.read_csv('AllUncannyValleyStatisticResults_DataFrame.csv', sep=";")

avgArrayOfAllModelsHumanLikeness=[]
nameArrayOfAllModelsHumanLikeness=[]

for index, row in df.iterrows():
    #print(index)
    if (index % 3) == 0:
        avgArrayOfAllModelsHumanLikeness.append(float(str(row[3]).replace(",", ".")))
        nameArrayOfAllModelsHumanLikeness.append(str(row[0]).replace("q1_", ""))

print(avgArrayOfAllModelsHumanLikeness)
print(nameArrayOfAllModelsHumanLikeness)

# sorting the list:
# Eyebot, Turret, JRRobo, Lloyd, Atlas, Ribbot, Katie, Alice, Freddy, Medic, Link, Dutchess, Zombie, MixamoGirl, Remy
# (Gettie)

tempArrayAvg = []
tempArrayNames = []

#0
tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[12])
tempArrayNames.append("Eyebot")

#1
tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[11])
tempArrayNames.append("Turret")

#2
tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[4])
tempArrayNames.append("JRRobo")

#3
tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[6])
tempArrayNames.append("Lloyd")

#4
tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[15])
tempArrayNames.append("Atlas")

#5
tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[13])
tempArrayNames.append("Ribbot")

#6
tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[10])
tempArrayNames.append("Katie")

#7
tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[9])
tempArrayNames.append("Alice")

#8
tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[0])
tempArrayNames.append("Freddy")

#9
tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[7])
tempArrayNames.append("Medic")

#10
tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[1])
tempArrayNames.append("Link")

#11
tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[8])
tempArrayNames.append("Dutchess")

#12
tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[3])
tempArrayNames.append("Zombie")

#13
tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[14])
tempArrayNames.append("MixamoGirl")

#14
tempArrayAvg.append(avgArrayOfAllModelsHumanLikeness[5])
tempArrayNames.append("Remy")

avgArrayOfAllModelsHumanLikeness_np = np.array(tempArrayAvg)
nameArrayOfAllModelsHumanLikeness_np = np.array(tempArrayNames)

print(avgArrayOfAllModelsHumanLikeness_np)
print(nameArrayOfAllModelsHumanLikeness_np)


plt.title("Perceived human likeness of all evaluated 3D-Models")
plt.xlabel("3D-Models")
plt.ylabel("Avg. human likeness")

plt.plot(nameArrayOfAllModelsHumanLikeness_np, avgArrayOfAllModelsHumanLikeness_np, color='gray', zorder=1)
plt.scatter(nameArrayOfAllModelsHumanLikeness_np, avgArrayOfAllModelsHumanLikeness_np, color='red', zorder=3) 
#plt.scatter(rr_2d_np[:,0], rr_2d_np[:,1], zorder=2)
plt.show() """

 
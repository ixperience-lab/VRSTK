import os

# Opening the file with absolute path
raw_data_file = open(r'BitalinoRawData_Stage 2_id-16_Condition B_2022-08-27_11-21-27.txt', 'r')

ecg_file = 'ecg.txt'
ecg_file_content = '# Simple Text Format\n# Sampling Rate (Hz):= 1000.00\n# Resolution:= 10\n# Labels:= ECG\n'

eda_file = 'eda.txt'
eda_file_content = '# Simple Text Format\n# Sampling Rate (Hz):= 1000.00\n# Resolution:= 10\n# Labels:= EDA\n'

# read file
for line in raw_data_file:
    firstSplit = line.split(";")
    #print (firstSplit)
    secondSplit = firstSplit[1].split("[")
    #print (secondSplit)
    thridSplit = secondSplit[1].split(" ")
    #print (secondSplit)
    #print (thridSplit[0])
    #print (thridSplit[1])
    eda_file_content += thridSplit[0] + "\n"
    ecg_file_content += thridSplit[1] + "\n"

# Closing the file after reading
raw_data_file.close()

# ".\\"
#-------------------------------
if os.path.exists(ecg_file):
    os.remove(ecg_file)

# open file for writting
ecg_file_open = open(ecg_file, "w")
# write to file
ecg_file_open.write(ecg_file_content)
# Closing the file after writting
ecg_file_open.close()

# "..\\results\\"
#-----------------------------------
ecg_file_raw = '../results/ecg_RAW.txt'
if os.path.exists(ecg_file_raw):
    os.remove(ecg_file_raw)

# open file for writting
ecg_file_raw_open = open(ecg_file_raw, "w")
# write to file
ecg_file_raw_open.write(ecg_file_content)
# Closing the file after writting
ecg_file_raw_open.close()

# ".\\"
#------------------------------------
if os.path.exists(eda_file):
    os.remove(eda_file)
    
# open file for writting
eda_file_open = open(eda_file, "w")
# write to file
eda_file_open.write(eda_file_content)
# Closing the file after writting
eda_file_open.close()

# "..\\results\\"
#------------------------------------
eda_file_raw = '../results/eda_RAW.txt'
if os.path.exists(eda_file_raw):
    os.remove(eda_file_raw)
    
# open file for writting
eda_file_raw_open = open(eda_file_raw, "w")
# write to file
eda_file_raw_open.write(eda_file_content)
# Closing the file after writting
eda_file_raw_open.close()
# Python Module ConvertBitalinoRawDataForBioSPPy

import os

def prepare_ecg_files_for_extraction(input_file = None, ouput_file_name = None):
    
    print(input_file)
    print(ouput_file_name)
    
    # Opening the file with absolute path
    raw_data_file = open(input_file, 'r')
    
    ecg_file = './examples/ecg.txt'
    ecg_file_content = '# Simple Text Format\n# Sampling Rate (Hz):= 1000.00\n# Resolution:= 10\n# Labels:= ECG\n'

    # read file
    for line in raw_data_file:
        firstSplit = line.split(";")
        secondSplit = firstSplit[1].split("[")
        thridSplit = secondSplit[1].split(" ")
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
    ecg_file_raw = './results/' + ouput_file_name + '_RAW.txt'
    if os.path.exists(ecg_file_raw):
        os.remove(ecg_file_raw)

    # open file for writting
    ecg_file_raw_open = open(ecg_file_raw, "w")
    # write to file
    ecg_file_raw_open.write(ecg_file_content)
    # Closing the file after writting
    ecg_file_raw_open.close()


def prepare_eda_files_for_extraction(input_file = None, ouput_file_name = None):
    
    print(input_file)
    print(ouput_file_name)
    
    # Opening the file with absolute path
    raw_data_file = open(input_file, 'r')
    
    eda_file = './examples/eda.txt'
    eda_file_content = '# Simple Text Format\n# Sampling Rate (Hz):= 1000.00\n# Resolution:= 10\n# Labels:= EDA\n'

    # read file
    for line in raw_data_file:
        firstSplit = line.split(";")
        secondSplit = firstSplit[1].split("[")
        thridSplit = secondSplit[1].split(" ")
        eda_file_content += thridSplit[0] + "\n"
    
    # Closing the file after reading
    raw_data_file.close()

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
    eda_file_raw = './results/' + ouput_file_name + '_RAW.txt'
    if os.path.exists(eda_file_raw):
        os.remove(eda_file_raw)
        
    # open file for writting
    eda_file_raw_open = open(eda_file_raw, "w")
    # write to file
    eda_file_raw_open.write(eda_file_content)
    # Closing the file after writting
    eda_file_raw_open.close()

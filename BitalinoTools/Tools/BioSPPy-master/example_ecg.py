import os
import sys

from biosppy import storage

import warnings

from biosppy.signals import ecg
from biosppy.signals import eda
from biosppy.signals.acc import acc

from examples.ConvertBitalinoRawDataForBioSPPy import prepare_ecg_files_for_extraction

# hewl1012
from os.path import exists

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_files(search_path):
     for (dirpath, _, filenames) in os.walk(search_path):
         for filename in filenames:
             yield os.path.join(dirpath, filename)


def run_heartrate_extraction():
    
    filenames = []
    list_files = get_files('./examples')
    for filename in list_files:
        if "_id-" in filename :
            filenames.append(filename)
        #print(filename)
    
    for file in filenames:
        #BitalinoRawData_Stage 0_id-20_Condition B_2022-09-01_03-34-29
        #Bitalinoi-Proband-Stage-0_id-19-Condition-B_ECG_FilteredHearRateResults
        temp_array = file.split("_")
        ecg_file = "Bitalinoi-Proband-" + temp_array[1].replace(" ", "-") + "_" + temp_array[2] + "_" + temp_array[3].replace(" ", "-") + "_ECG"
        #print(ecg_file)
        #print(file)

        prepare_ecg_files_for_extraction(file, ecg_file)
        # load raw ECG and ACC signals
        ecg_signal, _ = storage.load_txt('./examples/ecg.txt')

        # Setting current path
        current_dir = os.path.dirname(sys.argv[0])
        ecg_plot_path = os.path.join(current_dir, "results/" + ecg_file + '.png')

        # Process it and plot. Set interactive=True to display an interactive window
        out_ecg = ecg.ecg(signal=ecg_signal, sampling_rate=1000., path=ecg_plot_path + " # " + ecg_file, interactive=False)

        #print(out_ecg)

if __name__ == "__main__":
   run_heartrate_extraction()
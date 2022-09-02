import os
import sys

from biosppy import storage

import warnings

from biosppy.signals import ecg
from biosppy.signals import eda
from biosppy.signals.acc import acc

from examples.ConvertBitalinoRawDataForBioSPPy import prepare_eda_files_for_extraction


warnings.simplefilter(action='ignore', category=FutureWarning)


def get_files(search_path):
     for (dirpath, _, filenames) in os.walk(search_path):
         for filename in filenames:
             yield os.path.join(dirpath, filename)


def run_eda_extraction():

    filenames = []
    list_files = get_files('./examples')
    for filename in list_files:
        if "_id-" in filename :
            filenames.append(filename)
    
    for file in filenames:
        # BitalinoRawData_Stage 0_id-20_Condition B_2022-09-01_03-34-29
        # Bitalinoi-Proband-Stage-1_id-19-Condition-B_EDA_EdaResults
        temp_array = file.split("_")
        eda_file = "Bitalinoi-Proband-" + temp_array[1].replace(" ", "-") + "_" + temp_array[2] + "_" + temp_array[3].replace(" ", "-") + "_EDA"
        
        prepare_eda_files_for_extraction(file, eda_file)

        # load raw EDA signals
        eda_signal, _ = storage.load_txt('./examples/eda.txt')

        # Setting current path
        current_dir = os.path.dirname(sys.argv[0])
        eda_plot_path = os.path.join(current_dir, "results/" + eda_file + '.png')

        # Process it and plot. Set interactive=True to display an interactive window
        out_eda = eda.eda(signal=eda_signal, sampling_rate=1000., path=eda_plot_path + " # " + eda_file, show=True, min_amplitude=0.1)
        
        print(out_eda)

if __name__ == "__main__":
   run_eda_extraction()
import os
import sys

from biosppy import storage

import warnings

from biosppy.signals import ecg
from biosppy.signals import eda
from biosppy.signals.acc import acc

warnings.simplefilter(action='ignore', category=FutureWarning)

# load raw EDA signals
eda_signal, _ = storage.load_txt('./examples/eda.txt')

# Setting current path
current_dir = os.path.dirname(sys.argv[0])
eda_plot_path = os.path.join(current_dir, 'eda.png')

# Process it and plot. Set interactive=True to display an interactive window
out_eda = eda.eda(signal=eda_signal, sampling_rate=1000., path=eda_plot_path, show=True, min_amplitude=0.1)


from example_ecg import run_heartrate_extraction
from example_eda import run_eda_extraction

def run_biofeedback_extraction():
    # ecg
    run_heartrate_extraction()
    # eda
    run_eda_extraction()

if __name__ == "__main__":
   run_biofeedback_extraction()

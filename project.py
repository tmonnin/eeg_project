import os
import subprocess
import argparse
from matplotlib import pyplot as plt
import mne
import numpy as np
import utils

from _00_filter import Filter
from _01_clean_channels import CleanChannels
from _02_clean_segments import CleanSegments
from _03_ica import ICA
from _04_reference import Reference
from _10_erp_peak_extraction import ErpPeakExtraction
from _11_decoding_extraction import DecodingExtraction
from _12_time_frequency_extraction import TimeFrequencyExtraction
from _20_erp_peak_analysis import ErpPeakAnalysis
from _21_decoding_analysis import DecodingAnalysis
from _22_time_frequency_analysis import TimeFrequencyAnalysis

steps_subject = (Filter,
                 CleanChannels,
                 CleanSegments,
                 ICA,
                 Reference,
                 ErpPeakExtraction,
                 DecodingExtraction,
                 TimeFrequencyExtraction,
                )

steps_global = (ErpPeakAnalysis,
                DecodingAnalysis,
                TimeFrequencyAnalysis,
               )

# Subjects considered in analysis
subjects = ["001","002","003","004","005","006","007","008","009","010","011","012","013","014","015","016","017","018","019","020","021","022","023","024","025","026","027","028","029","030","031","032","033","034","035","036","037","038","039","040"]
assert len(subjects) == 40

jobs = 1

def execute(subject_id):
    for step in steps_subject:
        step().run(subject_id)

if __name__ == "__main__":
    # Handle command line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', metavar='###', help='The subject to process')
    subject = parser.parse_args().subject
    if subject is None:
        for i in range(0, len(subjects), jobs):
            process_handles = []
            for subject in subjects[i:i+jobs]:
                process_handle = subprocess.Popen(["python", __file__, "--subject", subject])
                process_handles.append(process_handle)
            exit_codes = [p.wait() for p in process_handles]
        for step in steps_global:
            step().run()
    else:
        execute(subject)

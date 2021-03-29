import json
import scipy
import numpy as np
import matplotlib.pyplot as plt
import mne
from config import fname
from base import Base
from _05_erp_peak_extraction import ErpPeakExtraction

class ErpPeakAnalysis(Base):

    def __init__(self):
        prev = ErpPeakExtraction()
        super().__init__(self.__class__.__name__.lower(), prev)

    def run(self):
        # Extract the study-relevant ERP peak subjectwise (e.g. one value per subject) and statistically test them. 
        # RQ: On which ERP-peaks do we find major difference between the conditions?
        # Channels, times, peaks for N170: --> Rossion 2008, ERP Core Paper
        # P7, PO7, P8, PO8; 130-200ms after stimulus onset
        electrode = self.config["electrode"]
        peaks = {}
        peak_lst = []
        evoked_faces_lst = []
        evoked_cars_lst = []
        for self.subject in self.config["subjects"]:
            #self.load()
            epoch = mne.read_epochs(fname.epochs(subject=self.subject))
            # Equalize the number of events per condition
            # http://predictablynoisy.com/mne-python/generated/mne.Epochs.html#mne.Epochs.equalize_event_counts
            # Dropped 9 epochs: 40, 116, 161, 389, 393, 399, 403, 455, 468
            epoch.equalize_event_counts(["faces", "cars"])
            evoked_faces = epoch["faces"].average()
            evoked_cars = epoch["cars"].average()
            evoked_faces_lst.append(evoked_faces)
            evoked_cars_lst.append(evoked_cars)
            evoked_difference = mne.combine_evoked([evoked_faces, evoked_cars], weights=[1, -1])
            #mne.viz.plot_compare_evokeds({"faces": evoked_faces, "cars": evoked_cars, "difference": evoked_difference}, picks=electrode, show=True)
            # Crop to relevant time frame between 150ms and 200ms as proposed in following tutorial:
            # https://mne.tools/dev/auto_tutorials/stats-sensor-space/plot_stats_cluster_1samp_test_time_frequency.html
            evoked_difference_cropped = evoked_difference.crop(tmin=0.13, tmax=0.2)
            # TODO use peak finder: https://mne.tools/dev/generated/mne.preprocessing.peak_finder.html
            # Extract peak amplitude on electrode PO8 with mne function
            # https://mne.tools/stable/generated/mne.EvokedArray.html#mne.EvokedArray.get_peak
            _, _, peak_difference = evoked_difference_cropped.pick(electrode).get_peak(return_amplitude=True)
            # Fill array with peak difference and 0 to statistically test if difference is significantly different from 0
            peak_lst.append([peak_difference, 0])
            peaks[self.subject] = peak_difference
        with open(fname.erppeaks(electrode=electrode), "w") as json_file:
            json.dump(peaks, json_file, indent=4)

        # Calculate grand average
        grand_avg_faces = mne.grand_average(evoked_faces_lst)
        grand_avg_cars = mne.grand_average(evoked_cars_lst)
        evoked_difference = mne.combine_evoked([grand_avg_faces, grand_avg_cars], weights=[1, -1])

        average = {"faces": grand_avg_faces, "cars": grand_avg_cars, "difference": evoked_difference}

        figure_grand_avg_difference = mne.viz.plot_compare_evokeds(average, picks=electrode, show=True)
        #self.add_figure

        # Evaluate t-test
        data = np.array(peak_lst)
        hist = plt.hist(data, bins=10)
        plt.show()
        alpha = 0.05
        # Non-parametric paired t-test
        t_values, p_value = scipy.stats.ttest_1samp(data[:,0], 0.0, alternative="less")

        word = "not "*(p_value >= alpha)
        print(f"Difference of ERP peak between face and car condition is {word}significant with alpha={alpha} and p-value={p_value}.")

if __name__ == '__main__':
    ErpPeakAnalysis().run()
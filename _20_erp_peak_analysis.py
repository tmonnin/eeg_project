import json
import scipy
import numpy as np
import matplotlib.pyplot as plt
import mne
from config import fname
from base import Base
from _10_erp_peak_extraction import ErpPeakExtraction

class ErpPeakAnalysis(Base):

    def __init__(self):
        prev = ErpPeakExtraction()
        super().__init__(self.__class__.__name__.lower(), prev, section=("Analysis", "ERP Peak Analysis"))

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
            epoch = mne.read_epochs(fname.epochs(subject=self.subject))
            # Do not equalize the number of events per condition to avoid problems with all subjects counting equally
            # http://predictablynoisy.com/mne-python/generated/mne.Epochs.html#mne.Epochs.equalize_event_counts
            #epoch.equalize_event_counts(["faces", "cars"])
            evoked_faces = epoch["faces"].average()
            evoked_cars = epoch["cars"].average()
            evoked_faces_lst.append(evoked_faces)
            evoked_cars_lst.append(evoked_cars)
            evoked_difference = mne.combine_evoked([evoked_faces, evoked_cars], weights=[1, -1])
            #mne.viz.plot_compare_evokeds({"faces": evoked_faces, "cars": evoked_cars, "difference": evoked_difference}, picks=electrode, show=True)
            # Crop to relevant time frame between 150ms and 200ms as proposed in Rossion 2008
            evoked_difference_cropped = evoked_difference.crop(tmin=0.13, tmax=0.2)
            # Potential extension: use peak finder: https://mne.tools/dev/generated/mne.preprocessing.peak_finder.html
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

        figure_grand_avg_difference = mne.viz.plot_compare_evokeds(average, picks=electrode, show=False)
        self.add_figure(figure_grand_avg_difference, caption="Grand average of evokeds for conditions 'faces' and 'cars'")


        # Non-parametric paired t-test
        data = np.array(peak_lst)
        alpha = 0.05
        t_values, p_value = scipy.stats.ttest_1samp(data[:,0], 0.0, alternative="less")

        # Evaluate distribution
        figure_histogram, ax_histogram = plt.subplots()
        ax_histogram.hist(data*1e6, bins=10)
        ax_histogram.set_xlabel(r"Potential [$\mu V$]")
        ax_histogram.set_ylabel("Count")
        if p_value < alpha:
            ax_histogram.set_title(f"Difference between face/car condition is significant:\nalpha={alpha}, p-value={p_value:.5f}")
        else:
            ax_histogram.set_title(f"Difference between face/car condition is NOT significant:\nalpha={alpha}, p-value={p_value:.5f}")
        self.add_figure(figure_histogram, caption="Histogram of ERP peaks in difference wave 'faces-cars' (blue) vs. Null hypothesis (orange)")
        

        self.report(analysis=True)


if __name__ == '__main__':
    ErpPeakAnalysis().run()
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from config import fname
from base import Base
from _12_time_frequency_extraction import TimeFrequencyExtraction


class TimeFrequencyAnalysis(Base):
    def __init__(self):
        prev = TimeFrequencyExtraction()
        super().__init__(self.__class__.__name__.lower(), prev, section=("Analysis", "Time Frequency Analysis"))

    def run(self):
        # Time Frequency analysis: Calculate an induced time-frequency analysis of the main experimental contrast
        # RQ: What oscillations underley our effect of interest?

        power_difference_total_lst = []
        power_difference_evoked_lst = []
        power_difference_induced_lst = []

        for self.subject in self.config["subjects"]:
            power_difference_total = mne.time_frequency.read_tfrs(fname.power_difference_total(subject=self.subject))
            power_difference_total_lst.append(power_difference_total[0])
            power_difference_evoked = mne.time_frequency.read_tfrs(fname.power_difference_evoked(subject=self.subject))
            power_difference_evoked_lst.append(power_difference_evoked[0])
            power_difference_induced = mne.time_frequency.read_tfrs(fname.power_difference_induced(subject=self.subject))
            power_difference_induced_lst.append(power_difference_induced[0])

        power_difference_total_avg = mne.combine_evoked(power_difference_total_lst, weights="equal")
        power_difference_evoked_avg = mne.combine_evoked(power_difference_evoked_lst, weights="equal")
        power_difference_induced_avg = mne.combine_evoked(power_difference_induced_lst, weights="equal")

        figure_power_spectrum_avg, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, figsize=(14,4))
        mode = "mean"
        baseline = None
        cmin = -3e-10
        cmax = -cmin

        ax1.set_title(f"Average: Difference total at {self.config['electrode']}")
        power_difference_total_avg.plot(
            axes=ax1,
            baseline=baseline,
            picks=self.config["electrode"],
            mode=mode,
            vmin=cmin,
            vmax=cmax,
            show=False
        )
        ax2.set_title(f"Average: Difference evoked at {self.config['electrode']}")
        power_difference_evoked_avg.plot(
            axes=ax2,
            baseline=baseline,
            picks=self.config["electrode"],
            mode=mode,
            vmin=cmin,
            vmax=cmax,
            show=False
        )
        ax3.set_title(f"Average: Difference induced at {self.config['electrode']}")
        power_difference_induced_avg.plot(
            axes=ax3,
            baseline=baseline,
            picks=self.config["electrode"],
            mode=mode,
            vmin=cmin,
            vmax=cmax,
            show=False
        )
        figure_power_spectrum_avg.suptitle(f"Difference at {self.config['electrode']}")

        self.add_figure(
            figure=figure_power_spectrum_avg,
            caption="Average power spectrum: Total, Evoked, Induced"
        )

        self.report(analysis=True)


if __name__ == "__main__":
    TimeFrequencyAnalysis().run()

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
        power_difference_total_data = []
        power_difference_evoked_lst = []
        power_difference_evoked_data = []
        power_difference_induced_lst = []
        power_difference_induced_data = []

        for self.subject in self.config["subjects"]:
            power_difference_total = mne.time_frequency.read_tfrs(fname.power_difference_total(subject=self.subject))
            power_difference_total_lst.append(power_difference_total[0])
            power_difference_total_data.append(power_difference_total[0].data[0])
            power_difference_evoked = mne.time_frequency.read_tfrs(fname.power_difference_evoked(subject=self.subject))
            power_difference_evoked_lst.append(power_difference_evoked[0])
            power_difference_evoked_data.append(power_difference_evoked[0].data[0])
            power_difference_induced = mne.time_frequency.read_tfrs(fname.power_difference_induced(subject=self.subject))
            power_difference_induced_lst.append(power_difference_induced[0])
            power_difference_induced_data.append(power_difference_induced[0].data[0])

        power_difference_total_avg = mne.combine_evoked(power_difference_total_lst, weights="equal")
        power_difference_evoked_avg = mne.combine_evoked(power_difference_evoked_lst, weights="equal")
        power_difference_induced_avg = mne.combine_evoked(power_difference_induced_lst, weights="equal")

        figure_power_spectrum_avg, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, figsize=(14, 4))
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
            show=False,
        )
        ax2.set_title(f"Average: Difference evoked at {self.config['electrode']}")
        power_difference_evoked_avg.plot(
            axes=ax2,
            baseline=baseline,
            picks=self.config["electrode"],
            mode=mode,
            vmin=cmin,
            vmax=cmax,
            show=False,
        )
        ax3.set_title(f"Average: Difference induced at {self.config['electrode']}")
        power_difference_induced_avg.plot(
            axes=ax3,
            baseline=baseline,
            picks=self.config["electrode"],
            mode=mode,
            vmin=cmin,
            vmax=cmax,
            show=False,
        )
        figure_power_spectrum_avg.suptitle(f"Difference at {self.config['electrode']}")
        # Copy and reshape tick labels for usage in significance plots
        tick_labels = ax1.get_yticks().astype(int).tolist()
        del tick_labels[3::4]
        self.add_figure(
            figure=figure_power_spectrum_avg,
            caption="Average power spectrum",
        )

        # Topo not very helpful since pre-analysis step only exports electrode PO8
        #figure_topo_total = power_difference_total_avg.plot_topo(baseline=None, mode="logratio", title="Topo plot of total difference of power spectrum", show=False)
        #self.add_figure(figure=figure_topo_total, caption="Topo plot of total difference of power spectrum")
        #figure_topo_evoked = power_difference_evoked_avg.plot_topo(baseline=None, mode='logratio', title="Topo plot of evoked difference of power spectrum", show=False)
        #self.add_figure(figure=figure_topo_evoked, caption="Topo plot of evoked difference of power spectrum")
        #figure_topo_induced = power_difference_induced_avg.plot_topo(baseline=None, mode='logratio', title="Topo plot of induced difference of power spectrum", show=False)
        #self.add_figure(figure=figure_topo_induced, caption="Topo plot of induced difference of power spectrum")

        power_difference_total_data = np.array(power_difference_total_data)
        power_difference_evoked_data = np.array(power_difference_evoked_data)
        power_difference_induced_data = np.array(power_difference_induced_data)

        # Significance testing
        alpha = self.config["alpha"]
        threshold = self.config["threshold_cluster_permutation_test"]
        cluster_array_total = self.cluster_permutation_test(power_difference_total_data, alpha, threshold)
        cluster_array_evoked = self.cluster_permutation_test(power_difference_evoked_data, alpha, threshold)
        cluster_array_induced = self.cluster_permutation_test(power_difference_induced_data, alpha, threshold)

        figure_significance_test, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True, figsize=(14, 4))
        extent = (power_difference_total_avg.times[0], power_difference_total_avg.times[-1], 5, 50)

        ax1.imshow(cluster_array_total, cmap="gray", origin="lower", aspect="auto", extent=extent)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Frequency [Hz]")
        ax1.set_yticklabels(tick_labels)
        ax1.set_title(f"Significant power areas for total effect (white)")

        ax2.imshow(cluster_array_evoked, cmap="gray", origin="lower", aspect="auto", extent=extent)
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Frequency [Hz]")
        ax2.set_yticklabels(tick_labels)
        ax2.set_title(f"Significant power areas for evoked effect (white)")

        ax3.imshow(cluster_array_induced, cmap="gray", origin="lower", aspect="auto", extent=extent)
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Frequency [Hz]")
        ax3.set_yticklabels(tick_labels)
        ax3.set_title(f"Significant power areas for induced effect (white)")

        self.add_figure(
            figure_significance_test,
            caption=f"Significance of averaged power spectrum difference, alpha={alpha}, threshold={threshold}",
        )

        self.report(analysis=True)

    @staticmethod
    def cluster_permutation_test(data, alpha, thresh):
        t_values, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
            [
                data,
                np.zeros(data.shape),
            ],
            threshold=thresh,
        )
        cluster_array = np.zeros_like(t_values)
        for cluster, p_value in zip(clusters, cluster_p_values):
            if p_value <= alpha:
                cluster_array[cluster] = 1.0

        return cluster_array


if __name__ == "__main__":
    TimeFrequencyAnalysis().run()

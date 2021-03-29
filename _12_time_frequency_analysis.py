import os
import json
import numpy as np
import matplotlib.pyplot as plt
import mne
from config import fname
from base import Base
from _05_erp_peak_extraction import ErpPeakExtraction


class TimeFrequencyAnalysis(Base):
    def __init__(self):
        prev = ErpPeakExtraction()
        super().__init__(self.__class__.__name__.lower(), prev)

    def run(self):
        # Time Frequency analysis: Calculate an induced time-frequency analysis of the main experimental contrast
        # RQ: What oscillations underley our effect of interest?

        power_difference_total_lst = []
        power_difference_evoked_lst = []
        power_difference_induced_lst = []

        for self.subject in self.config["subjects"]:
            epochs = mne.read_epochs(fname.epochs(subject=self.subject))
            # epochs.equalize_event_counts(["faces", "cars"])
            epochs.resample(256, npad="auto")
            epochs_induced_faces = epochs["faces"].copy().pick(self.config["electrode"]).subtract_evoked()
            epochs_induced_cars = epochs["cars"].copy().pick(self.config["electrode"]).subtract_evoked()

            freq = np.logspace(*np.log10([5, 50]), num=30)
            # The number of cycles defines the trade-off between time and frequency
            # Increasing the number of cycles increases frequency resolution
            cycles = freq / 3

            # Transformation to frequency domain via Morlet wavelets

            power_faces = mne.time_frequency.tfr_morlet(
                epochs["faces"],
                freqs=freq,
                n_cycles=cycles,
                use_fft=True,
                return_itc=False,
                decim=3,
                n_jobs=0,
                picks=self.config["electrode"],
                average=True,
            )

            power_cars = mne.time_frequency.tfr_morlet(
                epochs["cars"],
                freqs=freq,
                n_cycles=cycles,
                use_fft=True,
                return_itc=False,
                decim=3,
                n_jobs=0,
                picks=self.config["electrode"],
                average=True,
            )

            power_faces_induced = mne.time_frequency.tfr_morlet(
                epochs_induced_faces,
                freqs=freq,
                n_cycles=cycles,
                use_fft=True,
                return_itc=False,
                decim=3,
                n_jobs=0,
                picks=self.config["electrode"],
                average=True,
            )

            power_cars_induced = mne.time_frequency.tfr_morlet(
                epochs_induced_cars,
                freqs=freq,
                n_cycles=cycles,
                use_fft=True,
                return_itc=False,
                decim=3,
                n_jobs=0,
                picks=self.config["electrode"],
                average=True,
            )

            # Evoke within subject based on power spectrum
            power_faces_evoked = mne.combine_evoked([power_faces, power_faces_induced], weights=[1, -1])
            power_cars_evoked = mne.combine_evoked([power_cars, power_cars_induced], weights=[1, -1])
            # Calculate difference of conditions in power spectrum
            power_difference_total = mne.combine_evoked([power_faces, power_cars], weights=[1, -1])
            power_difference_evoked = mne.combine_evoked([power_faces_evoked, power_cars_evoked], weights=[1, -1])
            power_difference_induced = mne.combine_evoked([power_faces_induced, power_cars_induced], weights=[1, -1])

            figure_power_spectrum_avg, (ax1, ax2, ax3) = plt.subplots(1, 3)
            mode = "mean"
            baseline = None
            cmin = -1e-9
            cmax = -cmin

            ax1.set_title(f"Difference total at {self.config['electrode']}")
            power_difference_total.plot(
                axes=ax1,
                baseline=baseline,
                picks=self.config["electrode"],
                mode=mode,
                vmin=cmin,
                vmax=cmax,
                show=False,
            )
            ax2.set_title(f"Difference evoked at {self.config['electrode']}")
            power_difference_evoked.plot(
                axes=ax2,
                baseline=baseline,
                picks=self.config["electrode"],
                mode=mode,
                vmin=cmin,
                vmax=cmax,
                show=False,
            )
            ax3.set_title(f"Difference induced at {self.config['electrode']}")
            power_difference_induced.plot(
                axes=ax3,
                baseline=baseline,
                picks=self.config["electrode"],
                mode=mode,
                vmin=cmin,
                vmax=cmax,
                show=False,
            )
            figure_power_spectrum_avg.suptitle(f"Difference at {self.config['electrode']}")

            self.add_figure(
                figure=figure_power_spectrum_avg,
                caption="Power spectrum: Total, Evoked, Induced",
                section="Analysis",
            )

            power_difference_total_lst.append(power_difference_total)
            power_difference_evoked_lst.append(power_difference_evoked)
            power_difference_induced_lst.append(power_difference_induced)

        # power_total.plot_topo(show=False)

        plt.close("all")

        power_difference_total_avg = mne.combine_evoked(power_difference_total_lst, weights="equal")
        power_difference_evoked_avg = mne.combine_evoked(power_difference_evoked_lst, weights="equal")
        power_difference_induced_avg = mne.combine_evoked(power_difference_induced_lst, weights="equal")

        figure_power_spectrum_avg, (ax1, ax2, ax3) = plt.subplots(1, 3)
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
        plt.show()
        self.add_figure(
            figure=figure_power_spectrum_avg,
            caption="Average power spectrum: Total, Evoked, Induced",
            section="Analysis",
        )


if __name__ == "__main__":
    TimeFrequencyAnalysis().run()

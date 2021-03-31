import os
import numpy as np
import mne
import utils
from config import fname
from base import Base
from _02_clean_segments import CleanSegments

class ICA(Base):

    def __init__(self):
        prev = CleanSegments()
        super().__init__(self.__class__.__name__.lower(), prev, section=("Preprocessing", "ICA"))

    def process(self):
        # Copy the raw object before altering due to ICA
        raw_pre_ica = self.raw.copy()
        raw_ica = self.raw.copy()
        raw_ica.load_data()
        # https://mne.tools/stable/generated/mne.preprocessing.ICA.html
        # "ICA is sensitive to low-frequency drifts and therefore requires the data to be high-pass filtered prior to fitting. Typically, a cutoff frequency of 1 Hz is recommended."
        # -> 2.0Hz High-pass results in -6dB cutoff frequency of 1.0Hz
        raw_ica.filter(self.config["fmin"], None, fir_design='firwin')
        #FIR filter parameters
            #---------------------
            #Designing a one-pass, zero-phase, non-causal highpass filter:
            #- Windowed time-domain design (firwin) method
            #- Hamming window with 0.0194 passband ripple and 53 dB stopband attenuation
            #- Lower passband edge: 2.00
            #- Lower transition bandwidth: 2.00 Hz (-6 dB cutoff frequency: 1.00 Hz)
            #- Filter length: 1691 samples (1.651 sec)

        if self.subject in self.config["subjects_preprocess"]:
            # Compute and fit ICA
            ica = mne.preprocessing.ICA(method="fastica")
            ica.fit(raw_ica, verbose=True)

            ica_interactive = False
            if ica_interactive:
                ica.plot_components(range(15), show=True)
                #ica.detect_artifacts()
                ica.plot_properties(raw_ica, picks=range(15), psd_args={'fmax': 35.}, reject=None, show=True)

            # Subject is one of the three ones that are cleaned manually
            path_badica = os.path.join(fname.annotations_dir, f"sub-{self.subject}_task-{self.task}_badica.txt")
            with open(path_badica, "r") as f:
                line = f.readline()
                bad_comps = line.split(",")
                bad_comps = np.array(bad_comps).astype(np.float)
        else:
            # Subject is not one of the manually cleaned ones, use precomputed ica
            ica, bad_comps = utils.load_ica(self.task, self.subject)
            ica = utils.ica_info(self.raw, ica)

        #ica.plot_properties(self.raw)
        # https://mne.tools/stable/generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.get_sources
        #Remove selected components from the signal.
        self.raw.load_data()
        # Apply bad ICA components on real raw object
        # ica.apply() works in-place and returns the altered object, doing both here
        self.raw = ica.apply(self.raw, exclude=bad_comps)

        raw_pre_ica.load_data()
        figure_ica_overlay = ica.plot_overlay(raw_pre_ica, exclude=bad_comps, show=False)
        self.add_figure(figure=figure_ica_overlay, caption="ICA Overlay, ordinate in [V]")


if __name__ == '__main__':
    ICA().run()
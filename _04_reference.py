import mne
from base import Base
from _03_ica import ICA

class Reference(Base):

    def __init__(self):
        prev = ICA()
        super().__init__(self.__class__.__name__.lower(), prev, section=("Preprocessing", "Re-Reference"))

    def process(self):
        # https://mne.tools/stable/generated/mne.set_eeg_reference.html
        # Literature suggests "Average" as new reference for N170 task 
        # https://www.sciencedirect.com/science/article/pii/S1053811920309502
        # "For analysis of the N170, the EEG signals were referenced to the average of all 33 sites (because the average reference is standard in the N170 literature)."
        # Choose projection=True to ensure that reference is correctly set even if channel choice is changed later on
        # "it is strongly recommended to use the average-reference-as-projection approach"
        # https://mne.tools/stable/auto_tutorials/preprocessing/plot_55_setting_eeg_reference.html
        # Terminal output shows warning: "Average reference projection was added, but has not been applied yet."
        # "By default, raw.plot() will apply the projectors in the background before plotting"
        # https://mne.tools/dev/auto_tutorials/preprocessing/plot_45_projectors_background.html
        self.raw.load_data()
        mne.set_eeg_reference(self.raw, ref_channels='average', copy=False, projection=True, ch_type='auto', forward=None, verbose=None)
        figure_without_reference = self.raw.plot(proj=False, n_channels=5, start=110, show=False, scalings=40e-6)
        self.add_figure(figure=figure_without_reference, caption="EEG without re-referencing for first 5 channels")
        figure_with_reference = self.raw.plot(proj=True, n_channels=5, start=110, show=False, scalings=40e-6)
        self.add_figure(figure=figure_with_reference, caption="EEG with re-referencing for first 5 channels")


if __name__ == '__main__':
    Reference().run()
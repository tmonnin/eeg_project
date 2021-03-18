import mne
from base import Base
from _03_ica import ICA

class Reference(Base):

    def __init__(self):
        prev = ICA()
        super().__init__(self.__class__.__name__.lower(), prev)

    def process(self):
        # https://mne.tools/stable/generated/mne.set_eeg_reference.html
        # Literature suggests "Average" as new reference for N170 task 
        # https://www.sciencedirect.com/science/article/pii/S1053811920309502
        # "For analysis of the N170, the EEG signals were referenced to the average of all 33 sites (because the average reference is standard in the N170 literature)."
        self.raw.load_data()
        mne.set_eeg_reference(self.raw, ref_channels='average', copy=False, projection=False, ch_type='auto', forward=None, verbose=None)
        figure_after_reference = self.raw.plot(n_channels=len(self.raw.ch_names), show=False)#,scalings =40e-6)
        self.add_figure(figure=figure_after_reference, caption="After EEG is re-referenced", section="Preprocessing")


if __name__ == '__main__':
    Reference().run()
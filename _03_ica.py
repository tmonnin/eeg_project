import mne
import utils
from base import Base
from _02_clean_segments import CleanSegments

class ICA(Base):

    def __init__(self):
        prev = CleanSegments()
        super().__init__(self.__class__.__name__.lower(), prev)

    def process(self):
        # TODO compare pre and post removing bad components with ICA
        figure_pre_ica = self.raw.plot(show=False)
        #self.add_figure(figure=figure_pre_ica, caption="Before applying ICA", section="Preprocessing")
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
        ica_interactive = False
        if ica_interactive:        
            ica = mne.preprocessing.ICA(method="fastica")
            ica.fit(raw_ica, verbose=True)
            ica.plot_components(range(10))
            # TODO check ica.detect_artifacts()
            #ica.plot_properties(raw_ica,picks=[0,1],psd_args={'fmax': 35.},reject=None)
            # Estimate sources given the unmixing matrix.
            #icaact = ica.get_sources(raw_ica)
            #plt.plot(icaact[5,0:20000][0].T)
            #ica.plot_overlay(raw_ica,exclude=[1,8,9])

        #### Use preprocessed data
        ica, bad_comps = utils.load_ica(self.task, self.subject)
        ica = utils.ica_info(self.raw, ica)
        # TODO check how preprocessed ICA matches my preprocessed raw object
        # https://mne.tools/stable/generated/mne.preprocessing.ICA.html#mne.preprocessing.ICA.get_sources
        #Remove selected components from the signal.
        self.raw.load_data()
        # Apply bad ICA components on real raw object
        ica.apply(self.raw, exclude=bad_comps)

        figure_post_ica = self.raw.plot(show=False)
        #self.add_figure(figure=figure_post_ica, caption="After applying ICA", section="Preprocessing")

        figure_ica_overlay = ica.plot_overlay(self.raw, exclude=bad_comps, show=False)
        self.add_figure(figure=figure_ica_overlay, caption="ICA Overlay", section="Preprocessing")


if __name__ == '__main__':
    ICA().run()
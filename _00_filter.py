from base import Base

class Filter(Base):
# Use default filter params according to mne tutorial:
# https://mne.tools/stable/auto_tutorials/discussions/plot_background_filtering.html#defaults-in-mne-python
# Low-pass filter at 0.6Hz for slow drifts
# High-pass filter at 50Hz for line noise
    def __init__(self):
        prev = None
        super().__init__(self.__class__.__name__.lower(), prev)

    def process(self):
        # Plot fourier space
        fourier_pre_filter = self.raw.plot_psd(area_mode='range', tmax=10.0, average=True,xscale="log", show=False)
        self.add_figure(figure=fourier_pre_filter, caption="Frequency diagram before filtering", section="Preprocessing")
        self.raw.filter(0.6, 50, fir_design='firwin')
        #FIR filter parameters
            #---------------------
            #Designing a one-pass, zero-phase, non-causal bandpass filter:
            #- Windowed time-domain design (firwin) method
            #- Hamming window with 0.0194 passband ripple and 53 dB stopband attenuation
            #- Lower passband edge: 0.50
            #- Lower transition bandwidth: 0.50 Hz (-6 dB cutoff frequency: 0.25 Hz)
            #- Upper passband edge: 50.00 Hz
            #- Upper transition bandwidth: 12.50 Hz (-6 dB cutoff frequency: 56.25 Hz)
            #- Filter length: 6759 samples (6.601 sec)
            #
            #Effective window size : 2.000 (s)
        fourier_post_filter = self.raw.plot_psd(area_mode='range', tmax=10.0, average=True,xscale="log", show=False)
        self.add_figure(figure=fourier_post_filter, caption="Frequency diagram after filtering", section="Preprocessing")


if __name__ == '__main__':
    Filter().run()
import mne
import utils
from base import Base
from _02_clean_segments import CleanSegments

class ICA(Base):

    def __init__(self):
        prev = CleanSegments()
        super().__init__(self.__class__.__name__.lower(), prev)

    def process(self):
        figure_pre_ica = self.raw.plot(show=False)
        #self.add_figure(figure=figure_pre_ica, caption="Before applying ICA", section="Preprocessing")

        ica_interactive = False
        if ica_interactive:        
            ica = mne.preprocessing.ICA(method="fastica")
            ica.fit(self.raw, verbose=True)
            ica.plot_components(range(10))
            #ica.plot_properties(raw,picks=[0,1],psd_args={'fmax': 35.},reject=None)
            # Estimate sources given the unmixing matrix.
            #icaact = ica.get_sources(raw)
            #plt.plot(icaact[5,0:20000][0].T)
            #ica.plot_overlay(raw,exclude=[1,8,9])

        #### Use preprocessed data
        ica, bad_comps = utils.load_ica(self.task, self.subject)
        ica = utils.ica_info(self.raw, ica)
        #Remove selected components from the signal.
        self.raw.load_data()
        ica.apply(self.raw, exclude=bad_comps)

        figure_post_ica = self.raw.plot(show=False)
        #self.add_figure(figure=figure_post_ica, caption="After applying ICA", section="Preprocessing")

        figure_ica_overlay = ica.plot_overlay(self.raw, exclude=bad_comps, show=False)
        self.add_figure(figure=figure_ica_overlay, caption="ICA Overlay", section="Preprocessing")


if __name__ == '__main__':
    ICA().run()
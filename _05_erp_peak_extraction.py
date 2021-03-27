import json
import mne
from config import fname
from base import Base
from _04_reference import Reference

class ErpPeakExtraction(Base):

    def __init__(self):
        prev = Reference()
        super().__init__(self.__class__.__name__.lower(), prev)

    def process(self):
        electrode = self.config["electrode"]
        epochs, evts_dict_categorized = self.get_epochs()
        epochs_face = epochs["faces"]
        epochs_car = epochs["cars"]
        figure_face = epochs_face.plot_image(combine='mean', picks=electrode, show=False, title="Face Stimulus")
        figure_car = epochs_car.plot_image(combine='mean', picks=electrode, show=False, title="Car Stimulus")
        average = {"Face": epochs_face.average(), "Car": epochs_car.average()}
        #mne.viz.plot_compare_evokeds(average, picks=electrode)

        difference = mne.combine_evoked([epochs_face.average(),
                                         epochs_car.average()],
                                         weights=[1, -1])
        # plot difference wave
        figure_difference = difference.plot_joint(times=[0.15], title='Face - Car', show=False)

        evt_plot = mne.viz.plot_events(self.evts, event_id=evts_dict_categorized, show=False)
        self.add_figure(figure=evt_plot, caption="Overview of events", section="Analyse")
        self.add_figure(figure=figure_difference, caption="Difference of conditions", section="Analyse")
        self.add_figure(figure=figure_face, caption="Face condition", section="Analyse")
        self.add_figure(figure=figure_car, caption="Car condition", section="Analyse")

        epochs.save(fname.epochs(subject=self.subject), overwrite=True)


if __name__ == '__main__':
    ErpPeakExtraction().run()
import json
import mne
from config import fname
from base import Base
from _04_reference import Reference

class ErpPeakExtraction(Base):

    def __init__(self):
        prev = Reference()
        super().__init__(self.__class__.__name__.lower(), prev, section=("Pre-Analysis", "ERP Peak Extraction"))

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

        average = {"faces": epochs_face.average(), "cars": epochs_car.average(), "difference": difference}
        # plot ERP
        figure_compare_evokeds = mne.viz.plot_compare_evokeds(average, picks=electrode, show=False)
        # plot difference wave
        figure_difference = difference.plot_joint(times=self.config["difference_wave_times"], title="Difference waves for conditions 'Face - Car'", show=False)

        evt_plot = mne.viz.plot_events(self.evts, event_id=evts_dict_categorized, show=False)
        self.add_figure(figure=evt_plot, caption="Overview of events")
        self.add_figure(figure=figure_difference, caption="Difference of conditions")
        self.add_figure(figure=figure_compare_evokeds, caption="Comparison of evokeds")
        self.add_figure(figure=figure_face, caption="Face condition")
        self.add_figure(figure=figure_car, caption="Car condition")

        epochs.save(fname.epochs(subject=self.subject), overwrite=True)


if __name__ == '__main__':
    ErpPeakExtraction().run()
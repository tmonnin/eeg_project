import json
import mne
from config import fname
from base import Base
from _04_reference import Reference

class ErpPeakAnalysis(Base):

    def __init__(self):
        prev = Reference()
        super().__init__(self.__class__.__name__.lower(), prev)

    def process(self):
        # Extract the study-relevant ERP peak subjectwise (e.g. one value per subject) and statistically test them. 
        # RQ: On which ERP-peaks do we find major difference between the conditions?
        # Channels, times, peaks for N170: --> Rossion 2008, ERP Core Paper
        # P7, PO7, P8, PO8; 130-200ms after stimulus onset
        electrode = self.config["electrode"]
        epochs_face = self.get_epochs(condition="face")
        epochs_car = self.get_epochs(condition="car")
        figure_face = epochs_face.plot_image(combine='mean', picks=electrode, show=False, title="Face Stimulus")
        figure_car = epochs_car.plot_image(combine='mean', picks=electrode, show=False, title="Car Stimulus")
        average = {"Face": epochs_face.average(), "Car": epochs_car.average()}
        #mne.viz.plot_compare_evokeds(average, picks=electrode)

        difference = mne.combine_evoked([epochs_face.average(),
                                         epochs_car.average()],
                                         weights=[1, -1])
        # plot difference wave
        figure_difference = difference.plot_joint(times=[0.15], title='Face - Car', show=False)

        evt_plot = mne.viz.plot_events(self.evts, event_id=self.evts_dict, show=False)
        self.add_figure(figure=evt_plot, caption="Overview of events", section="Analyse")
        self.add_figure(figure=figure_difference, caption="Difference of conditions", section="Analyse")
        self.add_figure(figure=figure_face, caption="Face condition", section="Analyse")
        self.add_figure(figure=figure_car, caption="Car condition", section="Analyse")

        try:
            with open(fname.erppeaks(electrode=electrode), "r") as json_file:
                peaks = json.load(json_file)
        except FileNotFoundError:
            peaks = {}
        peaks[self.subject] = {"faces": {}, "cars": {}}
        peaks[self.subject]["faces"] = epochs_face.load_data().pick(electrode).average().get_peak(ch_type="eeg")
        peaks[self.subject]["cars"] = epochs_car.load_data().pick(electrode).average().get_peak(ch_type="eeg")
        with open(fname.erppeaks(electrode=electrode), "w") as json_file:
            json.dump(peaks, json_file)


if __name__ == '__main__':
    ErpPeakAnalysis().run()
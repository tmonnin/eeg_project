import mne
from base import Base
import utils
from _00_filter import Filter

class CleanChannels(Base):

    def __init__(self):
        prev = Filter()
        super().__init__(self.__class__.__name__.lower(), prev, section=("Preprocessing", "Clean Channels"))

    def process(self):
        # Remove and interpolate bad channels
        select_bad_interactive = False
        if select_bad_interactive:
        # select bad channels
        # https://mne.tools/dev/auto_tutorials/preprocessing/plot_15_handling_bad_channels.html
        # visualize averaged epochs
            epochs.average().plot()
            # TODO spot bad channels in ICA

        if self.subject in self.config["subjects_preprocess"]:
            # Subject is one of the three ones that are cleaned manually
            bad_channels = self.config["bad_channels"]

        else:
            # Subject is not one of the manually cleaned ones, use precomputed data
            bad_channels = utils.load_bad_channels(task=self.task, subject_id=self.subject)
            if bad_channels is not None:
                if bad_channels.ndim == 0:  # catch 0 dimensional arrays and handle them with direct access
                    bad_channels = [bad_channels]
        bad_channels = [self.raw.info.ch_names[idx] for idx in bad_channels]
        self.raw.info['bads'].extend(bad_channels)

        #picks = mne.pick_channels_regexp(raw.ch_names, regexp='MEG 2..3')

        # compare raw eeg and eeg without bad channels
        all_eeg = mne.pick_types(self.raw.info, meg=False, eeg=True, eog=False, exclude=[])
        figure_all_channels = self.raw.plot(start=105, duration=2, butterfly=True, bad_color='r', order=all_eeg, n_channels=len(all_eeg), show=False)
        self.add_figure(figure=figure_all_channels, caption="Diagram with all channels (bad are red)")

        good_eeg = mne.pick_types(self.raw.info, meg=False, eeg=True, eog=False, exclude='bads')
        figure_good_channels = self.raw.plot(start=105, duration=2, butterfly=True, order=good_eeg, n_channels=len(good_eeg), show=False)
        self.add_figure(figure=figure_good_channels, caption="Diagram with good channels (bad are grayed out)")

        #print(np.setdiff1d(all_eeg, good_eeg))
        #print(np.array(raw.ch_names)[np.setdiff1d(all_eeg, good_eeg)])

        # interpolate bad times and channels
        self.raw.set_montage('standard_1020',match_case=False)
        self.raw.load_data()  # required for interpolation command
        self.raw.interpolate_bads()

        post_eeg = mne.pick_types(self.raw.info, meg=False, eeg=True, eog=False)
        figure_interpolated_bads = self.raw.plot(start=105, duration=2, butterfly=True, order=post_eeg, bad_color='r', n_channels=len(self.raw.ch_names), show=False)
        self.add_figure(figure=figure_interpolated_bads, caption="Diagram after interpolating bad channels")


if __name__ == '__main__':
    CleanChannels().run()
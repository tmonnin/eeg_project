import mne
from base import Base
from _00_filter import Filter

class CleanChannels(Base):

    def __init__(self):
        prev = Filter()
        super().__init__(self.__class__.__name__.lower(), prev)

    def process(self):
        # Remove and interpolate bad channels
        select_bad_interactive = False
        if select_bad_interactive:
        # select bad channels
        # https://mne.tools/dev/auto_tutorials/preprocessing/plot_15_handling_bad_channels.html
        # visualize averaged epochs
            epochs.average().plot()
            # TODO spot bad channels in ICA

        else:
            self.raw.info['bads'] = self.config["bad_channels"]#['FP2']
            #picks = mne.pick_channels_regexp(raw.ch_names, regexp='MEG 2..3')

            # compare raw eeg and eeg without bad channels
            all_eeg = mne.pick_types(self.raw.info, meg=False, eeg=True, exclude=[])
            figure_all_channels = self.raw.plot(order=all_eeg, n_channels=len(all_eeg), show=False)
            self.add_figure(figure=figure_all_channels, caption="Diagram with all channels", section="Preprocessing")

            good_eeg = mne.pick_types(self.raw.info, meg=False, eeg=True)#, exclude='bads')
            figure_good_channels = self.raw.plot(order=good_eeg, n_channels=len(good_eeg), show=False)
            self.add_figure(figure=figure_good_channels, caption="Diagram with good channels", section="Preprocessing")

            #print(np.setdiff1d(all_eeg, good_eeg))
            #print(np.array(raw.ch_names)[np.setdiff1d(all_eeg, good_eeg)])

            # interpolate bad times and channels
            self.raw.set_montage('standard_1020',match_case=False)
            self.raw.load_data()  # required for interpolation command
            self.raw.interpolate_bads()

            figure_interpolated_bads = self.raw.plot(n_channels=len(self.raw.ch_names), show=False)#,scalings =40e-6)
            self.add_figure(figure=figure_interpolated_bads, caption="Diagram after interpolating bad channels", section="Preprocessing")


if __name__ == '__main__':
    CleanChannels().run()
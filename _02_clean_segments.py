import mne
from autoreject import AutoReject
from base import Base
from _01_clean_channels import CleanChannels

class CleanSegments(Base):

    def __init__(self):
        prev = CleanChannels()
        super().__init__(self.__class__.__name__.lower(), prev)

    def process(self):
        epochs_none = self.get_epochs_none()
        epochs_manual = self.get_epochs_manual()
        epochs_thresh = self.get_epochs_thresh()
        epochs_ar = self.get_epochs_autoreject()
        # Compare different rejection techniques
        figure_compare_evoked = mne.viz.plot_compare_evokeds({
            'raw': epochs_none.average(),
            'manual': epochs_manual.average(),
            'thresh': epochs_thresh.average(),
            'ar': epochs_ar.average()
            }, picks="Cz", show=False)
        self.add_figure(figure=figure_compare_evoked, caption="Diagram of different approaches for cleaning segments", section="Preprocessing")

    def get_epochs_none(self):
        epochs = mne.Epochs(self.raw,self.evts,self.evts_dict_stim,tmin=-0.1,tmax=1,reject_by_annotation=False)
        return epochs

    def get_epochs_manual(self):
        select_bad_interactive = False
        if select_bad_interactive:
            # Open interactive tool to select bad segments
            self.raw.plot(n_channels=len(self.raw.ch_names), show=True)#,scalings =40e-6)
            #plt.show()
            bad_ix = [i for i,a in enumerate(self.raw.annotations) if a['description']=="BAD_"]
            self.raw.annotations[bad_ix].save(f"sub-{self.subject}_task-{self.task}_badannotations.txt")
        #https://mne.tools/dev/generated/mne.read_annotations.html
        #The annotations stored in a .csv require the onset columns to be timestamps. If you have onsets as floats (in seconds), you should use the .txt extension.
        annotations = mne.read_annotations(f"sub-{self.subject}_task-{self.task}_badannotations_.txt")
        assert annotations.onset.all() != 0.0  # Very important check to uncover the latent bug of mne.read_annotations()
        assert annotations.duration.all() != 0.0
        #assert annotations.description.all() == 'BAD_'
        self.raw.annotations.append(annotations.onset,annotations.duration,annotations.description)
        epochs_manual = mne.Epochs(self.raw,self.evts,self.evts_dict_stim,tmin=-0.1,tmax=1,reject_by_annotation=True)

        return epochs_manual

    def get_epochs_thresh(self):
        # specify rejection criterion for a peak-to-peak rejection method
        # the choice of bad channels/times affects all subsequent steps in the analysis pipeline
        # hence, create new epochs object based on the annotations
        # get epochs with and without rejection
        reject_criteria = dict(eeg=200e-6,       # 100 µV # HAD TO INCREASE IT HERE, 100 was too harsh
                               eog=200e-6)       # 200 µV
        epochs_thresh = mne.Epochs(self.raw,self.evts,self.evts_dict_stim,tmin=-0.1,tmax=1,reject=reject_criteria,reject_by_annotation=False)
        return epochs_thresh

    def get_epochs_autoreject(self):
        ar = AutoReject(verbose='tqdm')
        epochs  = mne.Epochs(self.raw,self.evts,self.evts_dict_stim,tmin=-0.1,tmax=1,reject_by_annotation=False)
        epochs.load_data()
        epochs_ar = ar.fit_transform(epochs) 
        #r = ar.get_reject_log(epochs)
        #r.plot(orientation="horizontal")
        return epochs_ar


if __name__ == '__main__':
    CleanSegments().run()
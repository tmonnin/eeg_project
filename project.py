from ccs_eeg_semesterproject import _get_filepath, load_precomputed_ica, add_ica_info, load_precomputed_badData
import ccs_eeg_utils
import os
from matplotlib import pyplot as plt
import mne
import numpy as np
import utils

from _00_filter import Filter
from _01_clean_channels import CleanChannels
from _02_clean_segments import CleanSegments
from _03_ica import ICA
from _04_reference import Reference

steps = (Filter,
         CleanChannels,
         CleanSegments,
         ICA,
         Reference
        )

for step in steps:
    step().run()

exit()

# Dataset N170: A face-viewing experiments, with an effect of faces at 170ms
raw = utils.load_data(task='N170', subject_id='001')
evts, evts_dict_stim = utils.load_annotations(raw)


### Create epochs from data TODO tmax?
epochs = mne.Epochs(raw,evts,evts_dict_stim,tmin=-0.1,tmax=1)
#epochs.average().plot()

target = epochs[["stimulus:{}{}".format(k,k) for k in [1,2,3,4,5]]].average()
distractor = epochs[["stimulus:{}{}".format(k,j) for k in [1,2,3,4,5] for j in [1,2,3,4,5] if k!=j]].average()
#mne.viz.plot_compare_evokeds([target,distractor])


########## Preprocessing
########## Filtering



######### re-referencening

######### Data cleaning: Time, channel and subjects



#bad_ix = [i for i,a in enumerate(annotations) if a['description']=="BAD_"]
#raw.annotations[bad_ix].save("sub-{}_task-P3_badannotations.csv".format(subject_id))
#annotations = mne.read_annotations("sub-{}_task-P3_badannotations.csv".format(subject_id))
raw.annotations.append(annotations.onset,annotations.duration,annotations.description)

######## ERP peak analysis 
# Extract the study-relevant ERP peak subjectwise (e.g. one value per subject) and statistically test them. 
# RQ: On which ERP-peaks do we find major difference between the conditions?
# Channels, times, peaks for N170: --> Rossion 2008, ERP Core Paper
# P7, PO7, P8, PO8; 130-200ms after stimulus onset
# tmin tmax -> start time before / end time after event
epochs = mne.Epochs(raw,evts,evts_dict_stim,tmin=-0.1,tmax=1)
#epochs = mne.Epochs(ica.get_sources(raw),evts,evts_dict_stim,tmin=-0.1,tmax=1)
#epochs.average(picks=).plot(picks='Oz')
#epochs.average().plot()
# get epochs with and without rejection
epochs        = mne.Epochs(raw,evts,evts_dict_stim,tmin=-0.1,tmax=1,reject_by_annotation=False)
#reject: Reject epochs based on peak-to-peak signal amplitude (PTP), i.e. the absolute difference between the lowest and the highest signal value. In each individual epoch, the PTP is calculated for every channel. If the PTP of any one channel exceeds the rejection threshold, the respective epoch will be dropped. The dictionary keys correspond to the different channel types; valid keys are: 'grad', 'mag', 'eeg', 'eog', and 'ecg'.
#reject_by_annotation: Whether to reject based on annotations. If True (default), epochs overlapping with segments whose description begins with 'bad' are rejected. If False, no rejection based on annotations is performed.
epochs_manual = mne.Epochs(raw,evts,evts_dict_stim,tmin=-0.1,tmax=1,reject_by_annotation=True)
# TODO eeg eog?
reject_criteria = dict(eeg=200e-6,       # 100 µV # HAD TO INCREASE IT HERE, 100 was too harsh
                       eog=200e-6)       # 200 µV
epochs_thresh = mne.Epochs(raw,evts,evts_dict_stim,tmin=-0.1,tmax=1,reject=reject_criteria,reject_by_annotation=False)
mne.viz.plot_compare_evokeds({'raw':epochs.average(),'clean':epochs_manual.average(),'thresh':epochs_thresh.average()},picks="P8")

#Choose 2 out of 4 (including statistics):
#Mass Univariate Use a multiple regression of the main experimental contrast, controlling for reaction time (you need to calculate RT yourself). RQ: When/Where do we find differences between our conditions? Is there an influence of reaction time?
#Decoding analysis Decode the main contrast of the experiment across time RQ: When is information about the conditions in our data available?
#Source space Use source localization to visualize the source of the main experimental contrast RQ: Where does our effect come from?
#Time Frequency analysis Calculate an induced time-frequency analysis of the main experimental contrast RQ: What oscillations underley our effect of interest?
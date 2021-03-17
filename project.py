from ccs_eeg_semesterproject import _get_filepath, load_precomputed_ica, add_ica_info, load_precomputed_badData
import ccs_eeg_utils
import os
from matplotlib import pyplot as plt
import mne
import numpy as np
import utils


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
# Remove and interpolate bad channels
select_bad_interactive = False
if select_bad_interactive:
    # Open interactive tool to select bad snippets
    raw.plot(n_channels=len(raw.ch_names))#,scalings =40e-6)
    plt.show()
    bad_ix = [i for i,a in enumerate(raw.annotations) if a['description']=="BAD_"]
    raw.annotations[bad_ix].save(f"sub-{subject_id}_task-{task}_badannotations.txt")
else:
    #https://mne.tools/dev/generated/mne.read_annotations.html
    #The annotations stored in a .csv require the onset columns to be timestamps. If you have onsets as floats (in seconds), you should use the .txt extension.
    annotations = mne.read_annotations(f"sub-{subject_id}_task-{task}_badannotations_.txt")
    assert annotations.onset.all() != 0.0  # Very important check to uncover the latent bug of mne.read_annotations()
    assert annotations.duration.all() != 0.0
    #assert annotations.description.all() == 'BAD_'
    raw.annotations.append(annotations.onset,annotations.duration,annotations.description)
    raw.plot(n_channels=len(raw.ch_names))#,scalings =40e-6)
    plt.show()

# select bad channels
# https://mne.tools/dev/auto_tutorials/preprocessing/plot_15_handling_bad_channels.html
# visualize averaged epochs
if True: #select_bad_interactive:
    epochs.average().plot()
# TODO spot bad channels in ICA
#raw.info['bads'] = ['FP2']
#picks = mne.pick_channels_regexp(raw.ch_names, regexp='MEG 2..3')
#raw.plot(order=picks, n_channels=len(picks))

# compare raw eeg and eeg without bad channels
good_eeg = mne.pick_types(raw.info, meg=False, eeg=True)#, exclude='bads')
all_eeg = mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
print(np.setdiff1d(all_eeg, good_eeg))
print(np.array(raw.ch_names)[np.setdiff1d(all_eeg, good_eeg)])

# interpolate bad times and channels
raw.set_montage('standard_1020',match_case=False)
raw.interpolate_bads()

# specify rejection criterion for a peak-to-peak rejection method

# the choice of bad channels/times affects all subsequent steps in the analysis pipeline
# hence, create new epochs object based on the annotations
# get epochs with and without rejection
epochs        = mne.Epochs(raw,evts,evts_dict_stim,tmin=-0.1,tmax=1,reject_by_annotation=False)
epochs_manual = mne.Epochs(raw,evts,evts_dict_stim,tmin=-0.1,tmax=1,reject_by_annotation=True)
reject_criteria = dict(eeg=200e-6,       # 100 µV # HAD TO INCREASE IT HERE, 100 was too harsh
                       eog=200e-6)       # 200 µV
epochs_thresh = mne.Epochs(raw,evts,evts_dict_stim,tmin=-0.1,tmax=1,reject=reject_criteria,reject_by_annotation=False)
# Compare different rejection techniques
#mne.viz.plot_compare_evokeds({'raw':epochs.average(),'clean':epochs_manual.average(),'thresh':epochs_thresh.average()},picks="Cz")

# TODO auto reject
from autoreject import AutoReject
ar = AutoReject(verbose='tqdm')
epochs.load_data()
#epochs_ar = ar.fit_transform(epochs) 
#r = ar.get_reject_log(epochs)
#r.plot(orientation="horizontal")
# TODO image epochs autoreject
#mne.viz.plot_compare_evokeds({
#    'raw':epochs.average(),
#    'clean':epochs_manual.average(),
#    'ar':epochs_ar.average()
#    },picks="Cz")

###### ICA
ica = mne.preprocessing.ICA(method="fastica")
ica.fit(raw,verbose=True)
ica.plot_components(range(10))
#ica.plot_properties(raw,picks=[0,1],psd_args={'fmax': 35.},reject=None)
# Estimate sources given the unmixing matrix.
#icaact = ica.get_sources(raw)
#plt.plot(icaact[5,0:20000][0].T)
#ica.plot_overlay(raw,exclude=[1,8,9])

#### Use preprocessed data
ica,badComps = load_precomputed_ica(filepath, subject_id)
ica = add_ica_info(raw,ica)
annotations,bad_ix = load_precomputed_badData(filepath, subject_id)
reconst_raw = raw.copy()
#Remove selected components from the signal.
ica.apply(reconst_raw,exclude=badComps)
raw.plot()
reconst_raw.plot() 
#ica.plot_overlay(raw,exclude=badComps)
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
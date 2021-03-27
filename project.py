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


#Choose 2 out of 4 (including statistics):
#Mass Univariate Use a multiple regression of the main experimental contrast, controlling for reaction time (you need to calculate RT yourself). RQ: When/Where do we find differences between our conditions? Is there an influence of reaction time?
#Decoding analysis Decode the main contrast of the experiment across time RQ: When is information about the conditions in our data available?
#Source space Use source localization to visualize the source of the main experimental contrast RQ: Where does our effect come from?
#Time Frequency analysis Calculate an induced time-frequency analysis of the main experimental contrast RQ: What oscillations underley our effect of interest?
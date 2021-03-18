
#Load the data
import mne
from mne_bids import (BIDSPath, read_raw_bids)
from ccs_eeg_semesterproject import _get_filepath, load_precomputed_ica, add_ica_info, load_precomputed_badData
from ccs_eeg_utils import read_annotations_core


def load_data(task, subject_id):
    # path where to save the datasets
    bids_root = f"../local/bids_project/{task.lower()}"
    bids_path, filepath = _get_filepath(bids_root, subject_id, task=task)
    raw = read_raw_bids(bids_path)
    # Required because the *channels.tsv file is not correctly loaded due to problem with naming convention
    raw.set_channel_types({'HEOG_left': 'eog', 'HEOG_right': 'eog',  'VEOG_lower': 'eog'})
    # fix the annotations readin
    read_annotations_core(bids_path, raw)
    raw.load_data()

    #print(raw.info)
        #bads: []
        #ch_names: FP1, F3, F7, FC3, C3, C5, P3, P7, P9, PO7, PO3, O1, Oz, Pz, CPz, ...
        #chs: 33 EEG
        #custom_ref_applied: False
        #highpass: 0.0 Hz
        #line_freq: 60
        #lowpass: 512.0 Hz
        #meas_date: unspecified
        #nchan: 33
        #projs: []
        #sfreq: 1024.0 Hz

    #print(dir(raw))
    #raw.ch_names
    #raw.n_times
    #len(raw.times)
    # 683008
    #print(raw.get_data().shape)
    #raw.to_data_frame().shape
    # (33, 683008)

    return raw

def load_annotations(raw):
    ### Annotations
    print(raw.annotations)
    # mne.Annotation has onset, duration, description
    #<Annotations | 642 segments: response:201 (314), response:202 (8), ...>
    evts,evts_dict = mne.events_from_annotations(raw)
    #evts
        #array([[ 29501,      0,      2],
        #       [ 33941,      0,      2],
        #       ...,
        #       [675339,      0,    101],
        #       [675770,      0,      1]])
    # evts_dict {'stimulus:1': 3, 'stimulus:10': 4, 'stimulus:101': 5, ...}
    #Used Annotations descriptions: ['response:201', 'response:202', 'stimulus:1', 'stimulus:10',
    wanted_keys = [e for e in evts_dict.keys() if not "response" in e]
    evts_dict_stim=dict((k, evts_dict[k]) for k in wanted_keys if k in evts_dict)
    #print(sorted(evts_dict_stim.values())) 
    #[3, 4, 5, .., 161, 162] --> event_ids
    #{'stimulus:1': 3, 'stimulus:10': 4, 'stimulus:101': 5}
    return evts, evts_dict_stim

def load_bad_channels(task, subject_id):
    bids_root = f"../local/bids_project/{task.lower()}"
    bad_segments, bad_channels = load_precomputed_badData(bids_root, subject_id, task)
    return bad_channels

def load_bad_segments(task, subject_id):
    bids_root = f"../local/bids_project/{task.lower()}"
    bad_segments, bad_ix = load_precomputed_badData(bids_root, subject_id, task)
    return bad_segments

def load_ica(task, subject_id):
    bids_root = f"../local/bids_project/{task.lower()}"
    ica, bad_comps = load_precomputed_ica(bids_root, subject_id, task)
    return ica, bad_comps

def ica_info(raw, ica):
    ica = add_ica_info(raw, ica)
    return ica

def misc():
    # Plot single channel
    #plt.plot(raw[10,1:2000][0].T)
    #plt.show()

    # Subselect individual channels
    raw_subselect = raw.copy().pick_channels(["Cz"])
    #plt.plot(raw_subselect[0,1:2000][0].T)


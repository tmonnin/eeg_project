#Choose 2 out of 4 (including statistics):
#Mass Univariate Use a multiple regression of the main experimental contrast, controlling for reaction time (you need to calculate RT yourself). RQ: When/Where do we find differences between our conditions? Is there an influence of reaction time?
#Decoding analysis Decode the main contrast of the experiment across time RQ: When is information about the conditions in our data available?
#Source space Use source localization to visualize the source of the main experimental contrast RQ: Where does our effect come from?
#Time Frequency analysis Calculate an induced time-frequency analysis of the main experimental contrast RQ: What oscillations underley our effect of interest?


import numpy as np
import scipy

import matplotlib
import matplotlib.pylab as pl
from matplotlib import pyplot as plt
#import seaborn as sns

import sklearn.pipeline
import sklearn
import sklearn.model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import mne.decoding

from config import fname
from base import Base
from _05_erp_peak_extraction import ErpPeakExtraction

class DecodingAnalysis(Base):

    def __init__(self):
        prev = ErpPeakExtraction()
        super().__init__(self.__class__.__name__.lower(), prev)

    def run(self):
        electrode = self.config["electrode"]
        for self.subject in self.config["subjects"]:#[1:6]:
            epochs = mne.read_epochs(fname.epochs(subject=self.subject))
            # Only consider epochs with faces or cars condition for classification
            epochs = epochs[["faces", "cars"]]
            #epochs = self.get_epochs_concat(self.config["subjects"])
            # TODO figure out time window that excludes response
            # TODO change to epochs_train
            #epochs = epochs.copy().crop(tmin=0.0, tmax=0.4)
            labels = self.get_labels(epochs)
            # TODO check picks
            #data = epochs.get_data(picks=["PO7", "PO8"]).mean(axis=2)
            #plt.scatter(data[:,0],data[:,1],color=np.array(["red","green"])[labels])


            csp = mne.decoding.CSP(n_components=2)
            try:
                csp.fit_transform(epochs.get_data(), labels)
                #print(csp)
                # CSP(component_order='mutual_info', cov_est='concat', cov_method_params=None,
                # log=None, n_components=2, norm_trace=False, rank=None, reg=None,
                # transform_into='average_power')
                csp_data = csp.transform(epochs.get_data())
                plt.scatter(csp_data[:,0],csp_data[:,1],color=np.array(["red","green"])[labels])
                #csp.plot_filters(epochs.info)
                #csp.plot_patterns(epochs.info)
                plt.show()
            except:
                pass

    @staticmethod
    def get_labels(epochs):
        events_dict_inv = {v: k for k, v in epochs.event_id.items()}
        # Assert uniqueness of dict values to allow inversion
        assert len(epochs.event_id) == len(events_dict_inv)

        evt_labels = epochs.events[:,-1]
        # Convert from evt_id to string, e.g. "faces/4"
        labels = [events_dict_inv[k].split("/")[0] for k in evt_labels]
        # Change to integer rerpresentation: faces == 0 and cars == 1
        labels = [int(l == "cars") for l in labels]
        labels = np.array(labels)
        print(labels)
        return labels

    @staticmethod
    def get_epochs_concat(subject_lst):
        epochs_lst = []
        for subject in subject_lst:
            epochs = mne.read_epochs(fname.epochs(subject=subject))
            epochs_lst.append(epochs[["faces", "cars"]])
        epochs = mne.concatenate_epochs(epochs_lst)
        return epochs


if __name__ == '__main__':
    DecodingAnalysis().run()
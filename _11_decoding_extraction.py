import itertools
import json
import numpy as np
import scipy

import matplotlib
import matplotlib.pylab as pl
from matplotlib import ticker, pyplot as plt

import sklearn.pipeline
import sklearn
import sklearn.model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import mne.decoding

from config import fname
from base import Base
from _10_erp_peak_extraction import ErpPeakExtraction

class DecodingExtraction(Base):

    def __init__(self):
        prev = ErpPeakExtraction()
        super().__init__(self.__class__.__name__.lower(), prev, section=("Pre-Analysis", "Decoding Extraction"))

    def process(self):
        # Decoding analysis Decode the main contrast of the experiment across time
        # RQ: When is information about the conditions in our data available?
        # https://mne.tools/stable/auto_tutorials/machine-learning/plot_sensors_decoding.html
        electrode = self.config["electrode"]
        epochs = mne.read_epochs(fname.epochs(subject=self.subject))
        # Only consider epochs with faces or cars condition for classification
        epochs = epochs[["faces", "cars"]]
        epochs = epochs.crop(tmin=-0.1, tmax=1.0)
        labels = self.get_labels(epochs)

        models = (("LDA", LinearDiscriminantAnalysis()),
                  ("LogisticRegression", sklearn.linear_model.LogisticRegression(solver="lbfgs", max_iter=500)),
                  ("SVM", sklearn.svm.LinearSVC()),
        )
        feature_space = (#("SPoC", mne.decoding.SPoC(n_components=2)), # PermissionError for temp file
                         #("CSP", mne.decoding.CSP(n_components=2, norm_trace=False)), # PermissionError for temp file
                         ("StandardScaler", sklearn.preprocessing.StandardScaler()),  # equals mne.decoding.Scaler(scalings='mean')
                         ("Vectorizer", mne.decoding.Vectorizer()),
        )
        fig, axs = plt.subplots(len(models), len(feature_space), constrained_layout=True, figsize=(16, 10))
        fig.suptitle("Decoding Analysis")
        scores_dict = {}
        for ax, ((model_name, model), (feature_space_name, feature_space)) in zip(axs.flatten(), itertools.product(models, feature_space)):
            pipe_simple = sklearn.pipeline.Pipeline([('feature_space', feature_space), ('model', model)])
            cv = sklearn.model_selection.StratifiedShuffleSplit(10, test_size=0.2, random_state=0)
            timeDecode = mne.decoding.SlidingEstimator(pipe_simple, scoring='roc_auc', n_jobs=0, verbose=True)
            scores = mne.decoding.cross_val_multiscore(timeDecode, epochs.get_data(), labels, cv=cv, n_jobs=0)
            scores = scores.mean(axis=0).tolist()
            times = epochs.times.tolist()
            peak_time = times[np.argmax(scores)]
            peak_score = np.max(scores)
            scores_dict[f"{feature_space_name}-{model_name}"] = {"times": times, "scores": scores}
            self.plot(ax, times, scores, model_name, feature_space_name, peak_time, peak_score)

        self.add_figure(figure=fig, caption="Comparison of decoding techniques")

        with open(fname.decoding_score(subject=self.subject), "w") as json_file:
            json.dump(scores_dict, json_file, indent=4)

    @staticmethod
    def plot(ax, x, y, model_name, feature_space_name, peak_time, peak_score):
        ax.axhline(0.5, color="lightcoral", label="chance")  # Horizontal line indicating chance (50%)
        ax.plot(x, y, label="score")
        ax.scatter(peak_time, peak_score, s=200, color='red', marker='x', linewidths=3)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("ROC_AUC")
        ax.set_xlim([0.0, 0.5])
        ax.set_ylim([0.3, 1.0])
        ax.set_title(f"{feature_space_name} - {model_name}")
        ax.legend()
        ax.grid(linestyle="dashed")
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))


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
    DecodingExtraction().run()

import json
import numpy as np
import matplotlib.pyplot as plt
import mne
from config import fname
from base import Base
from _06_decoding_peak_extraction import DecodingPeakExtraction

class DecodingPeakAnalysis(Base):

    def __init__(self):
        prev = DecodingPeakExtraction()
        super().__init__(self.__class__.__name__.lower(), prev)

    def run(self):
        # Decoding analysis Decode the main contrast of the experiment across time
        # RQ: When is information about the conditions in our data available?
        # https://mne.tools/stable/auto_tutorials/machine-learning/plot_sensors_decoding.html
        test_results = []
        for start_time in np.arange(self.config["t_start"], self.config["t_end"], self.config["t_sampling"]):
            #peaks = []
            data = []
            for self.subject in self.config["subjects"][0:15]:
                with open(fname.decodingpeak(subject=self.subject), "r") as json_file:
                    json_data = json.load(json_file)
                    json_data = json_data["StandardScaler-LogisticRegression"]
                    times, scores = json_data["times"], json_data["scores"]
                    time_idx = [i for i, x in enumerate(times) if x >= start_time and x < start_time + self.config["t_sampling"]]
                    score = np.mean(scores[time_idx[0]:time_idx[-1]+1])
                    #peaks.append([peak["peak_time"], peak["peak_alt"]])
                    data.append([score, 0.5])
            #peaks = np.array(peaks)
            data = np.array(data)
            #plt.hist(peaks[:,0])
            plt.hist(data)
            #plt.show()
            #mean_peak_time = np.mean(peaks[:,0])
            # Non-parametric cluster-level paired t-test
            # The first dimension should correspond to the difference between paired samples (observations) in two conditions.
            # https://mne.tools/dev/generated/mne.stats.permutation_cluster_1samp_test.html
            # TODO test is non-deterministic and stochastically distributed -> eval distribution parameters
            alpha = 0.05
            t_values, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(data)
            test_results.append([start_time, cluster_p_values[0]])
        #word = "not "*(p_value >= alpha)
        #print(f"Difference of ERP peak between face and car condition is {word}significant with alpha={alpha} and p-value={p_value}.")
        print(test_results)
if __name__ == '__main__':
    DecodingPeakAnalysis().run()
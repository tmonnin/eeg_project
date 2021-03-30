import json
import numpy as np
from matplotlib import ticker, pyplot as plt
import scipy
import mne
from config import fname
from base import Base
from _11_decoding_extraction import DecodingExtraction

class DecodingAnalysis(Base):

    def __init__(self):
        prev = DecodingExtraction()
        super().__init__(self.__class__.__name__.lower(), prev, section=("Analysis", "Decoding Analysis"))

    def run(self):
        # Decoding analysis Decode the main contrast of the experiment across time
        # RQ: When is information about the conditions in our data available?
        # https://mne.tools/stable/auto_tutorials/machine-learning/plot_sensors_decoding.html
        pvalues = []
        cohensds = []
        score_mean = []
        time_lst = []
        time_lst_plt = []
        for start_time in np.arange(self.config["t_start"], self.config["t_end"], self.config["t_sampling"]):
            time_lst += [start_time]
            time_lst_plt += [start_time, start_time + self.config["t_sampling"]]
            data = []
            for self.subject in self.config["subjects"]:
                with open(fname.decoding_score(subject=self.subject), "r") as json_file:
                    json_data = json.load(json_file)
                    json_data = json_data["StandardScaler-LogisticRegression"]
                    times, scores = json_data["times"], json_data["scores"]
                    time_idx = [i for i, x in enumerate(times) if x >= start_time and x < start_time + self.config["t_sampling"]]
                    score = np.mean(scores[time_idx[0]:time_idx[-1]+1])
                    data.append(score)
            score_mean += [np.mean(data)]
            alpha = 0.025
            statistics, pvalue = scipy.stats.ttest_1samp(data, 0.5, alternative='greater')
            pvalues.append(pvalue)
            # Calculate effect size with Cohan's d
            cohensds.append((np.mean(data) - 0.5) / (np.sqrt((np.std(data) ** 2 + 0 ** 2) / 2)))

        pvalues = np.array(pvalues)
        figure_decoding_analysis, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, constrained_layout=True, gridspec_kw={'height_ratios': [4, 6, 1, 1]}, figsize=(14, 8))
        ax1.plot(time_lst_plt, np.repeat(score_mean, 2), label="score")
        ax1.axhline(y=0.5, color="lightcoral", linestyle='--', label="chance")  # Horizontal line indicating chance (50%)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("ROC AUC Score")
        ax1.set_xlim([0.0, 1.0])
        ax1.set_title(f"Average decoding score across time")
        ax1.legend()
        ax1.grid(linestyle="dashed")
        ax1.set_axisbelow(True)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

        ax2.plot(time_lst_plt, np.repeat(pvalues, 2)*100, label="P-values")
        ax2.axhline(y=alpha*100, color="lightcoral", linestyle='--', label=f"alpha={alpha*100}%")  # Horizontal line indicating alpha
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("%")
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0*100, 1.0*100])
        ax2.set_title(f"P-value of t-test across time")
        ax2.legend()
        ax2.grid(linestyle="dashed")
        ax2.set_axisbelow(True)
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.05*100))

        ax3.pcolormesh(time_lst, [0,1], [pvalues < alpha], cmap="RdYlGn")
        ax3.set_xlabel("Time [s]")
        ax3.set_xlim([0.0, 1.0])
        ax3.yaxis.set_visible(False)
        ax3.set_title(f"Significance of t-test across time")
        ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

        ax4.pcolormesh(time_lst, [0,1], [np.clip(cohensds, a_min=0, a_max=3)], cmap="RdYlGn")
        ax4.set_xlabel("Time [s]")
        ax4.set_xlim([0.0, 1.0])
        ax4.yaxis.set_visible(False)
        ax4.set_title(f"Qualitative overview of effect size (Cohen's d) across time, d_max={np.max(cohensds):.3}, min_clipped at 0")
        ax4.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

        self.add_figure(figure_decoding_analysis, caption="Decoding score, significance and effect size across time")

        self.report(analysis=True)


if __name__ == '__main__':
    DecodingAnalysis().run()

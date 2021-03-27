import json
from config import fname
from base import Base
from _05_erp_peak_extraction import ErpPeakExtraction
class ErpPeakAnalysis(Base):

    def __init__(self):
        prev = ErpPeakExtraction()
        super().__init__(self.__class__.__name__.lower(), prev)

    def run(self):
        electrode = self.config["electrode"]
        peaks = {}
        for subject in self.config["subjects"]:
            with open(fname.erppeaks(subject=subject, electrode=electrode), "r") as json_file:
                peaks[subject] = json.load(json_file)
        with open(fname.erppeaks_combined(electrode=electrode), "w") as json_file:
            json.dump(peaks, json_file, indent=4)
        # TODO grand average
        # TODO t test

if __name__ == '__main__':
    ErpPeakAnalysis().run()
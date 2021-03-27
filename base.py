import argparse
import os
import numpy as np
import yaml
import mne
import utils
from config import fname


class Base:

    def __init__(self, filebase, prev):
        mne.set_log_level('INFO')
        self.filebase = filebase
        self.prev = prev
        self.load_config(self.filebase, path="config.yaml")
        self.raw = None
        self.figures = []

    def load_config(self, filename, path):
        with open(path) as f:
            self.config = yaml.safe_load(f)
            # Add step-specific params to root of dict
            self.config.update(self.config[filename])
            self.task = self.config["task"]

    def run(self):
        for self.subject in self.config["subjects_analysis"]:
            self.get_filename(self.subject)
            self.load()
            self.process()
            self.save()
            self.report()

    def get_filename(self, subject):
        self.config["subject"] = subject
        self.filename = fname.__dict__[self.filebase](**self.config)
        return self.filename

    def load(self):
        if self.prev is None:
            print(f"Load raw object because file not exists: {self.filename}")
            os.makedirs(fname.reports_dir, exist_ok=True)
            os.makedirs(fname.subject_dir(subject=self.subject), exist_ok=True)
            os.makedirs(fname.annotations_dir, exist_ok=True)
            os.makedirs(fname.results_dir, exist_ok=True)
            self.raw = utils.load_data(task=self.task, subject_id=self.subject)
        else:
            self.raw = mne.io.read_raw_fif(self.prev.get_filename(self.subject), preload=False)

        evts, evts_dict_all = utils.load_annotations(self.raw)
        events_dict_inv = {v: k for k, v in evts_dict_all.items()}
        # Assert uniqueness of dict values to allow inversion
        assert len(evts_dict_all) == len(events_dict_inv)
        code_response_wrong = self.config["dataset"]["response_wrong"][0]
        evt_id_response_wrong = evts_dict_all[f"response:{code_response_wrong}"]
        wrong_response_evts = []
        if evts[0][2] == evt_id_response_wrong:
            wrong_response_evts.append(0)
        # Check for all events if wrong response happened
        for i in range(0, len(evts)-1):
            if evts[i+1][2] == evt_id_response_wrong:
                prev_evt_id = evts[i][2]
                prev_evt_str = events_dict_inv[prev_evt_id]
                if prev_evt_str.startswith("stimulus"):
                    # Mark event with stimulus that caused wrong response
                    wrong_response_evts.append(i)
                # Mark event with wrong response
                wrong_response_evts.append(i+1)
        # Delete all events related to wrong response
        self.evts = np.delete(evts, wrong_response_evts, axis=0)
        # Create evts_dict from evts_dict_all by removing unused events
        self.evts_dict = dict((k,v) for k, v in evts_dict_all.items() if v in self.evts[:,2])
        # Create evts_dict_stim including only stimulus events for epoching
        wanted_keys = [e for e in self.evts_dict.keys() if not "response" in e]
        self.evts_dict_stim = dict((k, self.evts_dict[k]) for k in wanted_keys if k in self.evts_dict)
        # Ensure that no event with wrong response remains
        assert self.evts[:,2].all() != evt_id_response_wrong

    def process(self):
        raise NotImplementedError("Method process not implemented")

    def save(self):
        self.raw.save(self.filename, overwrite=True)

    def report(self):
        if not self.figures:
            raise Exception("Figures are not set")
        # Append PDF plots to report
        with mne.open_report(fname.report(subject=self.subject)) as report:
            for f in self.figures:
                report.add_figs_to_section(
                    f[0],
                    captions=f[1],
                    section=f[2],
                    replace=True
                )
            report.save(fname.report_html(subject=self.subject), overwrite=True,
                        open_browser=False)

    def add_figure(self, figure, caption, section):
        self.figures.append((figure, caption, section))

    @staticmethod
    def parse_args():
        # Handle command line arguments
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('subject', metavar='sub###', help='The subject to process')
        try:
            args = parser.parse_args()
            subject = args.subject
        except SystemExit:
            subject = "001"
        return subject

    def get_epochs(self, condition=None):
        # Epoching requires to have bad segments set
        bad_segments_set = any(description.startswith('BAD_') for description in self.raw.annotations.description)
        if not bad_segments_set:
            raise Exception("Bad segments have not yet added to annotations!")

        if condition == "face":  # Face stimulus
            wanted_codes = self.config["dataset"]["faces"]
        elif condition == "car":  # Car stimulus
            wanted_codes = self.config["dataset"]["cars"]
        elif condition is None:  # Both stimuli
            wanted_codes = self.config["dataset"]["faces"] + self.config["dataset"]["cars"]
        else:
            raise NotImplementedError(f"Condition {condition} not implemented")

        wanted_event_keys = [key for key in self.evts_dict.keys() if any(str(code) in key for code in wanted_codes)]
        evt_dict_stim = dict((k, self.evts_dict[k]) for k in wanted_event_keys if k in self.evts_dict)

        epochs = mne.Epochs(self.raw, self.evts, evt_dict_stim, tmin=-0.1, tmax=1, reject_by_annotation=True)

        return epochs

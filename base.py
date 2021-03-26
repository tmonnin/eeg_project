import argparse
import os
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
            self.raw = utils.load_data(task=self.task, subject_id=self.subject)
        else:
            self.raw = mne.io.read_raw_fif(self.prev.get_filename(self.subject), preload=False)

        self.evts, self.evts_dict_stim = utils.load_annotations(self.raw)

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

    def get_epochs(self):
        bad_segments_set = any(description.startswith('BAD_') for description in self.raw.annotations.description)
        if not bad_segments_set:
            raise Exception("Bad segments have not yet added to annotations!")
        epochs = mne.Epochs(self.raw,self.evts,self.evts_dict_stim,tmin=-0.1,tmax=1,reject_by_annotation=True)

        return epochs

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
        self.task = "N170"
        self.subjects, self.kwargs = self.load_config(self.filebase, path="config.yaml")
        self.raw = None
        self.figures = []

    def run(self):
        for self.subject in self.subjects:
            self.get_filename(self.subject)
            self.load()
            self.process()
            self.save()
            self.report()

    def get_filename(self, subject):
        self.kwargs["subject"] = subject
        self.filename = fname.__dict__[self.filebase](**self.kwargs)
        return self.filename

    def load(self):
        if self.prev is None:
            print(f"Load raw object because file not exists: {self.filename}")
            os.makedirs(fname.reports_dir, exist_ok=True)
            os.makedirs(fname.subject_dir(subject=self.subject), exist_ok=True)
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

    @staticmethod
    def load_config(filename, path):
        with open(path) as f:
            config = yaml.safe_load(f)
            subjects = config["subject_ids"]
            kwargs = config[filename]
        return subjects, kwargs

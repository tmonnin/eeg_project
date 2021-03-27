# Author of this configuration file: Marijn van Vliet, all rights are with the author
# Taken from https://github.com/AaltoImagingLanguage/conpy/blob/master/scripts/config.py

"""
===========
Config file
===========
Configuration parameters for the study.
"""

import os
from fnames import FileNames

path = os.path.dirname(os.path.abspath(__file__))

fname = FileNames()

# Filenames for directories
fname.add('path', path)
fname.add('archive_dir', '{path}/archive')
fname.add('subjects_dir', '{path}/subjects')
fname.add('subject_dir', '{subjects_dir}/{subject}')
fname.add('annotations_dir', '{path}/annotations')
fname.add('results_dir', '{path}/results')

# Filenames for data files
fname.add('filter', '{subject_dir}/filter-{fmin}-{fmax}.fif')
fname.add('cleanchannels', '{subject_dir}/cleanchannels-{bad_channels}.fif')
fname.add('cleansegments', '{subject_dir}/cleansegments-{strategy}.fif')
fname.add('ica', '{subject_dir}/ica-{fmin}.fif')
fname.add('reference', '{subject_dir}/reference.fif')
fname.add('erppeakanalysis', '{subject_dir}/erppeakanalysis-{electrode}.fif')
fname.add('erppeaks', '{results_dir}/erppeaks-{electrode}.json')


# Filenames for MNE reports
fname.add('reports_dir', '{path}/reports/')
fname.add('report', '{reports_dir}/{subject}-report.h5')
fname.add('report_html', '{reports_dir}/{subject}-report.html')

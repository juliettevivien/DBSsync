"""General definitions."""

# https://mne.tools/stable/glossary.html#term-data-channels
data_channel_types = ['mag', 'grad', 'eeg', 'csd', 'seeg', 'ecog', 'dbs' 'hbo',
                      'hbr', 'fnirs_cw_amplitude', 'fnirs_fd_ac_amplitude',
                      'fnirs_fd_phase', 'fnirs_od'] + ['data']

microvolts = ('microvolt', 'microvolts', 'uV', 'µV', 'μV')
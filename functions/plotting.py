"""
This module contains all the encessary functions to create the different plots 
in the GUI.

It includes the following functions:
- plot_channel_intra: Plot the selected channel data from the intracranial file.
- plot_channel_extra: Plot the selected channel data from the external file.
- plot_synced_channels: Plot the synchronized channels from both datasets to look at eventual timeshift
- plot_scatter_channel_intra: Plot scatter plot of the selected channel data to show where
    the artifact has been detected.
- plot_scatter_channel_external: Plot scatter plot of the selected channel data to show where
    the artifact has been detected.
- plot_overlapped_channels_ecg: Plot the synchronized channels: one intracranial channel
    to clean, and if available one external ECG channel to help for the R-peak detection
- plot_scatter_channel_intra_sf: Plot scatter plot of the selected channel data 
(for the correction of the sampling frequency).
- plot_scatter_channel_extra_sf: Plot scatter plot of the selected external 
channel data (for the correction of the sampling frequency).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy

matplotlib.use("Qt5Agg")

## set font sizes and other parameters for the figures
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def plot_channel_intra(self):
    """Plot the selected channel data from the intracranial file."""
    if self.dataset_intra.raw_data and self.dataset_intra.selected_channel_index is not None:
        self.canvas_intra.setEnabled(True)
        self.toolbar_intra.setEnabled(True)
        self.ax_intra.clear()
        channel_data = self.dataset_intra.raw_data.get_data()[
             self.dataset_intra.selected_channel_index
             ]
        times = self.dataset_intra.times
        self.ax_intra.plot(times, channel_data)
        self.ax_intra.set_title(
             f"Channel {self.dataset_intra.selected_channel_index} data - {self.dataset_intra.selected_channel_name}"
             )
        self.ax_intra.set_xlabel("Time (s)")
        self.ax_intra.set_ylabel("Amplitude")
        self.canvas_intra.draw()


def plot_channel_extra(self):
    """Plot the selected channel data from the external file."""
    if self.dataset_extra.raw_data and self.dataset_extra.selected_channel_index is not None:
        self.canvas_xdf.setEnabled(True)
        self.toolbar_xdf.setEnabled(True)
        self.ax_xdf.clear()
        channel_data = self.dataset_extra.raw_data.get_data()[
             self.dataset_extra.selected_channel_index
             ]
        times = self.dataset_extra.times
        # apply a high-pass filter to detrend the data if the channel to plot is a bipolar channel:
        if self.dataset_extra.selected_channel_name.startswith("BIP"):
            b, a = scipy.signal.butter(1, 0.05, "highpass")
            channel_data_to_plot = scipy.signal.filtfilt(b, a, channel_data)
        else:
            channel_data_to_plot = channel_data
        self.ax_xdf.plot(times, channel_data_to_plot)
        self.ax_xdf.set_title(
             f"Channel {self.dataset_extra.selected_channel_index} data - {self.dataset_extra.selected_channel_name}"
             )
        self.ax_xdf.set_xlabel("Time (s)")
        self.ax_xdf.set_ylabel("Amplitude")
        self.canvas_xdf.draw()


def plot_synced_channels(self):
    """Plot the synchronized channels from both datasets to look at eventual timeshift."""
    self.toolbar_synced.setEnabled(True)
    self.canvas_synced.setEnabled(True)
    self.ax_synced.clear()

    # scale y-axis to the same range for both channels by modifying the ylim
    # for the external channel:
    y_max_factor = self.dataset_intra.max_y_value / self.dataset_extra.max_y_value

    # Plot the external channel synchronized
    data_extra = self.dataset_extra.raw_data.get_data()[
         self.dataset_extra.selected_channel_index
         ]
    data_extra_scaled = data_extra * y_max_factor

    b, a = scipy.signal.butter(1, 0.05, "highpass")
    data_extra_detrended = scipy.signal.filtfilt(b, a, data_extra_scaled)

    timescale_extra = self.dataset_extra.times
    art_start_0_extra = self.dataset_extra.art_start - 1

    # Find the index in self.dataset_extra.times corresponding to art_start_0
    art_start_index_extra = (timescale_extra >= art_start_0_extra).argmax()
    offset_data_extra = data_extra_detrended[art_start_index_extra:]
    offset_timescale_extra = timescale_extra[art_start_index_extra:] - art_start_0_extra
    self.dataset_extra.reset_timescale = offset_timescale_extra
    self.dataset_extra.reset_data = offset_data_extra
    self.ax_synced.scatter(
        offset_timescale_extra, offset_data_extra, s=8, 
        color='#90EE90', label='External'
        )

    # Plot the intracranial channel synchronized
    data_intra = self.dataset_intra.raw_data.get_data()[
         self.dataset_intra.selected_channel_index
         ]
    timescale_intra = self.dataset_intra.times
    art_start_0_intra = self.dataset_intra.art_start - 1
    # Find the index in self.dataset_intra.times corresponding to art_start_0
    art_start_index_intra = (timescale_intra >= art_start_0_intra).argmax()
    offset_data_intra = data_intra[art_start_index_intra:]
    offset_timescale_intra = timescale_intra[art_start_index_intra:] - art_start_0_intra
    self.dataset_intra.reset_timescale = offset_timescale_intra
    self.dataset_intra.reset_data = offset_data_intra
    self.ax_synced.scatter(
         offset_timescale_intra, offset_data_intra, s=8, 
         color='#6495ED', label='Intracranial'
         )
    self.ax_synced.legend(loc='upper left')
    self.canvas_synced.draw()
    self.btn_select_last_art_intra.setEnabled(True)
    self.btn_select_last_art_xdf.setEnabled(True)


def plot_scatter_channel_intra(self, art_start_intra=None):
    """
    Plot scatter plot of the selected channel data to show where the 
    artifact has been detected.
    """
    self.toolbar_intra.setEnabled(True)
    self.canvas_intra.setEnabled(True)
    self.ax_intra.clear()
    
    # Plot the channel data
    channel_data = self.dataset_intra.raw_data.get_data()[
         self.dataset_intra.selected_channel_index
         ]
    times = self.dataset_intra.raw_data.times  # Time vector corresponding to the data points
    
    # Plot scatter points
    start = int(round(
              art_start_intra * self.dataset_intra.sf
              )-round(self.dataset_intra.sf/10)
              )
    end = int(round(
         art_start_intra * self.dataset_intra.sf
         )+round(self.dataset_intra.sf/10)
         )
    times_array = np.array(times)
    channel_data_array = np.array(channel_data)
    self.ax_intra.scatter(
         times_array[start:end], channel_data_array[start:end], s=5
         )

    # Highlight artifact start points if available
    if art_start_intra is not None:
            self.ax_intra.axvline(
                 x=art_start_intra, color='red', linestyle='--', 
                 label='Artifact Start'
                 )
    self.ax_intra.legend()
    # Allow interactive features like zoom and pan
    self.canvas_intra.draw()


def plot_scatter_channel_external(self, art_start_BIP=None):
    """Plot scatter plot of the selected channel data."""
    self.toolbar_xdf.setEnabled(True)
    self.canvas_xdf.setEnabled(True)
    self.ax_xdf.clear()

    # Plot the channel data
    channel_data = self.dataset_extra.raw_data.get_data()[
         self.dataset_extra.selected_channel_index
         ]
    b, a = scipy.signal.butter(1, 0.05, "highpass")
    channel_data_to_plot = scipy.signal.filtfilt(b, a, channel_data)
    times = self.dataset_extra.raw_data.times     

    # Plot scatter points
    start = int(round(
         art_start_BIP * self.dataset_extra.sf
         )-round(self.dataset_extra.sf/50))
    end = int(round(
         art_start_BIP * self.dataset_extra.sf
         )+round(self.dataset_extra.sf/50))
    times_array = np.array(times)
    channel_data_array = np.array(channel_data_to_plot)
    self.ax_xdf.scatter(
         times_array[start:end], 
         channel_data_array[start:end], s=5
         )
    # Highlight artifact start points if available
    if art_start_BIP is not None:
            self.ax_xdf.axvline(x=art_start_BIP, color='red', linestyle='--', label='Artifact Start')
    self.ax_xdf.legend()    
    self.canvas_xdf.draw()


def plot_overlapped_channels_ecg(self):
    """Plot the synchronized channels: one intracranial channel to clean,
    and if available one external ECG channel to help for the R-peak detection.
    """
    self.toolbar_overlapped.setEnabled(True)
    self.canvas_overlapped.setEnabled(True)
    self.ax_overlapped.clear()

    # Plot the external channel synchronized if one was selected:
    if self.dataset_extra.selected_channel_name_ecg is not None:
        data_extra = self.dataset_extra.synced_data.get_data()[
             self.dataset_extra.selected_channel_index_ecg
             ]

        # Apply 0.1 Hz-100Hz band-pass filter to ECG data
        b, a = scipy.signal.butter(1, 0.05, "highpass")
        detrended_data = scipy.signal.filtfilt(b, a, data_extra)
        low_cutoff = 100.0  # Hz
        b2, a2 = scipy.signal.butter(
            N=4,  # Filter order
            Wn=low_cutoff,
            btype="lowpass",
            fs=self.dataset_extra.sf 
        )
        ecg_data = scipy.signal.filtfilt(b2, a2, detrended_data)
        timescale_extra = np.linspace(
            0, 
            self.dataset_extra.synced_data.get_data().shape[1]/self.dataset_extra.sf, 
            self.dataset_extra.synced_data.get_data().shape[1]
            )

        self.ax_overlapped.plot(
             timescale_extra, ecg_data, color='#90EE90', 
             label='External ECG channel'
             )
        
    # Plot the intracranial channel 
    if self.config['NoSync'] == True:
        data_intra = self.dataset_intra.raw_data.get_data()[
         self.dataset_intra.selected_channel_index_ecg
         ]
        timescale_intra = np.linspace(
            0, 
            self.dataset_intra.raw_data.get_data().shape[1]/self.dataset_intra.sf, 
            self.dataset_intra.raw_data.get_data().shape[1]
            )
        self.ax_overlapped.plot(
            timescale_intra, data_intra, color='#6495ED', 
            label='Intracranial channel to clean'
            )
        self.ax_overlapped.legend(loc='upper left')
        self.canvas_overlapped.draw()
        self.btn_detect_r_peaks.setEnabled(True)
        self.combo_r_peak_threshold.setEnabled(True)
        self.btn_manual_override.setEnabled(True)

    else:
        data_intra = self.dataset_intra.synced_data.get_data()[
            self.dataset_intra.selected_channel_index_ecg
            ]
        timescale_intra = np.linspace(
            0, 
            self.dataset_intra.synced_data.get_data().shape[1]/self.dataset_intra.sf, 
            self.dataset_intra.synced_data.get_data().shape[1]
            )
        self.ax_overlapped.plot(
            timescale_intra, data_intra, color='#6495ED', 
            label='Intracranial channel to clean'
            )
        self.ax_overlapped.legend(loc='upper left')
        self.canvas_overlapped.draw()
        self.btn_detect_r_peaks.setEnabled(True)
        self.combo_r_peak_threshold.setEnabled(True)
        self.btn_manual_override.setEnabled(True)


def plot_scatter_channel_intra_sf(self):
    """Plot scatter plot of the selected channel data."""
    
    self.toolbar_intra_sf.setEnabled(True)
    self.canvas_intra_sf.setEnabled(True)
    self.ax_intra_sf.clear()
    data = self.dataset_intra.raw_data.get_data()[
         self.dataset_intra.selected_channel_index
         ]
    timescale = np.linspace(
         0, 
         self.dataset_intra.raw_data.get_data().shape[1]/self.dataset_intra.sf, 
         self.dataset_intra.raw_data.get_data().shape[1]
         )
    self.ax_intra_sf.scatter(timescale, data, s=8)
    self.canvas_intra_sf.draw()


def plot_scatter_channel_extra_sf(self):
    """Plot scatter plot of the selected external channel data."""
    self.toolbar_extra_sf.setEnabled(True)
    self.canvas_extra_sf.setEnabled(True)
    self.ax_extra_sf.clear()
    b, a = scipy.signal.butter(1, 0.05, "highpass")
    data = scipy.signal.filtfilt(
         b, a, 
         self.dataset_extra.raw_data.get_data()[
              self.dataset_extra.selected_channel_index
              ])
    timescale = self.dataset_extra.times
    self.ax_extra_sf.scatter(timescale, data, s=8)
    self.canvas_extra_sf.draw()

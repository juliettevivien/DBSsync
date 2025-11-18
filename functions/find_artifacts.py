"""
This module contains functions to detect artifacts in external and intracranial 
recordings. It includes methods for automatic detection based on signal 
characteristics and manual selection by the user.

The functions are designed to work with data from deep brain stimulation (DBS) 
recordings, specifically focusing on identifying artifacts caused by stimulation
pulses in both external (bipolar electrode) and intracranial (LFP) recordings.

The functions include:
- `detect_artifacts_external`: Automatically detects artifacts in external 
    recordings using the find_external_sync_artifact and plots it.
- `find_external_sync_artifact`: Finds artifacts in external recordings based on signal characteristics.
- `manual_selection_external`: Allows manual selection of artifacts in external recordings.
- `detect_artifacts_intra`: Automatically detects artifacts in intracranial recordings.
- `manual_selection_intra`: Allows manual selection of artifacts in intracranial recordings.
"""

import numpy as np
import scipy
from matplotlib.backend_bases import MouseButton

from functions.plotting import (
    plot_scatter_channel_intra, 
    plot_scatter_channel_external
    )


def detect_artifacts_external(self):
    """Detect artifacts in the bipolar external channel."""
    channel_data = self.dataset_extra.raw_data.get_data()[
        self.dataset_extra.selected_channel_index
        ]
    b, a = scipy.signal.butter(1, 0.05, "highpass")
    chan_data_detrend = scipy.signal.filtfilt(b, a, channel_data)        
    self.dataset_extra.art_start = find_external_sync_artifact(
        data=chan_data_detrend, sf_external=self.dataset_extra.sf, 
        times = self.dataset_extra.times
        )
    self.label_automatic_artifact_time_xdf.setText(
        f"Artifact start detected at: {self.dataset_extra.art_start} s"
        )
    self.label_manual_artifact_time_xdf.setText(
        "No artifact manually selected"
        )

    # Plot the channel with artifact
    plot_scatter_channel_external(self, art_start_BIP=self.dataset_extra.art_start)
    self.update_synchronize_button_state()  # Check if we can enable the button

def find_external_sync_artifact(
        data: np.ndarray, 
        sf_external: float, 
        times: np.ndarray
        ):
    """
    Function that finds artifacts caused by increasing/reducing
    stimulation from 0 to 1mA without ramp.
    For correct functioning, the external data recording should
    start in stim-off, and typically short pulses are given
    (without ramping). The first 2 seconds are used for threshold calculation
    and should therefore be free of any artifact.
    The signal are pre-processed previously with a high-pass
    Butterworth filter (1Hz) to ensure removal of slow drifts
    and offset around 0 (using _detrend_data function in utils.py).

    Inputs:
        - data: np.ndarray, single external channel (from bipolar electrode)
        - sf_external: float, sampling frequency of external recording
        - times: np.ndarray, timescale of the signal
    Returns:
        - art_time_BIP: the timestamp where the artifact starts in external recording (in seconds)
    """
    # check polarity of artifacts before detection:
    if abs(max(data[:-1000])) > abs(min(data[:-1000])):
        data = data * -1

    # find indexes of artifacts
    # the external sync artifact is a sharp deflection repeated at a high
    # frequency (stimulation frequency). Therefore, the artifact is detected when
    # the signal is crossing the threshold, and when the signal is lower than the
    # previous and next sample (first peak of the artifact).
    start_index = 0
    art_time_BIP = None
    while art_time_BIP == None:
        thresh_BIP = -1.5 * (np.ptp(data[start_index:(start_index + int(sf_external * 2))]))
        for q in range(start_index, len(data) - 2):
            if (
                (data[q] <= thresh_BIP)
                and (data[q] < data[q + 1])
                and (data[q] < data[q - 1])
            ):
                art_time_BIP = times[q]
                break
        start_index += 1*sf_external

    return art_time_BIP

def manual_selection_external(self):
    """Allow manual selection of artifacts in the bipolar external channel in
    case the automatic method fails."""
    self.toolbar_xdf.setEnabled(True)
    self.canvas_xdf.setEnabled(True)
    self.ax_xdf.clear()
    data = self.dataset_extra.raw_data.get_data()[
        self.dataset_extra.selected_channel_index
        ]
    b, a = scipy.signal.butter(1, 0.05, "highpass")
    channel_data_to_plot = scipy.signal.filtfilt(b, a, data)
    timescale = self.dataset_extra.times

    pos = []

    self.ax_xdf.scatter(timescale, channel_data_to_plot, s=8)
    self.canvas_xdf.draw()
    self.ax_xdf.set_title(
        'Right click on the plot to select the start of the artifact (shown by the black "+")'
    )

    (plus_symbol,) = self.ax_xdf.plot([], [], "k+", markersize=10)

    def onclick(event):
        if event.inaxes is not None:  # Check if the click is inside the axes
            if event.button == MouseButton.RIGHT:
                pos.append([event.xdata, event.ydata])

                # Update the position of the black "+" symbol
                closest_index_x = np.argmin(np.abs(timescale - event.xdata))
                closest_value_x = timescale[closest_index_x]
                closest_value_y = channel_data_to_plot[closest_index_x]
                plus_symbol.set_data([closest_value_x], [closest_value_y])
                self.canvas_xdf.draw()
                self.dataset_extra.art_start = closest_value_x
                self.label_manual_artifact_time_xdf.setText(
                    f"Selected Artifact start: {closest_value_x} s"
                    )
                self.label_automatic_artifact_time_xdf.setText(
                    "No artifact automatically detected"
                    )
                self.update_synchronize_button_state()

    self.canvas_xdf.mpl_connect("button_press_event", onclick)

def detect_artifacts_intra(self):
    """Detect artifacts in the intracranial LFP channel and plots it."""
    thres_window = round(self.dataset_intra.sf * 2)
    data = self.dataset_intra.raw_data.get_data()[
        self.dataset_intra.selected_channel_index
        ]
    thres = np.ptp(data[:thres_window])
    # Compute absolute value to be invariant to the polarity of the signal
    abs_data = np.abs(data)
    # Check where the data exceeds the threshold
    over_thres = np.where(abs_data[0:] > thres)[0][0]
    # Take last sample that lies within the value distribution of the thres_window 
    # before the threshold passing, and add 4 samples
    # The percentile is something that can be varied
    stim_idx = (np.where(
        abs_data[:over_thres] <= np.percentile(abs_data[:over_thres], 95)
    )[0][-1]) + 4
    self.dataset_intra.art_start = stim_idx / self.dataset_intra.sf
    plot_scatter_channel_intra(
        self, art_start_intra = self.dataset_intra.art_start
        )
    self.update_synchronize_button_state()  # Check if we can enable the button
    self.label_automatic_artifact_time_intra.setText(
        f"Artifact start: {self.dataset_intra.art_start} s"
        )
    self.label_manual_artifact_time_intra.setText(
        "No artifact manually selected"
        )

def manual_selection_intra(self):
    """Allow manual selection of artifacts in the intracranial LFP channel in
    case the automatic method fails."""
    # Enable the toolbar and canvas for intra-channel plotting
    self.toolbar_intra.setEnabled(True)
    self.canvas_intra.setEnabled(True)
    self.ax_intra.clear()
    data = self.dataset_intra.raw_data.get_data()[
        self.dataset_intra.selected_channel_index
        ]
    timescale = self.dataset_intra.times
    pos = []

    self.ax_intra.scatter(timescale, data, s=8)
    self.canvas_intra.draw()
    self.ax_intra.set_title(
        'Right click on the plot to select the start of the artifact (shown by the black "+")'
    )

    (plus_symbol,) = self.ax_intra.plot([], [], "k+", markersize=10)

    def onclick(event):
        if event.inaxes is not None:  # Check if the click is inside the axes
            if event.button == MouseButton.RIGHT:
                pos.append([event.xdata, event.ydata])

                # Update the position of the black "+" symbol
                closest_index_x = np.argmin(np.abs(timescale - event.xdata))
                closest_value_x = timescale[closest_index_x]
                closest_value_y = data[closest_index_x]
                plus_symbol.set_data([closest_value_x], [closest_value_y])
                self.canvas_intra.draw()
                self.dataset_intra.art_start = closest_value_x
                self.label_automatic_artifact_time_intra.setText(
                    "No artifact automatically detected"
                    )
                self.label_manual_artifact_time_intra.setText(
                    f"Selected Artifact start: {closest_value_x} s"
                    )
                self.update_synchronize_button_state()

    self.canvas_intra.mpl_connect("button_press_event", onclick)

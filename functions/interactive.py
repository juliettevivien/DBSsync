"""
This module contains functions to interactively select channels and artifacts
from intracranial and external recordings in a PyQt5 application. It allows users
to select channels from a list, choose artifacts by clicking on a plot, and apply
filters to the data. The selected channels and artifacts are then used for further
analysis or visualization.

The module includes the following functions:
    - `prompt_channel_name_intra`: Prompts the user to select a channel from
        the intracranial dataset (the one containing the synchronization pulses).
    - `select_channel_extra`: Opens a dialog to select a channel from the
        external dataset (the one containing the synchronization pulses).
    - `select_ecg_channel_to_compute_hr_external`: Opens a dialog to select an
        ECG channel from the external dataset for heart rate computation. The 
        heart rate will be displayed in the GUI.
    - `select_last_artifact_intra`: Allows the user to select the start of the
        last artifact in the intracranial recording by right-clicking on the plot.
    - `select_last_artifact_extra`: Allows the user to select the start of the
        last artifact in the external recording by right-clicking on the plot.
    - `choose_int_channel_for_cleaning`: Prompts the user to select a channel
        from the intracranial dataset for ECG cleaning.
    - `choose_ext_channel_for_cleaning`: Prompts the user to select a channel
        from the external dataset for ECG cleaning.
    - `validate_filtering`: Applies a low-pass filter to the left and right
        channels before cleaning, based on the user input in the filtering option box.
        

"""


import numpy as np
import matplotlib
import scipy

matplotlib.use("Qt5Agg")

from PyQt5.QtWidgets import (
    QVBoxLayout, QListWidget, QPushButton, QInputDialog, QListWidget, 
    QMessageBox, QPushButton, QVBoxLayout, QWidget
)
from matplotlib.backend_bases import MouseButton




def prompt_channel_name_intra(self):
    """Prompt for channel name selection for intracranial file, the channel 
    containing the synchronization pulses."""
    if self.dataset_intra.raw_data:
        try:
            channel_names = self.dataset_intra.ch_names  # List of channel names
            channel_name, ok = QInputDialog.getItem(
                self, "Channel Selection", "Select a channel:", channel_names, 
                0, False
                )

            if ok and channel_name:  # Check if a channel was selected
                self.dataset_intra.selected_channel_name = channel_name
                self.dataset_intra.selected_channel_index = channel_names.index(
                    channel_name
                    )  # Get the index of the selected channel
                self.channel_label_intra.setText(
                    f"Selected Channel: {channel_name}"
                    )
                self.dataset_intra.max_y_value = np.nanmax(
                    self.dataset_intra.raw_data.get_data()[
                        self.dataset_intra.selected_channel_index
                        ])
                # Enable the plot button since a channel has been selected
                self.btn_plot_channel_intra.setEnabled(True)
                self.btn_artifact_detect_intra.setEnabled(True)    
                self.label_automatic_artifact_time_intra.setVisible(True)   
                self.btn_manual_select_artifact_intra.setEnabled(True)
                self.label_manual_artifact_time_intra.setVisible(True)
                

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to select channel: {e}")


def select_channel_extra(self):
    """Open a dialog to select a channel by name from the external file (the 
    one containing the synchronization pulses)."""
    if self.dataset_extra.raw_data is not None:
        channel_names = self.dataset_extra.ch_names  # Get the channel names
        dialog = QWidget()
        dialog.setWindowTitle("Select Channel")
        layout = QVBoxLayout(dialog)

        # Create a list widget to display channel names
        channel_list = QListWidget(dialog)
        channel_list.addItems(channel_names)  # Add channel names to the list
        layout.addWidget(channel_list)

        # Create OK and Cancel buttons
        ok_button = QPushButton("OK", dialog)
        cancel_button = QPushButton("Cancel", dialog)
        layout.addWidget(ok_button)
        layout.addWidget(cancel_button)

        # Define button actions
        def on_ok():
            selected_items = channel_list.selectedItems()
            if selected_items:
                self.dataset_extra.selected_channel_name = selected_items[0].text()  # Get the selected channel name
                self.dataset_extra.selected_channel_index = channel_names.index(
                    self.dataset_extra.selected_channel_name
                    )  # Get the index
                channel_data = self.dataset_extra.raw_data.get_data()[
                    self.dataset_extra.selected_channel_index
                    ]
                b, a = scipy.signal.butter(1, 0.05, "highpass")
                detrended_data = scipy.signal.filtfilt(b, a, channel_data)
                self.dataset_extra.min_y_value = detrended_data.min()
                self.dataset_extra.max_y_value = detrended_data.max()
                self.channel_label_xdf.setText(f"Selected Channel: {self.dataset_extra.selected_channel_name}")
                self.btn_plot_channel_xdf.setEnabled(True)
                dialog.close()
                self.btn_artifact_detect_xdf.setEnabled(True)
                self.label_automatic_artifact_time_xdf.setVisible(True)
                self.btn_manual_select_artifact_xdf.setEnabled(True)
                self.label_manual_artifact_time_xdf.setVisible(True)
                self.update_synchronize_button_state()  # Check if we can enable the button

        def on_cancel():
            dialog.close()

        ok_button.clicked.connect(on_ok)
        cancel_button.clicked.connect(on_cancel)

        dialog.setLayout(layout)
        dialog.show()


def select_ecg_channel_to_compute_hr_external(self):
    """Open a dialog to select an external ECG channel by name from the external file."""
    if self.dataset_extra.raw_data is not None:
        channel_names = self.dataset_extra.ch_names  # Get the channel names
        dialog = QWidget()
        dialog.setWindowTitle("Select ECG channel")
        layout = QVBoxLayout(dialog)

        # Create a list widget to display channel names
        channel_list = QListWidget(dialog)
        channel_list.addItems(channel_names)  # Add channel names to the list
        layout.addWidget(channel_list)

        # Create OK and Cancel buttons
        ok_button = QPushButton("OK", dialog)
        cancel_button = QPushButton("Cancel", dialog)
        layout.addWidget(ok_button)
        layout.addWidget(cancel_button)

        # Define button actions
        def on_ok():
            selected_items = channel_list.selectedItems()
            if selected_items:
                self.dataset_extra.ecg_selected_channel_name = selected_items[0].text()  # Get the selected channel name
                self.dataset_extra.ecg_selected_channel_index = channel_names.index(
                    self.dataset_extra.ecg_selected_channel_name)  # Get the index
                channel_data = self.dataset_extra.raw_data.get_data()[
                    self.dataset_extra.ecg_selected_channel_index
                    ]
                # Apply 0.1 Hz-100Hz band-pass filter to ECG data
                b, a = scipy.signal.butter(1, 0.05, "highpass")
                detrended_data = scipy.signal.filtfilt(b, a, channel_data)
                low_cutoff = 100.0  # Hz
                b2, a2 = scipy.signal.butter(
                    N=4,  # Filter order
                    Wn=low_cutoff,
                    btype="lowpass",
                    fs=self.dataset_extra.sf 
                )
                ecg_data = scipy.signal.filtfilt(b2, a2, detrended_data)

                # Remove the first and last 30 seconds 
                # (to avoid sync pulses while computing the threshold):
                start = int(30*self.dataset_extra.sf)
                end = int((self.dataset_extra.times[-1] - 30)*self.dataset_extra.sf)
                ecg_data_cropped = ecg_data[start:end]
                threshold = np.percentile(ecg_data_cropped, 95)
                
                # calculate heart rate based on the ecg_data channel:
                heartbeats = scipy.signal.find_peaks(
                    ecg_data, height=threshold, distance = self.dataset_extra.sf // 2
                    )
                hr = len(heartbeats[0]) / (
                    len(ecg_data) / self.dataset_extra.sf
                    ) * 60  # in bpm

                if not 55 <= hr <= 120: # if the cardiac artifact polarity is reversed, it doesn't find peaks, so the signal should be reversed before computing again
                    # calculate heart rate based on the ecg_data channel:
                    threshold = np.percentile(- ecg_data_cropped, 95)
                    heartbeats = scipy.signal.find_peaks(
                        - ecg_data, height=threshold, 
                        distance = self.dataset_extra.sf // 2
                        )
                    hr = len(heartbeats[0]) / (
                        len(ecg_data) / self.dataset_extra.sf
                        ) * 60  # in bpm

                self.ecg_channel_label.setText(f"Heart rate: {hr} bpm")
                dialog.close()


        def on_cancel():
            dialog.close()

        ok_button.clicked.connect(on_ok)
        cancel_button.clicked.connect(on_cancel)

        dialog.setLayout(layout)
        dialog.show()



def select_last_artifact_intra(self):
    """
    This function is used to compute the effective sampling frequency.
    It allows the user to select the start of the last artifact in the intracranial recording
    by right-clicking on the plot. The selected position is marked with a black "+" symbol
    and the corresponding time is stored in the dataset.
    """
    # Check if we're already in external selection mode and prevent interference
    if hasattr(self, 'cid_extra') and self.cid_extra is not None:
        self.canvas_synced.mpl_disconnect(self.cid_extra)
        self.cid_extra = None

    pos_intra = []

    self.ax_synced.set_title(
        'Right click on the plot to select the start of the last artifact in the intracranial recording (shown by the black "+")'
    )
    self.canvas_synced.draw()

    # Create or update the intracranial "+" symbol
    if not hasattr(self, 'plus_symbol_intra'):
        self.plus_symbol_intra, = self.ax_synced.plot([], [], "k+", markersize=10)

    # Disconnect previous intracranial event listener, if any
    if hasattr(self, 'cid_intra') and self.cid_intra is not None:
        self.canvas_synced.mpl_disconnect(self.cid_intra)

    # Define the click handler for the intracranial selection
    def onclick(event_intra):
        if event_intra.inaxes is not None:  # Check if the click is inside the axes
            if event_intra.button == MouseButton.RIGHT:
                pos_intra.append([event_intra.xdata, event_intra.ydata])

                closest_index_x_intra = np.argmin(np.abs(
                    self.dataset_intra.reset_timescale - event_intra.xdata
                    ))
                closest_value_x_intra = self.dataset_intra.reset_timescale[
                    closest_index_x_intra
                    ]
                closest_value_y_intra = self.dataset_intra.reset_data[
                    closest_index_x_intra
                    ]
                self.plus_symbol_intra.set_data(
                    [closest_value_x_intra], [closest_value_y_intra]
                    )
                self.canvas_synced.draw()

                self.dataset_intra.last_artifact = closest_value_x_intra
                self.label_select_last_art_intra.setText(
                    f"Selected last artifact start: {self.dataset_intra.last_artifact} s"
                    )
                self.update_timeshift_button_state()

    # Connect and store the ID of the intracranial event listener
    self.cid_intra = self.canvas_synced.mpl_connect("button_press_event", onclick)
    

def select_last_artifact_extra(self):
    """
    This function is used to compute the effective sampling frequency.
    This function allows the user to select the start of the last artifact in the
    external recording by right-clicking on the plot. The selected position is marked
    with a red "+" symbol and the corresponding time is stored in the dataset.
    """
    # Check if we're already in intracranial selection mode and prevent interference
    if hasattr(self, 'cid_intra') and self.cid_intra is not None:
        self.canvas_synced.mpl_disconnect(self.cid_intra)
        self.cid_intra = None

    pos_extra = []

    self.ax_synced.set_title(
        'Right click on the plot to select the start of the last artifact in the external recording (shown by the red "+")'
    )
    self.canvas_synced.draw()

    # Create or update the external "+" symbol
    if not hasattr(self, 'plus_symbol_extra'):
        self.plus_symbol_extra, = self.ax_synced.plot([], [], "r+", markersize=10)

    # Disconnect previous external event listener, if any
    if hasattr(self, 'cid_extra') and self.cid_extra is not None:
        self.canvas_synced.mpl_disconnect(self.cid_extra)

    # Define the click handler for the external selection
    def onclick(event_extra):
        if event_extra.inaxes is not None:  # Check if the click is inside the axes
            if event_extra.button == MouseButton.RIGHT:
                pos_extra.append([event_extra.xdata, event_extra.ydata])

                closest_index_x_extra = np.argmin(np.abs(
                    self.dataset_extra.reset_timescale - event_extra.xdata
                    ))
                closest_value_x_extra = self.dataset_extra.reset_timescale[
                    closest_index_x_extra
                    ]
                closest_value_y_extra = self.dataset_extra.reset_data[
                    closest_index_x_extra
                    ]
                self.plus_symbol_extra.set_data(
                    [closest_value_x_extra], [closest_value_y_extra]
                    )
                self.canvas_synced.draw()

                self.dataset_extra.last_artifact = closest_value_x_extra
                self.label_select_last_art_xdf.setText(
                    f"Selected last artifact start: {self.dataset_extra.last_artifact} s"
                    )
                self.update_timeshift_button_state()
    # Connect and store the ID of the external event listener
    self.cid_extra = self.canvas_synced.mpl_connect("button_press_event", onclick)
    

def choose_int_channel_for_cleaning(self):
    """Prompt for channel name selection in intracranial file for ecg cleaning."""
    # if self.config['NoSync'] == True:
    #     channel_names = self.dataset_intra.ch_names

    # else:
    try:
    # if self.dataset_intra.synced_data:
        channel_names = self.dataset_intra.ch_names  # List of channel names
        channel_name, ok = QInputDialog.getItem(
            self, "Channel Selection", "Select a channel:", channel_names,
            0, False
            )

        if ok and channel_name:  # Check if a channel was selected
            self.dataset_intra.selected_channel_name_ecg = channel_name
            self.dataset_intra.selected_channel_index_ecg = channel_names.index(
                channel_name
                )  # Get the index of the selected channel
            self.label_selected_int_channel.setText(
                f"Selected Channel: {channel_name}"
                )    
            self.btn_confirm_and_plot_channels.setEnabled(True)    
            self.box_filtering_option.setEnabled(True)
            self.btn_validate_filtering.setEnabled(True)
                
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to select channel: {e}")

def choose_ext_channel_for_cleaning(self):
    """Prompt for channel name selection for external ECG channel"""
    try:
        if self.dataset_extra.synced_data:
            channel_names = self.dataset_extra.ch_names  # List of channel names
            channel_name, ok = QInputDialog.getItem(
                self, "Channel Selection", "Select a channel:", channel_names, 
                0, False
                )

            if ok and channel_name:  # Check if a channel was selected
                self.dataset_extra.selected_channel_name_ecg = channel_name
                self.dataset_extra.selected_channel_index_ecg = channel_names.index(
                    channel_name
                    )  # Get the index of the selected channel
                self.label_selected_ext_channel.setText(
                    f"Selected Channel: {channel_name}"
                    )                
                self.btn_confirm_and_plot_channels.setEnabled(True)   
                    
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to select channel: {e}")


def validate_filtering(self):
    """Apply a low-pass filter to the left and right channels before cleaning,
    based on the user input in the box_filtering_option."""
    try:    
        if self.box_filtering_option.text() != "":
            if self.config['NoSync'] == True:
                # Extract the original channel before you overwrite it
                raw_new_left = self.dataset_intra.raw_data.copy().pick_channels(
                    [self.dataset_intra.raw_data.ch_names[0]]
                    )
                raw_new_left.rename_channels(
                    {raw_new_left.ch_names[0]: 'RAW_Left_STN'}
                    )
                raw_new_right = self.dataset_intra.raw_data.copy().pick_channels(
                    [self.dataset_intra.raw_data.ch_names[1]]
                    )
                raw_new_right.rename_channels(
                    {raw_new_right.ch_names[0]: 'RAW_Right_STN'}
                    )

                # Add channels to existing Raw object
                self.dataset_intra.raw_data.add_channels([raw_new_left, raw_new_right])

                h_freq = float(self.box_filtering_option.text())
                self.dataset_intra.raw_data.filter(
                    l_freq=None, h_freq=h_freq, picks=[
                        self.dataset_intra.raw_data.ch_names[
                            0], self.dataset_intra.raw_data.ch_names[1]
                            ])
                QMessageBox.information(
                    self, "Filtering", 
                    f"Filtering applied to both left and right channels with cutoff frequency: {h_freq} Hz"
                    )
                
            else:
                # Extract the original channel before you overwrite it
                raw_new_left = self.dataset_intra.synced_data.copy().pick_channels(
                    [self.dataset_intra.synced_data.ch_names[0]]
                    )
                raw_new_left.rename_channels(
                    {raw_new_left.ch_names[0]: 'RAW_Left_STN'}
                    )
                raw_new_right = self.dataset_intra.synced_data.copy().pick_channels(
                    [self.dataset_intra.synced_data.ch_names[1]]
                    )
                raw_new_right.rename_channels(
                    {raw_new_right.ch_names[0]: 'RAW_Right_STN'}
                    )

                # Add channels to existing Raw object
                self.dataset_intra.synced_data.add_channels([raw_new_left, raw_new_right])

                h_freq = float(self.box_filtering_option.text())
                self.dataset_intra.synced_data.filter(
                    l_freq=None, h_freq=h_freq, picks=[
                        self.dataset_intra.synced_data.ch_names[
                            0], self.dataset_intra.synced_data.ch_names[1]
                            ])
                QMessageBox.information(
                    self, "Filtering", 
                    f"Filtering applied to both left and right channels with cutoff frequency: {h_freq} Hz"
                    )

    except ValueError as e:
        QMessageBox.warning(self, "Invalid Input, please enter an integer", str(e))


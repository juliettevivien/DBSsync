"""
This module contains functions for cleaning ECG data using various methods as 
described in Stam et al., 2023.

There are three main methods implemented:
1. Interpolation Method: This method uses simple interpolation to remove artifacts
    from the ECG signal, i.e. linearly interpolated over the R-peaks found.
2. Template Subtraction Method: This method creates a QRS template based on the 
    R-peaks found and subtracts it from the ECG signal, using a linear fit to adjust
    the template to the raw data.
3. Singular Value Decomposition (SVD) Method: This method uses SVD to decompose 
    the raw contaminated signal into components, allowing for the removal of noise 
    and artifacts by selecting the most significant components.

The detection of R-peaks is done using either only the LFP channel or an
addidtional external ECG channel synchronized with the LFP channel. 
The R-peaks are detected with the function scipy.signal.find_peaks, with
a threshold-based at 95th percentile of the signal amplitude, and a minimum
distance of 500 ms between peaks. The polarity of the R-peaks is determined
by comparing the mean amplitude of the detected peaks in both orientations
(positive and negative). The orientation with the higher mean absolute amplitude
is chosen as the orientation of the QRS complexes.

Reference paper:
Stam M.J., van Wijk B.C.M., Sharma P., Beudel M., Piña-Fuentes D.A., 
de Bie R.M.A., Schuurman P.R., Neumann W.J., Buijink A.W.G. (2023) 
A comparison of methods to suppress electrocardiographic artifacts in local 
field potential recordings. Clin Neurophysiol. doi: 10.1016/j.clinph.2022.11.011

This module contains the following functions:
- `find_r_peaks`: Finds R-peaks in the LFP channel using either only the LFP channel itself,
  or an additional external ECG channel synchronized with the LFP channel.
- `find_r_peaks_based_on_ext_ecg`: Finds R-peaks in the LFP channel based on an external ECG channel.
- `find_r_peaks_in_lfp_channel`: Finds R-peaks in the LFP channel using the LFP channel itself.
- `start_ecg_cleaning_interpolation`: Starts the ECG cleaning process using the interpolation method.
- `clean_ecg_interpolation`: Cleans the ECG signal using the interpolation method.
- `start_ecg_cleaning_template_sub`: Starts the ECG cleaning process using the template subtraction method.
- `clean_ecg_template_sub`: Cleans the ECG signal using the template subtraction method.
- `start_ecg_cleaning_svd`: Starts the ECG cleaning process using the SVD method.
- `clean_ecg_svd`: Cleans the ECG signal using the SVD method.
"""

    #######################################################################
    #                         ECG CLEANING FUNCTIONS                      #
    #######################################################################

from PyQt5.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QLabel, QLineEdit, QComboBox, QHBoxLayout, QPushButton
import numpy as np
import scipy
import scipy.signal
from scipy.signal import find_peaks
import mne

from functions.utils import get_start_end_times, find_similar_sample
from functions.classes import PlotWindow


def manual_override(self):
    # --- Create dialog ---
    dialog = QDialog()
    dialog.setWindowTitle("Set R-peak Detection Parameters")

    layout = QVBoxLayout()

    # --- Combo box for polarity ---
    polarity_layout = QHBoxLayout()
    polarity_label = QLabel("R-peak polarity (LFP):")
    combo_polarity = QComboBox()
    combo_polarity.addItems(["None", "Down", "Up"])
    combo_polarity.setCurrentText("None")
    polarity_layout.addWidget(polarity_label)
    polarity_layout.addWidget(combo_polarity)
    layout.addLayout(polarity_layout)

    # --- Line edits for start/end cleaning times ---
    start_layout = QHBoxLayout()
    start_label = QLabel("Start cleaning time (s):")
    start_edit = QLineEdit()
    start_edit.setPlaceholderText("None")
    start_layout.addWidget(start_label)
    start_layout.addWidget(start_edit)
    layout.addLayout(start_layout)

    end_layout = QHBoxLayout()
    end_label = QLabel("End cleaning time (s):")
    end_edit = QLineEdit()
    end_edit.setPlaceholderText("None")
    end_layout.addWidget(end_label)
    end_layout.addWidget(end_edit)
    layout.addLayout(end_layout)

    # --- OK / Cancel buttons ---
    button_layout = QHBoxLayout()
    ok_button = QPushButton("OK")
    cancel_button = QPushButton("Cancel")
    button_layout.addWidget(ok_button)
    button_layout.addWidget(cancel_button)
    layout.addLayout(button_layout)

    dialog.setLayout(layout)

    # --- Button connections ---
    def on_ok():
        try:
            r_peak_polarity_lfp = combo_polarity.currentText()
            if r_peak_polarity_lfp == "None":
                r_peak_polarity_lfp = None

            start_text = start_edit.text().strip()
            start_cleaning_time = float(start_text) if start_text else None

            end_text = end_edit.text().strip()
            end_cleaning_time = float(end_text) if end_text else None

            # store as instance attributes or process immediately
            self.r_peak_polarity_lfp = r_peak_polarity_lfp
            self.start_cleaning_time = start_cleaning_time
            self.end_cleaning_time = end_cleaning_time

            dialog.accept()
        except ValueError:
            QMessageBox.warning(dialog, "Invalid Input", "Please enter valid numbers for times.")
    
    ok_button.clicked.connect(on_ok)
    cancel_button.clicked.connect(dialog.reject)

    # --- Show dialog ---
    if dialog.exec_() == QDialog.Accepted:
        print(f"Polarity: {self.r_peak_polarity_lfp}, "
            f"Start: {self.start_cleaning_time}, End: {self.end_cleaning_time}")
    else:
        return  # User canceled

##################################################################

    #######################################################################
    #########                 FINDING THE R-PEAKS                 #########
    #######################################################################   

def find_r_peaks(self):
    """
    Find R-peaks in the LFP channel using either only the LFP channel itself,
    or an additional external ECG channel synchronized with the LFP channel.
    This function will be called when the user clicks the "Find R-peaks" button.
    """

    full_data = self.dataset_intra.synced_data.get_data()[
        self.dataset_intra.selected_channel_index_ecg
        ]
    times = np.linspace(
        0, 
        self.dataset_intra.synced_data.get_data().shape[1]/self.dataset_intra.sf, 
        self.dataset_intra.synced_data.get_data().shape[1]
        )
    # use the detection threshold set by the user, or 95 as default:
    detection_threshold = int(self.combo_r_peak_threshold.currentText() or 95)

    # # Get additional option to override some parameters if needed:
    # # open a pop-up window to get user input for polarity and time range
    # dialog = QDialog(self)
    # dialog.setWindowTitle("Override Parameters")
    # dialog.setGeometry(100, 100, 300, 200)

    # # Create layout and widgets for the dialog
    # layout = QVBoxLayout(dialog)    
    # label_info = QLabel("You can override the default parameters for R-peak detection below.\n"
    #                     "Leave fields empty to use default values.", dialog)
    # layout.addWidget(label_info)
    # label_polarity = QLabel("R-peak Polarity in LFP Channel (Up/Down):", dialog)
    # layout.addWidget(label_polarity)
    # combo_polarity = QComboBox(dialog)
    # combo_polarity.addItems(["Up", "Down"])
    # layout.addWidget(combo_polarity)
    # label_start_time = QLabel("Start Time for Cleaning (seconds):", dialog)
    # layout.addWidget(label_start_time)
    # input_start_time = QLineEdit(dialog)
    # layout.addWidget(input_start_time)
    # label_end_time = QLabel("End Time for Cleaning (seconds):", dialog)
    # layout.addWidget(label_end_time)
    # input_end_time = QLineEdit(dialog)
    # layout.addWidget(input_end_time)

    # r_peak_polarity_lfp = str(combo_polarity.currentText()) or None
    # start_cleaning_time = float(input_start_time.text()) or None
    # end_cleaning_time = float(input_end_time.text()) or None


    if self.dataset_extra.selected_channel_name_ecg is not None:
        # Use external ECG channel to find R-peaks
        final_peaks, polarity, mean_epoch = find_r_peaks_based_on_ext_ecg(
            self, full_data, times, detection_threshold,
            window_artifact = [-0.5, 0.5]
            )
        QMessageBox.information(
            self,
            "R-Peak Detection",
            f"R-peaks have been detected in the LFP channel using the ECG channel provided.",
            QMessageBox.Ok
        )    
    else:
        # Add a message window to inform the user about the method and threshold used:
        QMessageBox.information(
            self,
            "R-Peak Detection",
            f"R-peaks have been detected in the LFP channel alone, using a threshold of {detection_threshold}%.",
            QMessageBox.Ok
        )

        final_peaks, polarity, mean_epoch = find_r_peaks_in_lfp_channel(
            self, full_data, times, detection_threshold,
            window = [-0.5, 0.5]
            ) 
    
    # Check if peak detection was successful
    if len(final_peaks) == 0:
        print("Peak detection failed - no peaks found")
        # Disable cleaning buttons since we don't have valid peaks
        self.btn_start_ecg_cleaning_interpolation.setEnabled(False)
        self.btn_start_ecg_cleaning_template_sub.setEnabled(False)
        self.btn_start_ecg_cleaning_svd.setEnabled(False)
        return
        
    self.final_peaks = final_peaks
    self.polarity = polarity
    self.mean_epoch = mean_epoch

    self.btn_start_ecg_cleaning_interpolation.setEnabled(True)
    self.btn_start_ecg_cleaning_template_sub.setEnabled(True)
    self.btn_start_ecg_cleaning_svd.setEnabled(True)

    # reset params after use 
    self.r_peak_polarity_lfp = None
    self.start_cleaning_time = None
    self.end_cleaning_time = None

    print("R-peaks found and stored in the main window.")



def find_r_peaks_based_on_ext_ecg(
        self, 
        full_data: np.ndarray, 
        times: np.ndarray, 
        detection_threshold: int,
        window_artifact: list = [-0.5, 0.5]
        ):
    #### PREDETERMINE R-PEAKS TIMESTAMPS USING ECG CHANNEL ####
    """ 
    The externally recorded ECG signal is used to predetermine
    the timestamps of the R-peaks. The ECG signal is z-scored ((x-l)/r) 
    over the entire recording and the function findpeaks was used to search 
    for R-peaks with a specific height (95th percentile) and at a 
    specific inter-peak distance (minimally 500 ms). 
    The algorithm accounts for negative QRS complexes
    by repeating this procedure after multiplying the signal with
    -1. For both orientations of the LFP signal, the values of the
    peaks were averaged and the peaks with the highest mean
    determined the orientation of the QRS complexes.
    """
    last_peak_start, first_peak_end = get_start_end_times(full_data, times)
    
    # print(f"Debug ext ECG: LFP data length: {len(full_data)}")
    # print(f"Debug ext ECG: LFP time range: {times[0]:.2f} to {times[-1]:.2f} seconds")
    # print(f"Debug ext ECG: Crop range: {last_peak_start:.2f} to {first_peak_end:.2f} seconds")

    # Override with user-defined times if provided
    if self.start_cleaning_time is not None:
        last_peak_start = self.start_cleaning_time
    if self.end_cleaning_time is not None:
        first_peak_end = self.end_cleaning_time

    # Validate crop range (same as in find_r_peaks_in_lfp_channel)
    if last_peak_start >= first_peak_end:
        print(f"Warning ext ECG: Invalid crop range detected! Start ({last_peak_start:.2f}) >= End ({first_peak_end:.2f})")
        print("Using full signal instead of cropping...")
        # Use a reasonable portion of the signal (skip first and last 10 seconds)
        last_peak_start = max(10.0, times[0] + 10.0)
        first_peak_end = min(times[-1] - 10.0, times[-1] - 10.0)
        print(f"Debug ext ECG: Adjusted crop range: {last_peak_start:.2f} to {first_peak_end:.2f} seconds")
    
    data_extra = self.dataset_extra.synced_data.get_data()[
        self.dataset_extra.selected_channel_index_ecg
        ]

    # Apply 0.1 Hz-100Hz band-pass filter to ECG data
    b, a = scipy.signal.butter(1, 0.05, "highpass")
    detrended_data = scipy.signal.filtfilt(b, a, data_extra)
    low_cutoff = 60.0  # Hz
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

    # Z-score the ECG signal
    ecg_z = (ecg_data - np.mean(ecg_data)) / np.std(ecg_data)

    # Define peak detection params in the ECG channel
    threshold = np.percentile(ecg_z, detection_threshold)  # Convert to percentile for robustness
    min_distance_samples = int(0.5 * self.dataset_extra.sf)  # 500 ms in samples

    # Detect peaks in original signal
    peaks_pos, _ = scipy.signal.find_peaks(
        ecg_z,
        height=threshold,
        distance=min_distance_samples
    )

    # Detect peaks in inverted signal
    peaks_neg, __ = scipy.signal.find_peaks(
        -ecg_z,
        height=threshold,
        distance=min_distance_samples
    )

    # Select better polarity based on the number of peaks detected
    if len(peaks_pos) >= len(peaks_neg):
        chosen_peaks = peaks_pos
        polarity_ecg = 'Positive'
    else:
        chosen_peaks = peaks_neg
        polarity_ecg = 'Negative'
    
    # Check if any peaks were detected in the external ECG
    if len(chosen_peaks) == 0:
        print("No peaks detected in external ECG! Trying more lenient parameters...")
        
        # Try with lower thresholds
        for fallback_threshold in [90, 85, 80, 75, 70]:
            threshold_fallback = np.percentile(ecg_z, fallback_threshold)
            
            peaks_pos_fb, _ = scipy.signal.find_peaks(
                ecg_z, height=threshold_fallback, distance=min_distance_samples
            )
            peaks_neg_fb, _ = scipy.signal.find_peaks(
                -ecg_z, height=threshold_fallback, distance=min_distance_samples
            )
            
            print(f" Trying {fallback_threshold}% threshold: {len(peaks_pos_fb)} pos, {len(peaks_neg_fb)} neg peaks")
            
            if len(peaks_pos_fb) > 0 or len(peaks_neg_fb) > 0:
                chosen_peaks = peaks_pos_fb if len(peaks_pos_fb) >= len(peaks_neg_fb) else peaks_neg_fb
                polarity_ecg = 'Positive' if len(peaks_pos_fb) >= len(peaks_neg_fb) else 'Negative'
                print(f"  Success with {fallback_threshold}% threshold: {len(chosen_peaks)} peaks found ({polarity_ecg})")
                break
        
        # If still no peaks found
        if len(chosen_peaks) == 0:
            print("No peaks detected in external ECG with any threshold!")
            QMessageBox.warning(
                self,
                "External ECG Peak Detection Failed",
                "No R-peaks could be detected in the external ECG channel. Please check if:\n"
                "1. The correct ECG channel is selected\n"
                "2. The ECG signal quality is sufficient\n"
                "3. The synchronization between recordings is correct",
                QMessageBox.Ok
            )
            return np.array([]), 'Unknown', np.array([])
    
    print(f"External ECG: Found {len(chosen_peaks)} peaks with {polarity_ecg} polarity")     

    # Plot the detected peaks ####
    self.canvas_detected_peaks.setEnabled(True)
    self.toolbar_detected_peaks.setEnabled(True)
    self.ax_detected_peaks.clear()
    self.ax_detected_peaks.set_title('Detected Peaks')
    self.ax_detected_peaks.plot(timescale_extra, ecg_data, label='Raw ECG', alpha=0.1)
    self.ax_detected_peaks.plot(timescale_extra[chosen_peaks], ecg_data[chosen_peaks], 'ro', label='Detected Peaks', alpha=0.1)
    self.canvas_detected_peaks.draw()

    # Convert R-peaks from ECG samples to seconds
    r_peak_times_sec = chosen_peaks / self.dataset_extra.sf

    # Convert times to LFP sample indices
    r_peaks_lfp_idx = np.round(r_peak_times_sec * self.dataset_intra.sf).astype(int)

    window_around_peaks = 20  # ±20 LFP samples
    max_peaks = []
    min_peaks = []

    for idx in r_peaks_lfp_idx:
        start = max(idx - window_around_peaks, 0)
        end = min(idx + window_around_peaks + 1, len(full_data))
        segment = full_data[start:end]

        if len(segment) > 0:
            max_peaks.append(np.max(segment))
            min_peaks.append(np.min(segment))

    # Calculate mean absolute values
    mean_abs_max = np.nanmean(np.abs(max_peaks)) if max_peaks else 0
    mean_abs_min = np.nanmean(np.abs(min_peaks)) if min_peaks else 0

    # Override polarity if user specified
    if self.r_peak_polarity_lfp is not None:
        polarity = self.r_peak_polarity_lfp
        print(f"Overriding detected polarity to user-specified: {polarity}")
        if polarity == 'Up':
            mean_abs_max = 2
            mean_abs_min = 1
        else:
            mean_abs_max = 1
            mean_abs_min = 2

    # Choose the orientation with the higher mean absolute amplitude in the LFP channel
    lfp_peak_indices = []
    polarity = None
    if mean_abs_max >= mean_abs_min:
        polarity = 'Up'
        for idx in r_peaks_lfp_idx:
            start = idx - window_around_peaks
            end = idx + window_around_peaks + 1
            # Check signal boundaries
            if start < 0 or end > len(full_data):
                continue
            segment = full_data[start:end]
            if np.isnan(segment).any():
                continue
            local_max_idx = np.argmax(segment)
            peak_global_idx = start + local_max_idx
            lfp_peak_indices.append(peak_global_idx)
    else:
        polarity = 'Down'
        for idx in r_peaks_lfp_idx:
            start = idx - window_around_peaks
            end = idx + window_around_peaks + 1
            # Check signal boundaries
            if start < 0 or end > len(full_data):
                continue
            segment = full_data[start:end]
            if np.isnan(segment).any():
                continue
            local_min_idx = np.argmin(segment)
            peak_global_idx = start + local_min_idx
            lfp_peak_indices.append(peak_global_idx)

    # Remove peaks that are before last_peak_start and after first_peak_end:
    initial_lfp_count = len(lfp_peak_indices)
    lfp_peak_indices = [
        p for p in lfp_peak_indices if (
            p >= int(last_peak_start * self.dataset_intra.sf) and 
            p <= int(first_peak_end * self.dataset_intra.sf)
        )]
    
    print(f"LFP peaks: {initial_lfp_count} total, {len(lfp_peak_indices)} after time filtering")
    
    # Check if we have sufficient LFP peaks
    if len(lfp_peak_indices) == 0:
        print("No LFP peaks remain after time filtering!")
        QMessageBox.warning(
            self,
            "LFP Peak Detection Failed",
            "No valid R-peaks found in the LFP channel after applying time constraints. "
            "This might be due to timing issues between the external ECG and LFP recordings.",
            QMessageBox.Ok
        )
        return np.array([]), polarity, np.array([])

    # Plot the detected peaks
    self.ax_detected_peaks.plot(times, full_data, label='Raw LFP', color='black')
    self.ax_detected_peaks.plot(
        np.array(times)[lfp_peak_indices], 
        np.array(full_data)[lfp_peak_indices], 
        'ro', label='LFP Peaks'
        )
    self.ax_detected_peaks.legend()
    self.canvas_detected_peaks.draw()

    # Estimate HR
    peak_intervals = np.diff(lfp_peak_indices) / self.dataset_intra.sf  # Convert to seconds
    hr = 60 / np.mean(peak_intervals) if len(peak_intervals) > 0 else 0
    self.label_heart_rate_lfp.setText(f'Heart rate: {hr:.1f} bpm')
    print(f"Estimated heart rate from LFP: {hr:.1f} bpm")

    # Define epoch window
    sf_lfp = self.dataset_intra.sf
    pre_samples = int(abs(window_artifact[0]) * sf_lfp)
    post_samples = int(window_artifact[1] * sf_lfp)
    epoch_length = pre_samples + post_samples  # Total length of each epoch
    time = np.linspace(window_artifact[0], window_artifact[1], epoch_length)  # Time in seconds

    epochs = []  # Store extracted heartbeats

    for peak in lfp_peak_indices:
        start = peak - pre_samples
        end = peak + post_samples

        if (
            start >= last_peak_start*self.dataset_intra.sf
            ) and (
                end < first_peak_end*self.dataset_intra.sf
            ):  # Ensure we don't take the peaks that are in the stimulation pulses
            epochs.append(full_data[start:end])

    epochs = np.array(epochs)
    
    # Check if we have valid epochs
    if len(epochs) == 0:
        print("No valid epochs extracted from LFP peaks!")
        QMessageBox.warning(
            self,
            "LFP Epoch Extraction Failed",
            "No valid epochs could be extracted from the detected LFP peaks. "
            "This might be due to peaks being too close to signal boundaries.",
            QMessageBox.Ok
        )
        return np.array([]), polarity, np.array([])

    # Compute average heartbeat template
    mean_epoch = np.nanmean(epochs, axis=0)
    
    # Additional check for mean_epoch validity
    if np.isnan(mean_epoch).all() or len(mean_epoch) == 0:
        print("LFP mean epoch is invalid (all NaN or empty)!")
        QMessageBox.warning(
            self,
            "LFP Template Creation Failed",
            "Could not create a valid ECG template from the detected LFP peaks.",
            QMessageBox.Ok
        )
        return np.array([]), polarity, np.array([])
    
    print(f"LFP: Created template from {len(epochs)} valid epochs")

    # # Plot the detected ECG epochs
    # self.canvas_ecg_artifact.setEnabled(True)
    # self.toolbar_ecg_artifact.setEnabled(True)
    # self.ax_ecg_artifact.clear()
    # self.ax_ecg_artifact.set_title("Detected ECG epochs")

    # for epoch in epochs:
    #     self.ax_ecg_artifact.plot(time, epoch, color='gray', alpha=0.3)

    # self.ax_ecg_artifact.plot(
    #     time, 
    #     mean_epoch, 
    #     color='black', 
    #     linewidth=2, 
    #     label='Average ECG Template'
    #     )
    # self.ax_ecg_artifact.set_xlabel("Time (s)")
    # self.ax_ecg_artifact.set_ylabel("Amplitude")
    # self.ax_ecg_artifact.legend()
    # self.canvas_ecg_artifact.draw()


    return lfp_peak_indices, polarity, mean_epoch



def find_r_peaks_in_lfp_channel(
        self, 
        full_data: np.ndarray, 
        times: np.ndarray, 
        detection_threshold: int, 
        window = [-0.5, 0.5]
        ):
    sf_lfp = round(self.dataset_intra.sf)

    last_peak_start, first_peak_end = get_start_end_times(full_data, times)
    
    # Override with user-defined times if provided
    if self.start_cleaning_time is not None:
        last_peak_start = self.start_cleaning_time
    if self.end_cleaning_time is not None:
        first_peak_end = self.end_cleaning_time
    # print(f"Debug: Data length: {len(full_data)}, Sampling frequency: {sf_lfp}")
    # print(f"Debug: Time range: {times[0]:.2f} to {times[-1]:.2f} seconds")
    # print(f"Debug: Crop range: {last_peak_start:.2f} to {first_peak_end:.2f} seconds")
    
    # Validate crop range
    if last_peak_start >= first_peak_end:
        print(f"Warning: Invalid crop range detected! Start ({last_peak_start:.2f}) >= End ({first_peak_end:.2f})")
        print("Using full signal instead of cropping...")
        # Use a reasonable portion of the signal (skip first and last 10 seconds)
        last_peak_start = max(10.0, times[0] + 10.0)
        first_peak_end = min(times[-1] - 10.0, times[-1] - 10.0)
        print(f"Debug: Adjusted crop range: {last_peak_start:.2f} to {first_peak_end:.2f} seconds")
    
    # Calculate crop indices and validate
    start_idx = int(last_peak_start * sf_lfp)
    end_idx = int(first_peak_end * sf_lfp)
    
    # Ensure indices are within bounds
    start_idx = max(0, min(start_idx, len(full_data) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(full_data)))
    
    cropped_data = full_data[start_idx:end_idx]
    
    # Check if cropped data is valid
    if len(cropped_data) == 0:
        print("Error: Cropped data is empty! Using full signal...")
        cropped_data = full_data
        last_peak_start = times[0]
        first_peak_end = times[-1]
    
    print(f"Debug: Cropped data length: {len(cropped_data)}, Range: {np.min(cropped_data):.4f} to {np.max(cropped_data):.4f}")
    print(f"Debug: Data std: {np.std(cropped_data):.4f}, Detection threshold: {detection_threshold}%")
    
    #beginning_part = full_data[:int(last_peak_start * sf_lfp)]
    #end_part = full_data[int(first_peak_end * sf_lfp):]

    ecg = {'proc': {}}
    ns = len(cropped_data)  # Number of samples in the cropped data
    
    # Segment the signal into overlapping windows
    dwindow = int(round(sf_lfp))  # 1s window
    dmove = sf_lfp  # 1s step
    n_segments = (ns - dwindow) // dmove + 1
    
    #print(f"Debug: Processing {n_segments} segments of {dwindow} samples each")
    
    detected_peaks_positive = []  # Store peak indices in the original timescale of the cropped_data
    x = np.array(
        [
            cropped_data[
                i * dmove: i * dmove + dwindow
                ] for i in range(n_segments) if i * dmove + dwindow <= ns
        ])    
    # Loop through each segment and find peaks
    for i in range(n_segments):
        segment = x[i]
        # Skip segment if it contains any NaNs
        if np.isnan(segment).any():
            continue
        
        # Try different thresholds if 90th percentile fails
        for threshold_pct in [80, 70, 60, 50]:
            peaks, _ = scipy.signal.find_peaks(
                segment, height=np.percentile(segment, threshold_pct), distance=sf_lfp//3
                )
            if len(peaks) > 0:
                break
        
        real_peaks = peaks + (i * dmove)  # Convert to original timescale
        detected_peaks_positive.extend(real_peaks)

    # Repeat with reverted signal to find negative peaks
    detected_peaks_negative = []  # Store peak indices in the original timescale of the cropped_data
    x_neg = np.array(
        [
            -cropped_data[
                i * dmove: i * dmove + dwindow
                ] for i in range(n_segments) if i * dmove + dwindow <= ns
        ])
    for i in range(n_segments):
        segment = x_neg[i]
        # 6. Skip segment if it contains any NaNs
        if np.isnan(segment).any():
            continue
        
        # Try different thresholds if 90th percentile fails
        for threshold_pct in [80, 70, 60, 50]:
            peaks, _ = scipy.signal.find_peaks(
                segment, height=np.percentile(segment, threshold_pct), distance=sf_lfp//3
                )
            if len(peaks) > 0:
                break
        
        real_peaks = peaks + (i * dmove)  # Convert to original timescale
        detected_peaks_negative.extend(real_peaks)

    # Find which set of peaks has more elements
    print(f"Debug: Found {len(detected_peaks_positive)} positive peaks and {len(detected_peaks_negative)} negative peaks")
    
    if len(detected_peaks_positive) >= len(detected_peaks_negative):
        detected_peaks = detected_peaks_positive
        polarity = 'Up'
    else:
        detected_peaks = detected_peaks_negative
        polarity = 'Down'

    # Override polarity if user specified
    if self.r_peak_polarity_lfp is not None:
        polarity = self.r_peak_polarity_lfp
        print(f"Overriding detected polarity to user-specified: {polarity}")
        if polarity == 'Up':
            detected_peaks = detected_peaks_positive
        else:
            detected_peaks = detected_peaks_negative

    detected_peaks = np.array(detected_peaks)
    
    # Check if any peaks were detected
    if len(detected_peaks) == 0:
        print("No peaks detected with segmented approach! Trying fallback method...")
        
        # Try the simpler fallback method
        detected_peaks, polarity = simple_peak_detection_fallback(cropped_data, sf_lfp, detection_threshold)
        
        if len(detected_peaks) == 0:
            print("No peaks detected with any method!")
            QMessageBox.warning(
                self,
                "Peak Detection Failed",
                "No R-peaks could be detected in the signal with any method. Please check if:\n"
                "1. The signal contains visible ECG artifacts\n"
                "2. The channel selection is correct\n"
                "3. The signal amplitude is sufficient\n"
                "4. Try adjusting the threshold percentage",
                QMessageBox.Ok
            )
            # Return empty arrays to prevent crashes
            return np.array([]), 'Unknown', np.array([])

    print(f"Detected {len(detected_peaks)} peaks with {polarity} polarity")

    # Define epoch window
    pre_samples = int(abs(window[0]) * sf_lfp)
    post_samples = int(window[1] * sf_lfp)
    epoch_length = pre_samples + post_samples  # Total length of each epoch
    #time = np.linspace(window[0], window[1], epoch_length)  # Time in seconds

    epochs = []  # Store extracted heartbeats
    for peak in detected_peaks:
        start = peak - pre_samples
        end = peak + post_samples
        
        if start >= 0 and end < ns:  # Ensure we don't go out of bounds
            epoch = cropped_data[start:end]
            if np.isnan(epoch).any():
                continue
            else:
                epochs.append(epoch)

    epochs = np.array(epochs)
    
    # Check if we have valid epochs
    if len(epochs) == 0:
        print("No valid epochs extracted from detected peaks!")
        QMessageBox.warning(
            self,
            "Epoch Extraction Failed",
            "No valid epochs could be extracted from the detected peaks. This might be due to peaks being too close to signal boundaries.",
            QMessageBox.Ok
        )
        # Return empty arrays to prevent crashes
        return np.array([]), polarity, np.array([])

    # Compute average heartbeat template
    mean_epoch = np.nanmean(epochs, axis=0)
    
    # Additional check for mean_epoch validity
    if np.isnan(mean_epoch).all() or len(mean_epoch) == 0:
        print("Mean epoch is invalid (all NaN or empty)!")
        QMessageBox.warning(
            self,
            "Template Creation Failed",
            "Could not create a valid ECG template from the detected peaks.",
            QMessageBox.Ok
        )
        return np.array([]), polarity, np.array([])

    print(f"Created template from {len(epochs)} valid epochs")

    # Temporal correlation for ECG detection
    # adapt in case NaNs are present:
    if np.isnan(cropped_data).any():
        cropped_data_clean = np.nan_to_num(cropped_data, nan=0.0)
        r = np.correlate(cropped_data_clean, mean_epoch, mode='same')
    else:
        r = np.correlate(cropped_data, mean_epoch, mode='same')
    threshold = np.percentile(r, 95)
    detected_peaks, _ = scipy.signal.find_peaks(
        r, height=threshold, distance=sf_lfp//2
        )

    # Second pass for refining detection
    refined_template = np.nanmean(
        [cropped_data[
            p - dwindow//2 : p + dwindow//2
            ] for p in detected_peaks if p - dwindow//2 > 0 and p + dwindow//2 < ns
            ], axis=0
            )

    if np.isnan(cropped_data).any():
        cropped_data_clean = np.nan_to_num(cropped_data, nan=0.0)
        r2 = np.correlate(cropped_data_clean, refined_template, mode='same')  
    else:  
        r2 = np.correlate(cropped_data, refined_template, mode='same')
    threshold2 = np.percentile(r2, detection_threshold)
    final_peaks, _ = scipy.signal.find_peaks(
        r2, height=threshold2, distance=sf_lfp//2
        )

    # Adjust the final peaks to the original data scale
    final_peaks = final_peaks + start_idx

    # plot the detected peaks
    self.canvas_detected_peaks.setEnabled(True)
    self.toolbar_detected_peaks.setEnabled(True)
    self.ax_detected_peaks.clear()
    self.ax_detected_peaks.set_title('Detected Peaks')
    self.ax_detected_peaks.plot(full_data, label='Raw Channel')
    self.ax_detected_peaks.plot(
        final_peaks, full_data[final_peaks], 'ro', label='Detected Peaks'
        )
    self.canvas_detected_peaks.draw()

    # Estimate HR
    peak_intervals = np.diff(final_peaks) / sf_lfp  # Convert to seconds
    hr = 60 / np.mean(peak_intervals) if len(peak_intervals) > 0 else 0
    self.label_heart_rate_lfp.setText(f'Heart rate: {hr} bpm')


    return final_peaks, polarity, mean_epoch



    #######################################################################
    #########                  INTERPOLATION METHOD               #########
    #######################################################################        



def start_ecg_cleaning_interpolation(self):
    self.ax_ecg_clean.clear()
    self.ax_ecg_artifact.clear()
    self.ax_psd.clear()
    """Start the ECG cleaning process using the interpolation method from Perceive toolbox."""
    try:
        clean_ecg_interpolation(self)

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to clean ECG: {e}")



def clean_ecg_interpolation(self):
    full_data = self.dataset_intra.synced_data.get_data()[
        self.dataset_intra.selected_channel_index_ecg
        ]
    times = np.linspace(
        0, 
        self.dataset_intra.synced_data.get_data().shape[1]/self.dataset_intra.sf, 
        self.dataset_intra.synced_data.get_data().shape[1]
        )

    ############################################################################
    # prepare a copy of the full data to store the cleaned data
    clean_data = np.copy(full_data) 
    ns = len(full_data)

    #### INTERPOLATE DATA AT EACH R-PEAK FOUND ####
    # Remove artifacts (simple interpolation)
    for p in self.final_peaks:
        clean_data[max(0, p - 5): min(ns, p + 5)] = np.nan  # NaN out artifacts
    clean_data = np.interp(
        np.arange(ns), np.arange(ns)[~np.isnan(clean_data)], 
        clean_data[~np.isnan(clean_data)]
        )

    if self.dataset_intra.selected_channel_index_ecg == 0:
        self.dataset_intra.cleaned_ecg_left = clean_data
        print("Left channel cleaned")

    elif self.dataset_intra.selected_channel_index_ecg == 1:
        self.dataset_intra.cleaned_ecg_right = clean_data
        print("Right channel cleaned")

    # plot an overlap of the raw and cleaned data
    self.canvas_ecg_clean.setEnabled(True)
    self.toolbar_ecg_clean.setEnabled(True)
    self.ax_ecg_clean.clear()
    self.ax_ecg_clean.set_title("Cleaned ECG Signal")
    self.ax_ecg_clean.plot(times,full_data, label='Raw data')
    self.ax_ecg_clean.plot(times,clean_data, label='Cleaned data')
    self.ax_ecg_clean.set_xlabel("Time (s)")
    self.ax_ecg_clean.set_ylabel("Amplitude")
    self.ax_ecg_clean.legend()
    self.canvas_ecg_clean.draw()

    # Plot an overlap of the power spectrum using welch's method:
    n_fft = int(round(self.dataset_intra.sf))
    n_overlap=int(round(self.dataset_intra.sf)/2)

    psd_raw, freqs_raw = mne.time_frequency.psd_array_welch(
        full_data,self.dataset_intra.sf,fmin=0,
        fmax=125,n_fft=n_fft,
        n_overlap=n_overlap)
    
    psd_clean, freqs_clean = mne.time_frequency.psd_array_welch(
        clean_data,self.dataset_intra.sf,fmin=0,
        fmax=125,n_fft=n_fft,
        n_overlap=n_overlap)

    self.canvas_psd.setEnabled(True)
    self.toolbar_psd.setEnabled(True)
    self.ax_psd.clear()
    self.ax_psd.plot(
        freqs_raw, np.log(psd_raw), color='blue', label='PSD raw channel'
        )
    self.ax_psd.plot(
        freqs_clean, np.log(psd_clean), color = 'orange', 
        label='PSD cleaned channel'
        )
    self.ax_psd.legend()
    self.canvas_psd.draw()

    self.btn_confirm_cleaning.setEnabled(True)  # Enable the button after cleaning



    #######################################################################
    #########              TEMPLATE SUBSTRACTION METHOD           #########
    #######################################################################      


def start_ecg_cleaning_template_sub(self):
    self.ax_ecg_clean.clear()
    self.ax_ecg_artifact.clear()
    self.ax_psd.clear()
    """Start the ECG cleaning process using the template substraction method."""
    try:
        clean_ecg_template_sub(self)

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to clean ECG: {e}")


def clean_ecg_template_sub(self):
    full_data = self.dataset_intra.synced_data.get_data()[
        self.dataset_intra.selected_channel_index_ecg
        ]
    times = np.linspace(
        0, 
        self.dataset_intra.synced_data.get_data().shape[1]/self.dataset_intra.sf, 
        self.dataset_intra.synced_data.get_data().shape[1]
        )
    window = [-0.2, 0.2] # QRS complex window

    ############################################################################
    clean_data = np.copy(full_data)
    ns = len(full_data)

    # Create a QRS template #
    pre_samples = int(abs(window[0]) * self.dataset_intra.sf)
    post_samples = int(window[1] * self.dataset_intra.sf)
    epoch_length = pre_samples + post_samples  # Total length of each epoch
    timescale_epoch = np.linspace(
        window[0], window[1], epoch_length
        )  # Time in seconds

    epochs = []  # Store extracted heartbeats        

    for peak in self.final_peaks:
        start = peak - pre_samples
        end = peak + post_samples
        epochs.append(full_data[start:end])

    epochs = np.array(epochs)

    # Compute average QRS template
    mean_epoch = np.nanmean(epochs, axis=0)
    self.canvas_ecg_artifact.setEnabled(True)
    self.toolbar_ecg_artifact.setEnabled(True)
    self.ax_ecg_artifact.clear()
    self.ax_ecg_artifact.set_title("Detected QRS epochs")

    for epoch in epochs:
        self.ax_ecg_artifact.plot(
            timescale_epoch, epoch, color='gray', alpha=0.3
            )
    
    self.ax_ecg_artifact.plot(
        timescale_epoch, mean_epoch, color='black', linewidth=2, 
        label='Average QRS Template'
        )
    self.ax_ecg_artifact.set_xlabel("Time (s)")
    self.ax_ecg_artifact.set_ylabel("Amplitude")
    self.ax_ecg_artifact.legend()
    self.canvas_ecg_artifact.draw()

    ####################################################################
    #artifact_data = np.copy(full_data)
    pre_samples = int(abs(window[0]) * self.dataset_intra.sf)
    post_samples = int(window[1] * self.dataset_intra.sf)

    for _, peak in enumerate(self.final_peaks):
        raw_epoch = full_data[(peak - pre_samples):(peak + post_samples)]
        # Prepare design matrix for linear fit (scale + offset)
        X_template = np.vstack([mean_epoch, np.ones_like(mean_epoch)]).T
        # Solve for optimal scale (a) and offset (b) using least squares
        coeffs, _, _, _ = np.linalg.lstsq(X_template, raw_epoch, rcond=None)
        a, b = coeffs

        # Build fitted template
        fitted_template = a * mean_epoch + b

        # Equalize tails
        complex_qrs_template, start_idx, end_idx = find_similar_sample(
            fitted_template, tails=30
            )
        start = (peak - pre_samples) + start_idx
        end = (peak - pre_samples) + end_idx
        raw_epoch = full_data[start:end]
        assert len(raw_epoch) == len(complex_qrs_template), "Raw epoch length does not match complex QRS template length"
        clean_data[start:end] -= complex_qrs_template
        #artifact_data[start:end] = complex_qrs_template

    #artifact_data_full = artifact_data    

    if self.dataset_intra.selected_channel_index_ecg == 0:
        self.dataset_intra.cleaned_ecg_left = clean_data
        print("Left channel cleaned")

    elif self.dataset_intra.selected_channel_index_ecg == 1:
        self.dataset_intra.cleaned_ecg_right = clean_data
        print("Right channel cleaned")

    # plot an overlap of the raw and cleaned data
    self.canvas_ecg_clean.setEnabled(True)
    self.toolbar_ecg_clean.setEnabled(True)
    self.ax_ecg_clean.clear()
    self.ax_ecg_clean.set_title("Cleaned ECG Signal")
    self.ax_ecg_clean.plot(times, full_data, label='Raw data')
    self.ax_ecg_clean.plot(times, clean_data, label='Cleaned data')
    #self.ax_ecg_clean.plot(times, artifact_data_full, label='Artifact data', color='red', alpha=0.5)
    self.ax_ecg_clean.set_xlabel("Time (s)")
    self.ax_ecg_clean.set_ylabel("Amplitude")
    self.ax_ecg_clean.legend()
    self.canvas_ecg_clean.draw()

    # Plot an overlap of the power spectrum using welch's method:
    n_fft = int(round(self.dataset_intra.sf))
    n_overlap=int(round(self.dataset_intra.sf)/2)

    psd_raw, freqs_raw = mne.time_frequency.psd_array_welch(
        full_data,self.dataset_intra.sf,fmin=0,
        fmax=125,n_fft=n_fft,
        n_overlap=n_overlap)
    
    psd_clean, freqs_clean = mne.time_frequency.psd_array_welch(
        clean_data,self.dataset_intra.sf,fmin=0,
        fmax=125,n_fft=n_fft,
        n_overlap=n_overlap)

    self.canvas_psd.setEnabled(True)
    self.toolbar_psd.setEnabled(True)
    self.ax_psd.clear()
    self.ax_psd.plot(
        freqs_raw, np.log(psd_raw), color='blue', label='PSD raw channel'
        )
    self.ax_psd.plot(
        freqs_clean, np.log(psd_clean), color = 'orange', 
        label='PSD cleaned channel'
        )
    self.ax_psd.legend()
    self.canvas_psd.draw()

    self.btn_confirm_cleaning.setEnabled(True)  # Enable the button after cleaning

      


    #######################################################################
    #########          SINGULAR VALUE DECOMPOSITION METHOD        #########
    #######################################################################      

def start_ecg_cleaning_svd(self):
    """Start the ECG cleaning process using Singular Value Decomposition method."""
    self.ax_ecg_clean.clear()
    self.ax_ecg_artifact.clear()
    self.ax_psd.clear()
    try:
        clean_ecg_svd(self)

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to clean ECG: {e}")


def clean_ecg_svd(self):
    """
    This function cleans the ECG signal using Singular Value Decomposition (SVD).
    It extracts epochs around the R-peaks, performs SVD on these epochs,
    and then reconstructs the signal using the first few singular values.
    The function opens a secondary plotting window to visualize the SVD results,
    so that the user can choose which components to keep to reconstruct the signal.
    """
    self.full_data = self.dataset_intra.synced_data.get_data()[
        self.dataset_intra.selected_channel_index_ecg
        ]
    self.window = [-0.2, 0.2] # add an option to choose QRS or PQRST window??

    # Create a QRS template #
    pre_samples = int(abs(self.window[0]) * self.dataset_intra.sf)
    post_samples = int(self.window[1] * self.dataset_intra.sf)
    self.epoch_length = pre_samples + post_samples  # Total length of each epoch

    epochs = []  # Store extracted heartbeats

    for peak in self.final_peaks:
        start = peak - pre_samples
        end = peak + post_samples
        epochs.append(self.full_data[start:end])
    epochs = np.array(epochs)    # shape: (n timepoints, n epochs)

    ######### SINGULAR VALUE DECOMPOSITION ################
    X = epochs.T                        # shape: (n epochs, n timepoints)
    self.U, self.S, self.Vh = np.linalg.svd(X, full_matrices=False)

    # Open the secondary plotting window for the SVD template
    self.plot_window = PlotWindow(
        self.process_value_from_plot, self.U, self.S, self.window, 
        self.epoch_length
        )
    self.plot_window.show()


def simple_peak_detection_fallback(cropped_data, sf_lfp, detection_threshold):
    """
    Fallback method for peak detection using a simpler approach.
    This method applies peak detection directly to the entire signal.
    """
    print("Trying simple fallback peak detection method...")
    
    # Try peak detection on the entire signal with different parameters
    for height_pct in [90, 80, 70, 60, 50, 40]:
        for distance_factor in [2, 3, 4, 5]:  # Different distance constraints
            distance = sf_lfp // distance_factor
            
            # Try positive peaks
            height_pos = np.percentile(cropped_data, height_pct)
            peaks_pos, _ = scipy.signal.find_peaks(
                cropped_data, height=height_pos, distance=distance
            )
            
            # Try negative peaks
            height_neg = np.percentile(-cropped_data, height_pct)
            peaks_neg, _ = scipy.signal.find_peaks(
                -cropped_data, height=height_neg, distance=distance
            )
            
            print(f"  Trying threshold {height_pct}%, distance {distance}: {len(peaks_pos)} pos, {len(peaks_neg)} neg peaks")
            
            # Return the first successful detection with reasonable number of peaks
            if len(peaks_pos) >= 3:  # At least 3 peaks for meaningful analysis
                print(f"  Success with positive peaks: {len(peaks_pos)} peaks found")
                return peaks_pos, 'Up'
            elif len(peaks_neg) >= 3:
                print(f"  Success with negative peaks: {len(peaks_neg)} peaks found")
                return peaks_neg, 'Down'
    
    print("  Fallback method also failed to find peaks")
    return np.array([]), 'Unknown'


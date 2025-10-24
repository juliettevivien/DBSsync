"""
utilisation functions
"""

from tkinter.filedialog import askdirectory
import numpy as np
from scipy.signal import find_peaks

parameters = {}

def find_similar_sample(
        reconstructed_epoch: np.ndarray, 
        tails: int
        ):
    """
    This function finds two samples in the reconstructed epoch
    that have the smallest absolute difference in value.
    It returns the QRS complex template and the indices of the two samples.
    Args:
        reconstructed_epoch (np.ndarray): The reconstructed epoch of the ECG signal.
        tails (int): The number of samples to consider at the start and end of the epoch.
    Returns:
        complex_qrs_template (np.ndarray): The QRS complex template extracted from the reconstructed epoch.
        start_idx (int): The index of the first sample in the QRS complex template.
        end_idx (int): The index of the last sample in the QRS complex template.
    """
    # Step 1: look for samples with similar values.
    #  To avoid any introduced offsets between the corrected epochs and
    #  neighbouring data points, per epoch, the first and last 4 ms of
    #  the reconstructed ECG artifact were searched for two samples
    #  whose values had the smallest difference, and were set equal in
    #  case of dissimilar values (Stam et al 2023). BUT our sf is 250Hz, so 4 ms is 1 sample.
    #  We will therefore use more samples to ensure a good correction, e.g. 30 samples.
    n_samples = tails
    start_samples = reconstructed_epoch[:n_samples]
    right_tail_start = len(reconstructed_epoch) - n_samples
    end_samples = reconstructed_epoch[right_tail_start:]

    # Find pair (q, s) with smallest absolute difference
    min_diff = float("inf")
    start_idx, end_idx = None, None

    for i, q_val in enumerate(start_samples):
        for j, s_val in enumerate(end_samples):
            diff = abs(q_val - s_val)
            if diff < min_diff:
                min_diff = diff
                start_idx = i
                end_idx = j + right_tail_start  # Adjust index relative to full epoch


    if reconstructed_epoch[start_idx] != reconstructed_epoch[end_idx]:
        higher_value = max(
            reconstructed_epoch[start_idx], reconstructed_epoch[end_idx]
            )
        reconstructed_epoch[start_idx] = reconstructed_epoch[end_idx] = higher_value
    
    # Extract the QRS complex template 
    complex_qrs_template = reconstructed_epoch[start_idx:end_idx]   

    return complex_qrs_template, start_idx, end_idx

def get_start_end_times(
        full_data: np.ndarray, 
        times: np.ndarray,
        start_window_duration: float = None,
        end_window_duration: float = None
        ):
    """
    This function finds the last peak in the beginning of the recording
    and the first peak in the end of the recording. It is used to avoid 
    periods of time containing artifacts from the stimulation pulses.
    
    Args:
        full_data: The signal data
        times: Time array corresponding to the data
        start_window_duration: Duration (in seconds) to look for peaks at the beginning.
                               If None, uses 20% of recording duration.
        end_window_duration: Duration (in seconds) to look for peaks at the end.
                            If None, uses 20% of recording duration.
    """
    # Compute absolute value to be invariant to the polarity of the signal
    abs_data = np.abs(full_data)
    peaks, _ = find_peaks(abs_data, prominence=100)
    peak_times = times[peaks]
    
    rec_duration = times[-1] - times[0]
    
    # Define search windows for beginning and end of recording
    if start_window_duration is None:
        start_window_duration = rec_duration * 0.1  # First 10% of recording
    if end_window_duration is None:
        end_window_duration = rec_duration * 0.1   # Last 10% of recording
    
    start_window_end = times[0] + start_window_duration
    end_window_start = times[-1] - end_window_duration
    
    # Separate peaks into beginning and end windows, ignoring middle artifacts
    peak_times_start = peak_times[peak_times <= start_window_end]
    peak_times_end = peak_times[peak_times >= end_window_start]
    
    # Alternative approach: if you want to be more sophisticated and still use
    # the long interval detection but with better handling of middle artifacts
    if len(peak_times_start) == 0 and len(peak_times_end) == 0:
        # Fallback to original method if windowing doesn't work
        min_distance_between_peaks = rec_duration * 0.3  # Reduced from 0.5 to be more sensitive
        long_intervals = np.where(np.diff(peak_times) > min_distance_between_peaks)[0]
        
        if len(long_intervals) > 0:
            # Take the first long interval as the separator
            split_index = long_intervals[0] + 1
            peak_times_start = peak_times[:split_index]
            peak_times_end = peak_times[split_index:]
            
            # Remove any peaks that are clearly in the middle (artifacts)
            # Keep only peaks in the first and last portions
            middle_start = times[0] + rec_duration * 0.3
            middle_end = times[-1] - rec_duration * 0.3
            
            # Filter out middle artifacts from start peaks
            peak_times_start = peak_times_start[peak_times_start <= middle_start]
            # Filter out middle artifacts from end peaks  
            peak_times_end = peak_times_end[peak_times_end >= middle_end]
        else:
            # If no long intervals found, assume all peaks are either start or end
            # Split based on position in recording
            middle_time = times[0] + rec_duration * 0.5
            peak_times_start = peak_times[peak_times < middle_time]
            peak_times_end = peak_times[peak_times > middle_time]

    # extract the last peak of peak_times_start
    if len(peak_times_start) > 0:
        last_peak_start = peak_times_start[-1]
    else:
        last_peak_start = 10 #  this is a default value, if no peak is found in 
        #  the beginning of the recording it will automatically skip 10 seconds

    # extract the first peak of peak_times_end
    if len(peak_times_end) > 0:
        first_peak_end = peak_times_end[0]
    else:
        first_peak_end = times[-1] - 10 #  this is a default value, if no peak is found at 
        #  the end of the recording it will automatically skip the last 10 seconds

    return last_peak_start + 1, first_peak_end - 1

def _find_closest_index(array, value, tolerance=1e-6):
    diff = np.abs(array - value)
    if np.any(diff <= tolerance):
        return np.argmin(diff)
    else:
        raise ValueError(f"No element found within {tolerance} of {value}")



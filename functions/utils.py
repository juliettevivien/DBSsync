"""
utilisation functions
"""

from tkinter.filedialog import askdirectory
import numpy as np
from scipy.signal import find_peaks
import re

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



def convert_list_string_floats(
    string_list
):
    """
    Copied from jgvhabets pyPerceive repository.
    """
    try:
        floats = [float(v) for v in string_list.split(',')]
    except:
        floats = [float(v) for v in string_list[:-1].split(',')]

    return floats




def check_and_correct_missings_in_lfp(dat: dict):
    """"
    Copied from jgvhabets pyPerceive repository.

    Function checks missing packets based on start and endtime
    of first and last received packets, and the time-differences
    between consecutive packets. In case of a missing packet,
    the missing time window is filled with NaNs.
    """
    Fs = dat['SampleRateInHz']
    if type(dat['TicksInMses']) == str:
        ticksMsec = convert_list_string_floats(dat['TicksInMses'])
    else: 
        ticksMsec = dat['TicksInMses']
    ticksDiffs = np.diff(np.array(ticksMsec))

    data_is_missing = (ticksDiffs != 250).any()
    if type(dat['GlobalPacketSizes']) == str:
        packetSizes = convert_list_string_floats(dat['GlobalPacketSizes'])
    else:
        packetSizes = dat['GlobalPacketSizes']

    # small plot for debugging
    # plt.plot(ticksDiffs)
    # plt.plot(packetSizes)

    lfp_data = dat['TimeDomainData']

    if data_is_missing:
        print('LFP Data is missing!! performing function to fill NaNs in')
        # data_length_ms = (ticksMsec[-1] - ticksMsec[0]) + 250  # length of a packet in milliseconds is always 250
        # print(f'Start time (ms): {ticksMsec[0]}, End time (ms): {ticksMsec[-1]}')
        # print(f'Calculated data length (ms): {data_length_ms}')
        # data_length_samples = int(data_length_ms / 1000 * Fs) 
        # print(f'Calculated data length (samples): {data_length_samples}')
        #new_lfp_arr = np.array([np.nan] * data_length_samples)  # create new array full of NaNs but of proper length

        # fill nan array with real LFP values, use tickDiffs to decide start-points (and where to leave NaN)
        # Add first packet (data always starts with present packet)
        #new_lfp_arr[:int(packetSizes[0])] = lfp_data[:int(packetSizes[0])] # first packet of 250ms always present
        #new_lfp_arr[ticksMsec[-1]-packetSizes[-1]:ticksMsec[-1]] = lfp_data[-int(packetSizes[-1]):]  # last packet always present

        # compute indexes where nans should be added:
        # search for values larger than 250 in ticksDiff and return indexes and values
        
        missing_indexes = np.where(ticksDiffs > 250)[0]

        time_missing = (ticksDiffs[missing_indexes] - 250).astype(int)
        
        # convert indexes of ticks to time domain indexes with the formula index_in_timedomain = ((missing_indexes*62.5)+62.5)-1
        #indexes_start_in_timedomain = (np.round(((missing_indexes * 62.5) + 62.5) - 1)).astype(int)
        indexes_start_in_timedomain = []
        for i in range(len(missing_indexes)):
            index_start = int(np.sum(packetSizes[:missing_indexes[i]+1]))
            print(index_start)
            indexes_start_in_timedomain.append(index_start)
        indexes_start_in_timedomain = np.array(indexes_start_in_timedomain)

        # # then fill in the lfp values in between the missing indexes
        # t_arr=0
        # t_new=0
        # new_lfp_arr = []  # initialize empty list
        # for start, t_miss in zip(indexes_start_in_timedomain, time_missing): 
        #     # extract lfp data until start of missing data
        #     lfp_data_segment = lfp_data[t_arr:start]
        #     new_lfp_arr.extend(lfp_data_segment)  # add lfp data segment to new array
        #     # add NaNs for missing data
        #     nans_to_add = int(t_miss / 4)  # convert milliseconds to samples (Fs=250Hz -> 4ms per sample)
        #     new_lfp_arr.extend([np.nan] * nans_to_add)  # add NaNs for missing data
        #     t_arr = start  # update t_arr to the start of missing data
        # # add remaining lfp data after last missing segment
        # new_lfp_arr.extend(lfp_data[t_arr:])
        # new_lfp_arr = np.array(new_lfp_arr)  # convert list to numpy array

        new_lfp_arr = [] # initialize empty list
        for i in range(len(indexes_start_in_timedomain)):
            print(f'Missing data at index {indexes_start_in_timedomain[i]}, missing time (ms): {time_missing[i]}, adding {int(time_missing[i] / 4)} NaNs')
            lfp_array_segment = lfp_data[indexes_start_in_timedomain[i-1] : indexes_start_in_timedomain[i]] if i > 0 else lfp_data[0 : indexes_start_in_timedomain[i]]
            new_lfp_arr.extend(lfp_array_segment)  # add lfp data segment to new array
            total_samples_to_add = fill_missing_packets(missing_indexes[i], ticksDiffs, packetSizes[i])
            #nan_array = [np.nan] * int(time_missing[i] / 4)  # convert milliseconds to samples (Fs=250Hz -> 4ms per sample)
            nan_array = [np.nan] * total_samples_to_add
            #print(len(nan_array))
            new_lfp_arr.extend(nan_array)  # add NaNs for missing data
        # add remaining lfp data after last missing segment
        new_lfp_arr.extend(lfp_data[indexes_start_in_timedomain[-1]:])
        new_lfp_arr = np.array(new_lfp_arr)  # convert list to numpy array


        # # loop over every distance (index for packetsize is + 1 because first difference corresponds to seconds packet)
        # i_lfp = int(packetSizes[0])  # index to track which lfp values are already used
        # i_arr = int(packetSizes[0])  # index to track of new array index
        # i_packet = 1
        # diff_idx = 0
        # for diff in ticksDiffs:
        #     if diff == 250:
        #         # only lfp values, no nans if distance was 250 ms
        #         new_lfp_arr[
        #             i_arr:int(i_arr + packetSizes[i_packet])
        #         ] = lfp_data[i_lfp:int(i_lfp + packetSizes[i_packet])]
        #         i_lfp += int(packetSizes[i_packet])
        #         i_arr += int(packetSizes[i_packet])
        #         i_packet += 1
        #     else:
        #         print('add NaNs by skipping')
        #         print(f'Difference in ms: {diff} --> missing packets at index {diff_idx}')
        #         msecs_missing = (diff - 250)  # 250 milliseconds of the difference are the present previous packet
        #         secs_missing = msecs_missing / 1000
        #         #samples_missing = secs_missing / Fs
        #         samples_missing = int(secs_missing * Fs)
        #         print(f'Missing milliseconds: {msecs_missing}, Missing samples: {samples_missing}')
        #         # no filling with NaNs, bcs array is created full with NaNs
        #         i_arr += samples_missing  # shift array index up by number of NaNs left in the array
        #         i_packet += 1
        #     diff_idx += 1
        #     print(f'Next i_arr: {i_arr}, Next i_lfp: {i_lfp}, Next i_packet: {i_packet}')
        # correct in case one sample too many was in array shape
        # if np.isnan(new_lfp_arr[-1]): 
        #     new_lfp_arr = new_lfp_arr[:-1]

        # always remove 250 from unique diffs, as that is the normal packet distance
        unique_diff = np.unique(ticksDiffs)
        unique_diff = unique_diff[unique_diff != 250.0]

        # return how many times each missing packet length occurred
        missing_packet_lengths = np.unique(ticksDiffs[ticksDiffs != 250.0])
        missing_packet_real_lengths = missing_packet_lengths - 250  # in milliseconds
        missing_packet_counts = {length: int(np.sum(ticksDiffs == length+250)) for length in missing_packet_real_lengths}   
        #print(f'Missing packet lengths (ms) and counts: {missing_packet_counts}')

        return new_lfp_arr, f'Yes, missing packets: {missing_packet_counts}'  # return new lfp array with NaNs filled in and a flag that data was missing

    else:
        print('No LFP data missing based on timestamp '
            'differences between data-packets')
        return np.array(lfp_data), 'No'





def fill_missing_packets(index, ticksDiff, last_packet):
    print(f'Missing packet at index {index}')
    missing_time = ticksDiff[index] - 250
    print(f'Missing time (ms): {missing_time}')
    print(f'Last packet size (samples): {last_packet}')
    n_packets_to_add = missing_time // 250
    print(f'Number of packets to add: {n_packets_to_add}')
    Y = missing_time // 250

    if Y % 2 == 0:
        print(f'Even number of packets to add')
        num_62 = num_63 = Y // 2
    else:
        if last_packet == 63:
            num_62 = (Y // 2) + 1
            num_63 = Y // 2
        else:
            num_63 = (Y // 2) + 1
            num_62 = Y // 2
    
    total_samples_to_add = int((num_62 * 62) + (num_63 * 63))
    #print(f'Adding total samples: {total_samples_to_add} ({num_62} packets of 62 samples and {num_63} packets of 63 samples)')
    return total_samples_to_add



def convert_msec_to_min_sec_msec(milliseconds):
    minutes = milliseconds // 60000
    seconds = (milliseconds % 60000) // 1000
    msecs = milliseconds % 1000

    return minutes, seconds, msecs



def format_timedelta(td):
    total_ms = int(td.total_seconds() * 1000)

    days = total_ms // (24 * 60 * 60 * 1000)
    total_ms %= (24 * 60 * 60 * 1000)

    hours = total_ms // (60 * 60 * 1000)
    total_ms %= (60 * 60 * 1000)

    minutes = total_ms // (60 * 1000)
    total_ms %= (60 * 1000)

    seconds = total_ms // 1000
    msecs = total_ms % 1000

    return f"{days} days, {hours}h, {minutes}min, {seconds}s, {msecs}ms"


def time_to_ms(time_str):
    """
    Convert a time string like '0 days, 0h, 8min, 4s, 500ms' to total milliseconds.
    """
    # Extract numbers using regex (optional units for safety)
    days = int(re.search(r'(\d+)\s*day', time_str).group(1)) if re.search(r'day', time_str) else 0
    hours = int(re.search(r'(\d+)\s*h', time_str).group(1)) if re.search(r'\d+\s*h', time_str) else 0
    minutes = int(re.search(r'(\d+)\s*min', time_str).group(1)) if re.search(r'\d+\s*min', time_str) else 0
    seconds = int(re.search(r'(\d+)\s*s(?!\w)', time_str).group(1)) if re.search(r'\d+\s*s(?!\w)', time_str) else 0
    milliseconds = int(re.search(r'(\d+)\s*ms', time_str).group(1)) if re.search(r'\d+\s*ms', time_str) else 0

    # Compute total milliseconds
    total_ms = (
        days * 24 * 60 * 60 * 1000 +
        hours * 60 * 60 * 1000 +
        minutes * 60 * 1000 +
        seconds * 1000 +
        milliseconds
    )
    return total_ms
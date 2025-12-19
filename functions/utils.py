"""
utilisation functions
"""

from tkinter.filedialog import askdirectory
import numpy as np
from scipy.signal import find_peaks
import re
import mne

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
    This function finds the last ~stimulation peak in the beginning of the recording
    and the first stimulation peak in the end of the recording. It is used to avoid 
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



def check_and_correct_missing_packets(streamings_dict, BrainSenseRaws, streamings_df):
    """
    Inspired by jgvhabets pyPerceive repository check_and_correct_missings_in_lfp function.
    Detection similar but correction improved to better match the actual missing packets.
    """
    BrainSenseRawsCorrected = {}
    corrected_column = []

    for i in streamings_dict.keys():
        dat = streamings_dict[i][BrainSenseRaws[i].ch_names[0]]
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

        data_array = BrainSenseRaws[i].get_data()
        new_data_array = []
        metadata = BrainSenseRaws[i].info

        if data_is_missing:
            missing_indexes = np.where(ticksDiffs > 250)[0]
            print(f"Streaming {i} has missing ticks at indexes: {missing_indexes}")
            #time_missing = (ticksDiffs[missing_indexes] - 250).astype(int)
            duplicated_indexes = np.where(ticksDiffs < 250)[0] 
            duplicated_ticks_values = np.unique([int(ticksMsec[idx]) for idx in duplicated_indexes])
            print(f"Streaming {i} has duplicated ticks at indexes: {duplicated_indexes}. Duplicated ticks are {duplicated_ticks_values}.")
            # sometimes ticks are duplicated because many packets/samples were 
            # sent after a short interruption, this has to be taken into account 
            # before correcting timestamps.
            #irregular_sizes = [size for size in packetSizes if size not in [62, 63]]
            #bigger_packets_indexes = [np.where(size not in [62, 63])[0] for size in packetSizes if size not in [62, 63]]
            bigger_packets_indexes = np.where(~np.isin(packetSizes, [62, 63]))[0]
            print(f"Streaming {i} has bigger packets at indexes: {bigger_packets_indexes}. Packets are {[int(packetSizes[idx]) for idx in bigger_packets_indexes]} samples long.")


            """Here add part to handle cases where missing ticks are due to bigger packets/duplicated ticks/...
            Should start from the top (last Ticks) and go up packet by packet, always checking packet size and duplicated ticks.
            """
            reverted_ticks = ticksMsec[::-1]
            reverted_packetSizes = packetSizes[::-1]
            # create an array of nan with same size as the total number of packets
            #N = sum(packetSizes)
            #data_or_nan = np.full(N, np.nan)
            data_or_nan = []
            #current_idx_ticks = 0
            #current_idx_data_or_nan = 0

            pkt_idx = 0

            while pkt_idx < len(reverted_ticks) - 1:

                tick = int(reverted_ticks[pkt_idx])
                pkt_size = int(reverted_packetSizes[pkt_idx])

                if pkt_size not in [62, 63]:
                    print(f"Packet at index {len(reverted_ticks)-1 - pkt_idx}/{pkt_idx} has irregular size of {pkt_size} samples.")

                tick_pred = tick - (250 * round((pkt_size * 4) / 250))

                if tick_pred == int(reverted_ticks[pkt_idx + 1]):
                    #print("Predicted tick matches the next tick, no missing data, keeping all samples.")
                    data_or_nan.extend(['data'] * pkt_size)
                    pkt_idx += 1
                    continue

                else:
                    # ---------------- DUPLICATED TICK PATH  ----------------
                    if tick == int(reverted_ticks[pkt_idx + 1]):
                        print("Tick is duplicated, handling accordingly.")

                        num_duplicates = 1
                        while int(reverted_ticks[pkt_idx + 1 + num_duplicates]) == tick:
                            num_duplicates += 1

                        print(f" Num of duplicates for tick value {tick} is {num_duplicates}.")

                        total_samples_in_duplicated_ticks = 0
                        for dup_idx in range(pkt_idx, pkt_idx + 1 + num_duplicates):
                            total_samples_in_duplicated_ticks += int(reverted_packetSizes[dup_idx])
                            print(f" Adding packet of size {int(reverted_packetSizes[dup_idx])} samples.")

                        print(f"total samples : {total_samples_in_duplicated_ticks}")

                        tick_after_duplicates = int(reverted_ticks[pkt_idx + 1 + num_duplicates])
                        # double-check if previous packet also was of irregular size. If yes, then take it into account for expected tick calculation
                        previous_packet_size = reverted_packetSizes[pkt_idx - 1] if pkt_idx - 1 >= 0 else 0
                        if previous_packet_size not in [62, 63] and pkt_idx - 1 >= 0:
                            print(f"Previous packet at index {len(reverted_ticks)-1 - (pkt_idx - 1)}/{pkt_idx - 1} also has irregular size of {previous_packet_size} samples.")
                            global_extra_pkt_size = total_samples_in_duplicated_ticks + int(previous_packet_size)
                            previous_tick = int(reverted_ticks[pkt_idx - 1])
                            expected_tick_after_duplicates = previous_tick - (
                                250 * round(((global_extra_pkt_size) * 4) / 250)
                            )

                        else:
                            expected_tick_after_duplicates = tick - (
                                250 * round(((total_samples_in_duplicated_ticks) * 4) / 250)
                            )
                        print(f"Predicted next tick after duplicated ticks: {expected_tick_after_duplicates}, actual next tick: {tick_after_duplicates}")

                        if expected_tick_after_duplicates == tick_after_duplicates:
                            print(f"Duplicated tick but no loss of data detected")
                            data_or_nan.extend(['data'] * total_samples_in_duplicated_ticks)
                            pkt_idx += num_duplicates + 1 # <-- move to the next non-duplicate packet
                            print(f"Moving to packet index {pkt_idx}.")
                            continue
                        else:
                            print(f"Missing data detected after duplicated ticks")
                            data_or_nan.extend(['data'] * total_samples_in_duplicated_ticks) # add the data anyway, but complete with nans
                            # here add nan for missing data after duplicated ticks
                            missing_samples_after_duplicates = expected_tick_after_duplicates - tick_after_duplicates
                            n_missing_packets_after_duplicates = missing_samples_after_duplicates // 250
                            # determine how many 62 and 63 packets to add based on last packet size
                            if n_missing_packets_after_duplicates % 2 == 0:
                                num_62 = num_63 = n_missing_packets_after_duplicates // 2
                            else:
                                if pkt_size == 63:
                                    num_62 = (n_missing_packets_after_duplicates // 2) + 1
                                    num_63 = n_missing_packets_after_duplicates // 2
                                else:
                                    num_63 = (n_missing_packets_after_duplicates // 2) + 1
                                    num_62 = n_missing_packets_after_duplicates // 2
                            total_missing_samples_to_add = (num_62 * 62) + (num_63 * 63)
                            data_or_nan.extend(['nan'] * total_missing_samples_to_add)
                            print(f"Added {total_missing_samples_to_add} NaN samples for {n_missing_packets_after_duplicates} missing packets ")
                            pkt_idx += num_duplicates + 1  # move to the next non-duplicate packet
                            print(f"Moving to packet index {pkt_idx}.")
                            continue

                    # ---------------- MISSING TICK PATH ----------------
                    else:
                        print("Missing tick(s) detected, adding NaNs for missing data.")

                        total_missing_time = tick_pred - int(reverted_ticks[pkt_idx + 1])
                        n_missing_packets = total_missing_time // 250

                        # determine how many 62 and 63 packets to add based on last packet size
                        if n_missing_packets % 2 == 0:
                            num_62 = num_63 = n_missing_packets // 2
                        else:
                            if pkt_size == 63:
                                num_62 = (n_missing_packets // 2) + 1
                                num_63 = n_missing_packets // 2
                            else:
                                num_63 = (n_missing_packets // 2) + 1
                                num_62 = n_missing_packets // 2

                        total_samples_to_add = (num_62 * 62) + (num_63 * 63)
                        data_or_nan.extend(['nan'] * total_samples_to_add)

                        print(f"Added {total_samples_to_add} NaN samples for {n_missing_packets} missing packets "
                            f"({num_62} of 62 samples and {num_63} of 63 samples).")

                        pkt_idx += 1   # EXACTLY like you originally had
                        continue

            # for pkt_idx in range(0, len(reverted_ticks), 1):
            #     tick = int(reverted_ticks[pkt_idx])
            #     pkt_size = int(reverted_packetSizes[pkt_idx])
            #     if pkt_size not in [62, 63]:
            #         print(f"Packet at index {len(reverted_ticks)-1 - pkt_idx} has irregular size of {pkt_size} samples.")
            #     tick_pred = tick - (250 * round((pkt_size * 4) / 250))
            #     if tick_pred == int(reverted_ticks[pkt_idx + 1]):
            #         print("Predicted tick matches the next tick, no missing data, keeping all samples.")
            #         # ticks are matching, add 'data' string
            #         data_or_nan.extend(['data'] * pkt_size)
            #         #current_idx_data_or_nan += pkt_size
            #         #current_idx_ticks += 1
            #     else:
            #         # tick difference detected, need to assess if missing or duplicated tick
            #         if tick == int(reverted_ticks[pkt_idx + 1]):
            #             print("Tick is duplicated, handling accordingly.")
            #             # search for how many times this tick is duplicated
            #             num_duplicates = 1
            #             while int(reverted_ticks[pkt_idx + 1 + num_duplicates]) == tick :
            #                 num_duplicates += 1
            #             print(f" Num of duplicates for tick value {tick} is {num_duplicates}.")
            #             # calculate the total samples in the duplicated ticks
            #             total_samples_in_duplicated_ticks = 0
            #             for dup_idx in range(pkt_idx, pkt_idx + 1 + num_duplicates):
            #                 total_samples_in_duplicated_ticks += int(reverted_packetSizes[dup_idx])
            #                 print(f" Adding packet of size {int(reverted_packetSizes[dup_idx])} samples.")
            #             print(f"total samples : {total_samples_in_duplicated_ticks}")
            #             # check if number of samples in duplicated ticks correspond to the tick difference
            #             tick_after_duplicates = int(reverted_ticks[pkt_idx + 1 + num_duplicates])
            #             expected_tick_after_duplicates = tick - (250 * round(( (pkt_size + total_samples_in_duplicated_ticks) * 4) / 250))
            #             if expected_tick_after_duplicates == tick_after_duplicates:
            #                 print(f"Duplicated tick but no loss of data detected")
            #                 data_or_nan.extend(['data'] * total_samples_in_duplicated_ticks)                            
            #                 pkt_idx += num_duplicates  # skip the duplicated ticks
            #             else:
            #                 print(f"Missing data detected after duplicated ticks")
            #                 data_or_nan.extend(['data'] * total_samples_in_duplicated_ticks)
            #                 pkt_idx += num_duplicates  # skip the duplicated ticks

            #         else:
            #             print("Missing tick(s) detected, adding NaNs for missing data.")
            #             total_missing_time = tick_pred - int(reverted_ticks[pkt_idx + 1])
            #             n_missing_packets = total_missing_time // 250
            #             # determine how many 62 and 63 packets to add based on last packet size
            #             if n_missing_packets % 2 == 0:
            #                 num_62 = num_63 = n_missing_packets // 2
            #             else:
            #                 if pkt_size == 63:
            #                     num_62 = (n_missing_packets // 2) + 1
            #                     num_63 = n_missing_packets // 2
            #                 else:
            #                     num_63 = (n_missing_packets // 2) + 1
            #                     num_62 = n_missing_packets // 2
                        
            #             total_samples_to_add = (num_62 * 62) + (num_63 * 63)
            #             data_or_nan.extend(['nan'] * total_samples_to_add)
            #             print(f"Added {total_samples_to_add} NaN samples for {n_missing_packets} missing packets ({num_62} of 62 samples and {num_63} of 63 samples).")

                        
                # elif tick_pred > int(reverted_ticks[pkt_idx + 1]):
                #     # missing tick(s) detected, add NaNs for missing data
                #     data_or_nan.append('nan' * pkt_size)
                #     #current_idx_ticks += 1
                # elif tick_pred < int(reverted_ticks[pkt_idx + 1]):
                #     # tick might be repeated, check and handle based on packet sizes
                #     if tick == int(reverted_ticks[pkt_idx + 1]):
                #         print("Tick is duplicated, handling accordingly.")

            data_or_nan_reverted = data_or_nan[::-1]     

            # for each streaming, loop through each channel to add the missing NaNs
            for channel in range(len(data_array)):
                ch_data = data_array[channel, :]
                ch_data_idx = 0
                new_ch_arr = [] # initialize empty list
                # add ch_data values or NaNs based on data_or_nan_reverted
                for idx, item in enumerate(data_or_nan_reverted):
                    if item == 'data':
                        new_ch_arr.append(ch_data[ch_data_idx])
                        ch_data_idx += 1
                    elif item == 'nan':
                        new_ch_arr.append(np.nan)
                new_ch_arr = np.array(new_ch_arr)  # convert list to numpy array
                new_data_array.append(new_ch_arr)               




            """OLD VERSION:"""
            # # Verify that missing_indexes actually correspond to missing data samples and not just to bigger packets
            # verified_missing_indexes = []
            # for idx in missing_indexes:
            #     # Handle cases where the ticks is missing due to an increased size of the next packet without duplicated ticks
            #     if (idx+1) in bigger_packets_indexes and (idx+1) not in duplicated_indexes:
            #         print(f"Index {idx+1} corresponds to a bigger packet, maybe not missing data. Checking...")
            #         # next tick should be adjusted by the extra size of the bigger packet
            #         next_predicted_tick = 250 * round(((packetSizes[idx+1] * 4) + ticksMsec[idx]) / 250)
            #         # verify if the tick difference matches the adjusted expected tick
            #         if next_predicted_tick == ticksMsec[idx+1]:
            #             print(f"Index {idx} confirmed as not missing data after adjustment.")
            #             # don't keep this index as missing data, tick is just skipped and compensated for by bigger following packet
            #         else: # if the next predicted tick doesn't match the actual next tick, then it is missing data
            #             print(f"Index {idx} confirmed as missing data even after adjustment.")
            #             verified_missing_indexes.append(idx)
            #     # Handle cases where missing tick corresponds to both bigger packet and duplicated tick
            #     elif (idx+1) in bigger_packets_indexes and (idx+1) in duplicated_indexes:
            #         print(f"Index {idx+1} corresponds to a bigger packet and duplicated tick.")
            #         # get the value of the duplicated tick
            #         ticksMsec_int = np.round(np.array(ticksMsec)).astype(int)
            #         duplicated_tick_value = int(ticksMsec[idx+1])
            #         # check how many times this tick is duplicated
            #         num_duplicates = np.sum(ticksMsec_int == duplicated_tick_value)
            #         print(f" Num of duplicates for tick value {duplicated_tick_value} is {num_duplicates}.")

            #         # calculate the total samples in the duplicated ticks
            #         total_samples_in_duplicated_ticks = 0
            #         for dup_idx in range(idx+1, idx+2 + num_duplicates):
            #             total_samples_in_duplicated_ticks += packetSizes[dup_idx]
            #         print(f"total samples : {total_samples_in_duplicated_ticks}")
            #         # next predicted tick should be adjusted by the extra size of the bigger packet and duplicated ticks
            #         next_predicted_tick = 250 * round((total_samples_in_duplicated_ticks*4 + ticksMsec[idx]) / 250)
            #         print(f"Predicted next tick after adjustment for bigger packet and duplicated ticks: {next_predicted_tick}, actual next tick: {ticksMsec[idx + num_duplicates + 1]}")
            #         # verify if the tick difference matches the adjusted expected tick
            #         if next_predicted_tick == ticksMsec[idx + num_duplicates + 1]:
            #             print(f"Index {idx} confirmed as not missing data after adjustment for bigger packet and duplicated ticks.")
            #             # don't keep this index as missing data, tick is just skipped and compensated for by bigger following packet and duplicated ticks
            #         else: # if the next predicted tick doesn't match the actual next tick, then it is missing data
            #             print(f"Index {idx} confirmed as missing data even after adjustment for bigger packet and duplicated ticks.")
            #             verified_missing_indexes.append(idx)

            # # convert indexes of ticks to time domain indexes
            # indexes_start_in_timedomain = []
            # for k in range(len(verified_missing_indexes)):
            #     index_start = int(np.sum(packetSizes[:verified_missing_indexes[k]+1]))
            #     indexes_start_in_timedomain.append(index_start)
            # indexes_start_in_timedomain = np.array(indexes_start_in_timedomain)

            # # for each streaming, loop through each channel to add the missing NaNs
            # for channel in range(len(data_array)):
            #     ch_data = data_array[channel, :]
            #     new_ch_arr = [] # initialize empty list
            #     for m in range(len(indexes_start_in_timedomain)):
            #         lfp_array_segment = ch_data[indexes_start_in_timedomain[m-1] : indexes_start_in_timedomain[m]] if m > 0 else ch_data[0 : indexes_start_in_timedomain[m]]
            #         new_ch_arr.extend(lfp_array_segment)  # add lfp data segment to new array
            #         total_samples_to_add = fill_missing_packets(verified_missing_indexes[m], ticksDiffs, packetSizes[m])
            #         nan_array = [np.nan] * total_samples_to_add
            #         #print(len(nan_array))
            #         new_ch_arr.extend(nan_array)  # add NaNs for missing data
            #     # add remaining lfp data after last missing segment
            #     new_ch_arr.extend(ch_data[indexes_start_in_timedomain[-1]:])
            #     new_ch_arr = np.array(new_ch_arr)  # convert list to numpy array

                #new_data_array.append(new_ch_arr)

            # always remove 250 from unique diffs, as that is the normal packet distance
            unique_diff = np.unique(ticksDiffs)
            unique_diff = unique_diff[unique_diff != 250.0]

            # return how many times each missing packet length occurred
            missing_packet_lengths = np.unique(ticksDiffs[ticksDiffs != 250.0])
            missing_packet_real_lengths = missing_packet_lengths - 250  # in milliseconds
            missing_packet_counts = {length: int(np.sum(ticksDiffs == length+250)) for length in missing_packet_real_lengths}  
            corrected_or_not = f'Yes, missing packets: {missing_packet_counts}'

            raw_new = mne.io.RawArray(
                data = new_data_array,
                info = metadata
            )
            BrainSenseRawsCorrected[i] = raw_new

        else:
            corrected_or_not = 'No'
            BrainSenseRawsCorrected[i] = BrainSenseRaws[i]

        corrected_column.append(corrected_or_not)

    streamings_df_corrected = streamings_df.copy()
    streamings_df_corrected['LFP Corrected for missing packets'] = corrected_column

    return BrainSenseRawsCorrected, streamings_df_corrected


def fill_missing_packets(index, ticksDiff, last_packet):
    missing_time = ticksDiff[index] - 250
    #n_packets_to_add = missing_time // 250
    Y = missing_time // 250

    if Y % 2 == 0:
        num_62 = num_63 = Y // 2
    else:
        if last_packet == 63:
            num_62 = (Y // 2) + 1
            num_63 = Y // 2
        else:
            num_63 = (Y // 2) + 1
            num_62 = Y // 2
    
    total_samples_to_add = int((num_62 * 62) + (num_63 * 63))

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
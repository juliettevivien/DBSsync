"""
This module contains functions to load and save datasets in various formats.
It supports loading .mat files, .xdf files, and .poly5 files,
as well as saving synchronized datasets in .set, .pkl, and .mat formats.
It also includes functions to synchronize intracranial and external datasets,
crop them based on artifacts, and extract events from annotations.
The module uses MNE-Python for handling EEG data and PyQt5 for GUI interactions.
It is designed to work with datasets from TMSi and other compatible sources.
It provides a user-friendly interface for selecting files and saving results,
allowing users to easily manage their EEG data processing workflow.
It also includes functions to find synchronization streams in .xdf files
and to create a dialog for selecting streams containing sync pulses.


The module contains the following main functions:
- `load_mat_file`: Load a .mat file and extract relevant data.
- `load_ext_file`: Load an external file (.xdf or .poly5) and extract
    relevant data.
- `load_poly5_file`: Load a .poly5 file and extract relevant data.
- `load_xdf_file`: Load a .xdf file and extract relevant data.
- `find_sync_stream`: Find the stream containing synchronization pulses in a .xdf file.
- `select_saving_folder`: Open a dialog to select a folder for saving synchronized datasets.
- `save_datasets_as_set`: Save synchronized datasets as .set files.
- `synchronize_datasets_as_pickles`: Synchronize datasets and save them as .pkl files.
- `synchronize_datasets_as_one_pickle`: Synchronize datasets and save them as one .pkl file.
- `synchronize_datasets_as_mat`: Synchronize datasets and save them as .mat files.
- `write_set`: Export synchronized recordings to EEGLAB .set files.
"""
import mne
from mne.io import read_raw_fieldtrip
from mnelab.io.readers import read_raw
from numpy.core.records import fromarrays
from scipy.io import savemat
from pyxdf import resolve_streams
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
from os.path import basename, dirname, join
import json
from collections import defaultdict
from datetime import datetime, timedelta

from functions.tmsi_poly5reader import Poly5Reader
import functions.utils
from pyxdftools.xdfdata import XdfData

##############################  INPUT FUNCTIONS  ###############################

#### LFP DATASET ####
def load_int_file(self):
    """Load intracranial file. Supported file formats are .mat, .fif"""
    file_name, _ = QFileDialog.getOpenFileName(
        self, "Select Intracranial File", "", 
        "MAT Files (*.mat);;FIF Files (*.fif);;JSON Files (*.json)"
        )
    
    if file_name.endswith(".mat"):
        load_mat_file(self, file_name)
    
    elif file_name.endswith(".fif"):
        load_fif_file_int(self, file_name)

    elif file_name.endswith(".json"):
        load_json_file(self, file_name)
        
    if self.config['NoSync'] == True:
        self.btn_choose_int_channel_for_cleaning.setEnabled(True)

def load_mat_file(self, file_name):
    """Load .mat file."""
    # file_name, _ = QFileDialog.getOpenFileName(
    #     self, "Select MAT File", "", "MAT Files (*.mat);;All Files (*)"
    #     )
    
    if file_name:
        try:
            # Load the .mat file using mne's read_raw_fieldtrip
            raw_data = read_raw_fieldtrip(
                file_name, info={}, data_name="data"
                )
            self.dataset_intra.raw_data = raw_data  # Assign to dataset
            self.dataset_intra.sf = raw_data.info["sfreq"]  # Assign sampling frequency
            self.dataset_intra.ch_names = raw_data.ch_names  # Assign channel names#
            self.dataset_intra.times = np.linspace(
                0, raw_data.get_data().shape[1]/self.dataset_intra.sf, 
                raw_data.get_data().shape[1]
                )
            self.file_label_intra.setText(
                f"Selected File: {basename(file_name)}"
                )
            self.dataset_intra.file_name = basename(file_name)
            self.dataset_intra.file_path = dirname(file_name)

            # Show channel selection and plot buttons for intracranial
            self.btn_select_channel_intra.setEnabled(True)
            self.channel_label_intra.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load .mat file: {e}")



def load_fif_file_int(
        self,
        file_name: str,
        ):
    """Load .fif file."""
    # file_name, _ = QFileDialog.getOpenFileName(
    # self, "Select FIF File", "", "FIF Files (*.fif);;All Files (*)"
    # )
    try:
        raw_data = mne.io.read_raw_fif(file_name, preload=True)

        self.dataset_intra.raw_data = raw_data
        self.dataset_intra.sf = raw_data.info["sfreq"]
        self.dataset_intra.ch_names = raw_data.ch_names
        self.dataset_intra.times = raw_data.times

        self.file_label_intra.setText(
            f"Selected File: {basename(file_name)}"
            )
        self.dataset_intra.file_name = basename(file_name)
        self.dataset_intra.file_path = dirname(file_name)

        # Show channel selection and plot buttons for intracranial
        self.btn_select_channel_intra.setEnabled(True)
        self.channel_label_intra.setEnabled(True)
    
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to load .fif file: {e}")


def load_json_file(self, file_name: str):
    try:
        # Load the .json file and extract streams information
        # here write the logic to read json files, and create the dataframe
        # to show users all the different streams so that they can pick the one
        # they are interested in. Put all this in a separate function (dataframe creation)
        with open(file_name, 'r') as f:
            j = json.loads(f.read())        
        
        list_of_streamings = j['BrainSenseTimeDomain']
        n_streamings = len(list_of_streamings)

        list_of_stim_streamings = j['BrainSenseLfp']
        n_stim_streamings = len(list_of_stim_streamings)

        streamings_dict = defaultdict(lambda: defaultdict(dict))
            
        stream_times = [0]
        first_packet_time = None
        stream_count = 1

        for i_stream, dat in enumerate(list_of_streamings):
            print(f'Processing streaming {i_stream + 1} / {n_streamings}')
            print(f'Actual stream count: {stream_count}')
            first_packet_time = dat['FirstPacketDateTime']
            print(f'first_packet_time: {first_packet_time}')
            print(f'last stream time: {stream_times[-1]}')

            if first_packet_time != stream_times[-1] or i_stream == 0:
            # new stream
                #print('Entering new stream')
                streamings_dict[f'streaming_{stream_count}'][f'Channel_{dat["Channel"]}'] = {
                    'FirstPacketDateTime': first_packet_time,
                    'GlobalSequences': functions.utils.convert_list_string_floats(dat['GlobalSequences']),
                    'TicksInMses': functions.utils.convert_list_string_floats(dat['TicksInMses']),
                    'GlobalPacketSizes': functions.utils.convert_list_string_floats(dat['GlobalPacketSizes']),
                    'TimeDomainData': dat['TimeDomainData'], 
                    'SampleRateInHz': dat['SampleRateInHz']
                }
                stream_count += 1
            else:
            # other channel from same stream
                #print('Entering same stream, different channel')
                streamings_dict[f'streaming_{stream_count - 1}'][f'Channel_{dat["Channel"]}'] = {
                    'FirstPacketDateTime': dat['FirstPacketDateTime'],
                    'GlobalSequences': functions.utils.convert_list_string_floats(dat['GlobalSequences']),
                    'TicksInMses': functions.utils.convert_list_string_floats(dat['TicksInMses']),
                    'GlobalPacketSizes': functions.utils.convert_list_string_floats(dat['GlobalPacketSizes']),
                    'TimeDomainData': dat['TimeDomainData'],
                    'SampleRateInHz': dat['SampleRateInHz']
                }
            stream_times.append(first_packet_time)

        #  create a dataframe with all streamings and their channels
        ends = []
        prev_streaming_id = None
        stream_count = -1
        prev_stream_last_stim_ticks = None

        streamings_df = pd.DataFrame(columns=[
            'Streaming id', 'LFP Channels', 'LFP Recording start', 'LFP Recording end', 
            'LFP Recording duration', 
            #'First TicksInMses LFP', 'Last TicksInMses LFP', 
            # 'First TicksInMs stim', 'Last TicksInMs stim',
            # 'LFP time between rec based on First Packet Time',
            # 'LFP time between rec based on TicksInMses', 
            # 'LFP time between rec based on TicksInMs stim',
            # 'LFP Corrected for missing packets'
            ])
        for streaming_id in streamings_dict.keys():
            stream_count += 1
            channels = []
            time_since_last_rec_first_packet = None
            time_since_last_rec_ticks = None
            for channel in streamings_dict[streaming_id].keys():
                channels.append(channel)
                # check for missing packets and correct LFP data if necessary:
                print(f'Checking streaming {streaming_id}, channel {channel} for missing packets...')
                #correct_lfp_data, corrected_or_not = functions.utils.check_and_correct_missings_in_lfp(streamings_dict[streaming_id][channel])
                #streamings_dict[streaming_id][channel]['Corrected_TimeDomainData'] = correct_lfp_data

                ticks_in_ms = streamings_dict[streaming_id][channel]['TicksInMses']
                rec_dur_ms = ticks_in_ms[-1] - ticks_in_ms[0] + 250  # add 250 ms for last packet duration
                rec_dur_min, rec_dur_sec, rec_dur_msec = functions.utils.convert_msec_to_min_sec_msec(rec_dur_ms)
                dt_str = streamings_dict[streaming_id][channel]['FirstPacketDateTime']
                dt_obj = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S.%fZ')
                # compute rec_end_time using dt_obj + rec_duration: 
                rec_end_time = dt_obj +  timedelta(minutes=rec_dur_min, seconds=rec_dur_sec, milliseconds=rec_dur_msec)

                #time_since_last_rec = dt_obj - ends[-1] if ends else timedelta(0)
                if ends and streaming_id == prev_streaming_id:
                    time_since_last_rec = timedelta(0)
                elif ends:
                    time_since_last_rec = dt_obj - ends[-1]
                else:
                    time_since_last_rec = timedelta(0)

                dt1_parsed = datetime.strptime(streamings_dict[streaming_id][channel]['FirstPacketDateTime'], "%Y-%m-%dT%H:%M:%S.%fZ")
                # Simply output it in the same visual format as dt2
                dt1_reformatted = dt1_parsed.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # trim to .XXX milliseconds
                
                if time_since_last_rec_first_packet is None:
                    time_since_last_rec_first_packet = functions.utils.format_timedelta(time_since_last_rec)
                if time_since_last_rec_ticks is None:
                    time_since_last_rec_ticks = functions.utils.format_timedelta(timedelta(milliseconds=(ticks_in_ms[0] - (streamings_dict[prev_streaming_id][channel]['TicksInMses'][-1])) )) if ends and streaming_id != prev_streaming_id else '0 days, 0h, 0min, 0s, 0ms'
                prev_streaming_id = streaming_id
                # convert time_since_last_rec in milliseconds
                time_since_last_rec_ticks_ms = functions.utils.time_to_ms(time_since_last_rec_ticks)
                print(time_since_last_rec_ticks_ms)
                time_since_last_rec_first_packet_ms = functions.utils.time_to_ms(time_since_last_rec_first_packet)

                ends.append(rec_end_time)

            data = list_of_stim_streamings[stream_count]['LfpData']
            ticks = np.array([d['TicksInMs'] for d in data])
            right_mA = np.array([d['Right']['mA'] for d in data])
            left_mA = np.array([d['Left']['mA'] for d in data])
            time_since_last_rec_ticks_stim = (ticks[0] - 250) - prev_stream_last_stim_ticks if prev_stream_last_stim_ticks is not None else 0
            prev_stream_last_stim_ticks = ticks[-1]

            # streamings_df = streamings_df.append({
            # 'Streaming id': streaming_id,
            # 'LFP Channels': channels,
            # 'LFP Recording start': dt1_reformatted,
            # 'LFP Recording end': rec_end_time,
            # 'LFP Recording duration': f'{rec_dur_min} min, {rec_dur_sec} sec, {rec_dur_msec} ms',
            # # 'First TicksInMses LFP': ticks_in_ms[0],
            # # 'Last TicksInMses LFP': ticks_in_ms[-1],
            # # 'First TicksInMs stim': ticks[0],
            # # 'Last TicksInMs stim': ticks[-1],
            # # 'LFP time between rec based on First Packet Time': time_since_last_rec_first_packet_ms,
            # # 'LFP time between rec based on TicksInMses': time_since_last_rec_ticks_ms,
            # # 'LFP time between rec based on TicksInMs stim': time_since_last_rec_ticks_stim,
            # # 'LFP Corrected for missing packets': corrected_or_not
            # }, ignore_index=True)    
            new_row = pd.DataFrame([{
            'Streaming id': streaming_id,
            'LFP Channels': channels,
            'LFP Recording start': dt1_reformatted,
            'LFP Recording end': rec_end_time,
            'LFP Recording duration': f'{rec_dur_min} min, {rec_dur_sec} sec, {rec_dur_msec} ms',
            }])
            streamings_df = pd.concat([streamings_df, new_row], ignore_index=True)

        BrainSenseRaws = {}

        for i, stream in enumerate(streamings_dict.keys()):
            stream_dict = {}
            ch_names = []
            stim_ch_names = []
            data_arrays = []
            raw = streamings_dict[stream]
            for ch in raw.keys(): 
                ch_data = raw[ch]['TimeDomainData']
                ch_names.append(ch)
                #data_arrays.append(ch_data)
                data_arrays.append(np.array(ch_data) * 1e-6)
                side = 'Left' if 'LEFT' in ch else 'Right'
                stim_ch_names.append('STIM_' + ch)

        ### HERE ADD A PART TO GET THE STIMULATION CHANNELS DATA (+ THE FILTERED LFP DATA) ###
        # Because their sampling rate is different, resample them so that they match the LFP data length and crop if necessary (for small irregularities)
                
            stim_sf = j['BrainSenseLfp'][i]['SampleRateInHz']  # == 2Hz
            left_stim = []
            right_stim = []
            for k in range(len(j['BrainSenseLfp'][i]['LfpData'])):
                # get right and left mA values
                right_mA = j['BrainSenseLfp'][i]['LfpData'][k]['Right']['mA']
                left_mA = j['BrainSenseLfp'][i]['LfpData'][k]['Left']['mA']
                right_stim.append(right_mA)
                left_stim.append(left_mA)        
            stim_time = np.arange(len(right_stim)) / stim_sf      # e.g. every 0.5 s
            lfp_time  = np.arange(int(len(right_stim) * (250 / stim_sf))) / 250  # upsampled time

            # resample stim data to match lfp data length
            # Nearest-neighbor interpolation
            right_stim_resampled = np.interp(
                lfp_time,
                stim_time,
                right_stim,
                left=None,
                right=None,
            )

            left_stim_resampled = np.interp(
                lfp_time,
                stim_time,
                left_stim,
                left=None,
                right=None,
            )    

            # add some NaNs if resampled stim data is shorter than lfp data
            lfp_data_length = len(raw[ch]['TimeDomainData'])
            if len(left_stim_resampled) < lfp_data_length:
                left_stim_resampled = np.concatenate([np.nan * np.ones(lfp_data_length - len(left_stim_resampled)), left_stim_resampled])
            elif len(left_stim_resampled) > lfp_data_length:
                left_stim_resampled = left_stim_resampled[-lfp_data_length:]
            if len(right_stim_resampled) < lfp_data_length:
                right_stim_resampled = np.concatenate([np.nan * np.ones(lfp_data_length - len(right_stim_resampled)), right_stim_resampled])
            elif len(right_stim_resampled) > lfp_data_length:
                right_stim_resampled = right_stim_resampled[-lfp_data_length:]            
            left_stim_resampled_scaled = np.array(left_stim_resampled) * 1e-6
            right_stim_resampled_scaled = np.array(right_stim_resampled) * 1e-6
            data_arrays.append(left_stim_resampled_scaled)
            data_arrays.append(right_stim_resampled_scaled)

            info = mne.create_info(
            ch_names=ch_names + stim_ch_names,
            sfreq=250,
            ch_types=['eeg'] * len(ch_names) + ['eeg'] * len(stim_ch_names)
            )

            raw = mne.io.RawArray(
                data = np.array(data_arrays),
                info = info
            )
            #raw.plot(scalings='auto')
            #raw.save(os.path.join(saving_path, saving_names[i]), overwrite=True)
            BrainSenseRaws[stream] = raw                

        # Create a pop-up window to show the data frame and let user select the stream they want to load
        selected_streams = self.show_stream_selection_dialog(streamings_df)
        if not selected_streams:
            QMessageBox.warning(self, "No Selection", "No stream was selected.")
            return

        # Filter BrainSenseRaws based on user selection
        BrainSenseRaws = {k: v for k, v in BrainSenseRaws.items() if k in selected_streams}

        # There should be only one selected stream
        selected_stream_id = selected_streams[0]

        # Assign the corresponding MNE Raw object to your dataset
        self.dataset_intra.raw_data = BrainSenseRaws[selected_stream_id]

        # Optionally, also store related info for convenience
        self.dataset_intra.sf = self.dataset_intra.raw_data.info['sfreq']
        self.dataset_intra.ch_names = self.dataset_intra.raw_data.ch_names
        self.dataset_intra.times = np.linspace(
            0,
            self.dataset_intra.raw_data.n_times / self.dataset_intra.sf,
            self.dataset_intra.raw_data.n_times
        )

        # Update the GUI label
        self.file_label_intra.setText(f"Selected File: {basename(file_name)} (Stream: {selected_stream_id})")

        # Enable plotting/select channel buttons
        self.btn_select_channel_intra.setEnabled(True)
        self.channel_label_intra.setEnabled(True)

        # self.dataset_intra.raw_data = raw_data  # Assign to dataset
        # # self.dataset_intra.sf = raw_data.info["sfreq"]  # Assign sampling frequency
        # # self.dataset_intra.ch_names = raw_data.ch_names  # Assign channel names#
        # # self.dataset_intra.times = np.linspace(
        # #     0, raw_data.get_data().shape[1]/self.dataset_intra.sf, 
        # #     raw_data.get_data().shape[1]
        # #     )
        # self.file_label_intra.setText(
        #     f"Selected File: {basename(file_name)}"
        #     )
        # self.dataset_intra.file_name = basename(file_name)
        # self.dataset_intra.file_path = dirname(file_name)

        # # Show channel selection and plot buttons for intracranial
        # self.btn_select_channel_intra.setEnabled(True)
        # self.channel_label_intra.setEnabled(True)

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to load .json file: {e}")



def load_ext_file(self):
    """Load external file. Supported file formats are .xdf, .fif, .poly5"""
    file_name, _ = QFileDialog.getOpenFileName(
        self, "Select External File", "", 
        "XDF Files (*.xdf);;FIF Files (*.fif);;Poly5 Files (*.Poly5)"
        )
    self.file_label_xdf.setText(f"Selected File: {basename(file_name)}")
    self.dataset_extra.file_name = basename(file_name)
    self.dataset_extra.file_path = dirname(file_name)
    
    if file_name.endswith(".xdf"):
        xdf_datas = resolve_streams(file_name)
        streams_list = [stream['name'] for stream in xdf_datas]
        self.dataset_extra.list_of_streams = streams_list

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Stream containing sync pulses")
        layout = QVBoxLayout()

        list_widget = QListWidget()
        list_widget.addItems(streams_list)
        layout.addWidget(list_widget)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        # Connect buttons
        def accept_selection():
            selected_items = list_widget.selectedItems()
            if selected_items:
                dialog.selected_stream = selected_items[0].text()
                self.dataset_extra.selected_stream = selected_items[0].text()
                dialog.accept()
            else:
                QMessageBox.warning(
                    dialog, "No Selection", "Please select a stream."
                    )
        
        ok_button.clicked.connect(accept_selection)
        cancel_button.clicked.connect(dialog.reject)

        dialog.setLayout(layout)

        # Show dialog and wait
        if dialog.exec_() == QDialog.Accepted:
            stream_name = dialog.selected_stream
            load_xdf_file(self, file_name, stream_name)
        
        else:
            QMessageBox.information(
                self, "Stream Selection", "Stream selection was cancelled."
                )

    elif file_name.endswith(".Poly5"):
        load_poly5_file(self, file_name)

    elif file_name.endswith(".fif"):
        load_fif_file_ext(self, file_name)


def load_poly5_file(
        self, 
        file_name: str
        ):
    """Load .poly5 file."""
    try:
        TMSi_data = Poly5Reader(file_name)
        toMNE = True
        TMSi_rec = TMSi_data.read_data_MNE()
        self.dataset_extra.raw_data = TMSi_rec
        self.dataset_extra.sf = TMSi_rec.info["sfreq"]  # Get the sampling frequency
        self.dataset_extra.ch_names = TMSi_rec.ch_names  # Get the channel names
        self.dataset_extra.times = TMSi_rec.times # Get the timescale

        # Show channel selection and plot buttons for .xdf
        self.channel_label_xdf.setEnabled(True)
        self.btn_select_channel_xdf.setEnabled(True)
        self.btn_select_ecg_channel.setEnabled(True)
        #self.ecg_channel_label.setEnabled(True)
        
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to load .poly5 file: {e}")


def load_xdf_file(
        self, 
        file_name: str, 
        stream_name: str = 'SAGA'
        ):
    """Load .xdf file."""
    try:
        # Load the .xdf file using the read_raw function
        stream_id = find_sync_stream(self, file_name, stream_name=stream_name)
        raw_data = read_raw(file_name, stream_ids=[stream_id], preload=True)

        self.dataset_extra.raw_data = raw_data
        self.dataset_extra.sf = raw_data.info["sfreq"]
        self.dataset_extra.ch_names = raw_data.ch_names
        self.dataset_extra.times = raw_data.times

        # Enable channel selection and plot buttons
        self.channel_label_xdf.setEnabled(True)
        self.btn_select_channel_xdf.setEnabled(True)
        self.btn_select_ecg_channel.setEnabled(True)

    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to load .xdf file: {e}")


def load_fif_file_ext(
        self,
        file_name: str,
        ):
    """Load .fif file."""
    try:
        raw_data = mne.io.read_raw_fif(file_name, preload=True)

        self.dataset_extra.raw_data = raw_data
        self.dataset_extra.sf = raw_data.info["sfreq"]
        self.dataset_extra.ch_names = raw_data.ch_names
        self.dataset_extra.times = raw_data.times

        # Enable channel selection and plot buttons
        self.channel_label_xdf.setEnabled(True)
        self.btn_select_channel_xdf.setEnabled(True)
        self.btn_select_ecg_channel.setEnabled(True)
    
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to load .fif file: {e}")





def find_sync_stream(
        self, 
        fpath_external: str, 
        stream_name: str
        ):
    """Find the stream containing the channel with the synchronization pulses in the .xdf file."""
    xdf_datas = resolve_streams(fpath_external)
    streams_dict = {stream['name']: stream['stream_id'] for stream in xdf_datas}
    stream_id = streams_dict.get(stream_name)

    if stream_id is None:
        raise ValueError(f"Stream '{stream_name}' not found in the XDF file.")
    
    return stream_id




##############################  OUTPUT FUNCTIONS  ##############################
def select_saving_folder(self):
    """Open a dialog to select a folder for saving synchronized datasets."""
    # Open a QFileDialog to select a folder
    folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
    self.folder_path = folder_path
    self.label_saving_folder.setText(f"Results will be saved in: {folder_path}")
    
    if folder_path:  # Check if the user selected a folder
        print(f"Selected folder: {folder_path}")


def save_datasets(self):
    saving_format_internal = self.config["SavingFormatInternal"]

    if self.config['NoSync'] == True:
        print('true')
        if saving_format_internal == ".set":
            save_int_as_set(self)
        elif saving_format_internal == ".fif":
            save_int_as_fif(self)
        elif saving_format_internal == ".pkl":
            save_int_as_pickle(self)
        elif saving_format_internal == ".mat":
            save_int_as_mat(self)
    else:
        # Check dataformat of external recording
        ext_extension = self.dataset_extra.file_name.split('.')[-1]

        # Determine saving format from config
        saving_format_external = self.config["SavingFormatExternal"]
        saving_format_all_as_one = self.config["SavingFormatAllAsOne"]
        saving_all_as_one = self.config["SavingAllAsOne"]

        if ext_extension == "xdf":
            if saving_all_as_one:
                if saving_format_all_as_one == ".pkl":
                    synchronize_datasets_as_one_pickle(self)
                else:
                    QMessageBox.warning(
                        self, "Saving Format Error", 
                        f"Unsupported saving format for all-in-one: {saving_format_all_as_one} \nOnly .pkl is supported."
                        )
            else:
                if saving_format_external == ".set" and saving_format_internal == ".set":
                    save_datasets_as_set(self)
                elif saving_format_external == ".fif" and saving_format_internal == ".fif":
                    save_datasets_as_fif(self)  
                elif saving_format_external == ".pkl" and saving_format_internal == ".pkl":
                    synchronize_datasets_as_pickles(self)
                else:
                    QMessageBox.warning(
                        self, "Saving Format Error", 
                        f"Unsupported saving formats: External {saving_format_external}, Internal {saving_format_internal}"
                        )   
        else:
            if saving_format_external == ".mat" and saving_format_internal == ".mat":
                    synchronize_datasets_as_mat(self)
            else:
                QMessageBox.warning(
                    self, "Saving Format Error", 
                    f"Unsupported saving formats for external file type {ext_extension}: External {saving_format_external}, Internal {saving_format_internal} \nOnly .mat is supported for non-xdf external files."
                    )



def save_int_as_set(self):
    if self.dataset_intra.flag_cleaned == True:
        lfp_title = (
            "INTRACRANIAL_CLEANED_" + str(self.dataset_intra.file_name[:-4]) + ".set"
            )
    else:
        lfp_title = (
            "INTRACRANIAL_" + str(self.dataset_intra.file_name[:-4]) + ".set"
            )

    if self.folder_path is not None:
        fname_lfp_out =join(self.folder_path, lfp_title)
    else:
        fname_lfp_out = lfp_title
    
    lfp_timescale = np.linspace(
        0, self.dataset_intra.raw_data.get_data().shape[1]/self.dataset_intra.sf, 
        self.dataset_intra.raw_data.get_data().shape[1]
        )
    
    write_set(
        fname = fname_lfp_out, 
        raw = self.dataset_intra.raw_data, 
        fs = self.dataset_intra.sf,
        times = lfp_timescale
        )

    QMessageBox.information(
        self, "Saving",
        "Saving done. Intracranial file saved as .SET"
        )


def save_int_as_fif(self):
    if self.dataset_intra.flag_cleaned == True:
        lfp_title = (
            "INTRACRANIAL_CLEANED_" + str(self.dataset_intra.file_name[:-4]) + "_raw.fif"
            )
    else:
        lfp_title = (
            "INTRACRANIAL_" + str(self.dataset_intra.file_name[:-4]) + "_raw.fif"
            )

    if self.folder_path is not None:
        fname_lfp_out =join(self.folder_path, lfp_title)
    else:
        fname_lfp_out = lfp_title

    # lfp_timescale = np.linspace(
    #     0, self.dataset_intra.raw_data.get_data().shape[1]/self.dataset_intra.sf, 
    #     self.dataset_intra.raw_data.get_data().shape[1]
    #     )
    
    data_lfp = self.dataset_intra.raw_data.get_data()
    ch_names = self.dataset_intra.raw_data.info['ch_names']

    # Create a new Info object with the correct sampling frequency
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=self.dataset_intra.sf,
        ch_types='eeg'
    )

    # Create a new Raw object using the corrected info
    lfp_rec = mne.io.RawArray(data_lfp, info)

    # Save the corrected Raw
    lfp_rec.save(fname_lfp_out, overwrite=True)


    QMessageBox.information(
        self, "Saving",
        "Saving done. Intracranial file saved as .fif"
        )


def save_int_as_pickle(self):
    if self.dataset_intra.flag_cleaned == True:
        lfp_title = (
            "INTRACRANIAL_CLEANED_" + str(self.dataset_intra.file_name[:-4]) + ".pkl"
            )
    else:
        lfp_title = (
            "INTRACRANIAL_" + str(self.dataset_intra.file_name[:-4]) + ".pkl"
            )
    
    LFP_array = self.dataset_intra.raw_data.get_data()
    LFP_df = pd.DataFrame(LFP_array.T)
    LFP_df.columns = self.dataset_intra.ch_names
    LFP_timescale = self.dataset_intra.times

    # Save as pickle file:
    LFP_df["sf_LFP"] = self.dataset_intra.sf
    LFP_df["time_stamp"] = LFP_timescale

    if self.folder_path is not None:
        LFP_filename = join(self.folder_path, lfp_title)
    else: LFP_filename = lfp_title
    # Save the dataset to a pickle file
    with open(LFP_filename, "wb") as file:
        pickle.dump(LFP_df, file)
    
    QMessageBox.information(
        self, "Saving", 
        "Saving done. Intracranial file saved as .pkl"
        )


def save_int_as_mat(self):
    if self.dataset_intra.flag_cleaned == True:
        lfp_title = (
            "INTRACRANIAL_CLEANED_" + str(self.dataset_intra.file_name[:-4]) + ".mat"
            )
    else:
        lfp_title = (
            "INTRACRANIAL_" + str(self.dataset_intra.file_name[:-4]) + ".mat"
            )
        
    if self.folder_path is not None:
        LFP_filename = join(self.folder_path, lfp_title)
    else:
        LFP_filename = lfp_title

    LFP_array = self.dataset_intra.raw_data.get_data()
    LFP_df = pd.DataFrame(LFP_array.T)
    LFP_df.columns = self.dataset_intra.ch_names

    savemat(
        LFP_filename,
        {
            "data": LFP_df.T,
            "fsample": self.dataset_intra.sf,
            "label": np.array(
                LFP_df.columns.tolist(), dtype=object
            ).reshape(-1, 1),
        },
    )    
    QMessageBox.information(
        self, "Saving", 
        "Saving done. Intracranial file saved as .mat"
        )

def save_datasets_as_set(self):
    """
    Save synchronized datasets as .set files.
    This function synchronizes the intracranial and external datasets,
    crops them so that they start 1 second before the first artifact,
    and saves them as 2 separate .set files.
    It also extracts events from annotations in the external dataset.
    The events are then transferred to the intracranial dataset.
    The function saves the synchronized datasets in the specified folder.
    If no folder is selected, it saves them in the current working directory.
    """
    print("events from annotations extraction")
    events, _ = mne.events_from_annotations(self.dataset_extra.raw_data)
    inv_dic = {v: str(k) for k, v in _.items()}

    ## offset intracranial recording (crop everything that is more than 1s before the artifact)
    new_start_intracranial = self.dataset_intra.art_start - 1
    lfp_rec_offset = self.dataset_intra.synced_data

    ## offset external recording (crop everything that is more than 1s before the artifact)
    new_start_external = self.dataset_extra.art_start - 1
    TMSi_rec_offset = self.dataset_extra.synced_data

    ## transfer of the events from the external to the intracranial recording
    # create a duplicate of the events to manipulate it without changing the external one
    events_lfp = deepcopy(events)

    # get the events from the external in time instead of samples to account 
    # for the different sampling frequencies
    events_in_time = events[:,0]/self.dataset_extra.sf

    # then offset the events in time to the new start of the external recording
    events_in_time_offset = events_in_time - new_start_external

    if self.dataset_intra.eff_sf is not None:
        lfp_sf = self.dataset_intra.eff_sf
    else:
        lfp_sf = self.dataset_intra.sf

    # convert the events in time offset to samples corresponding to 
    # the sampling frequency of the intracranial recording
    # because the annotations object works with samples, not timings
    events_in_time_offset_lfp = events_in_time_offset * lfp_sf
    events_lfp[:,0] = events_in_time_offset_lfp

    ## create an annotation object for the intracranial recording
    annotations_lfp = mne.annotations_from_events(
        events_lfp, sfreq=lfp_sf, event_desc=inv_dic
        )

    lfp_rec_offset.set_annotations(None) # make sure that no annotations are present
    lfp_rec_offset.set_annotations(annotations_lfp) # set the new annotations

    external_title = (
        "SYNCHRONIZED_EXTERNAL_" + str(self.dataset_extra.file_name[:-4]) + ".set"
        )

    if self.dataset_intra.flag_cleaned == True:
        lfp_title = (
            "SYNCHRONIZED_INTRACRANIAL_CLEANED_" + str(self.dataset_intra.file_name[:-4]) + ".set"
            )
    else:
        lfp_title = (
            "SYNCHRONIZED_INTRACRANIAL_" + str(self.dataset_intra.file_name[:-4]) + ".set"
            )

    if self.folder_path is not None:
        fname_external_out=join(self.folder_path, external_title)
        fname_lfp_out =join(self.folder_path, lfp_title)
    else:
        fname_external_out = external_title
        fname_lfp_out = lfp_title

    TMSi_rec_offset_annotations_onset = TMSi_rec_offset.annotations.onset - new_start_external
    lfp_rec_offset_annotations_onset= lfp_rec_offset.annotations.onset - new_start_intracranial
    
    lfp_timescale = np.linspace(
        0, self.dataset_intra.synced_data.get_data().shape[1]/lfp_sf, 
        self.dataset_intra.synced_data.get_data().shape[1]
        )
    
    write_set(
        fname = fname_external_out, 
        raw = TMSi_rec_offset, 
        fs = TMSi_rec_offset.info['sfreq'],
        times = TMSi_rec_offset.times,
        annotations_onset = TMSi_rec_offset_annotations_onset
        )
    write_set(
        fname = fname_lfp_out, 
        raw = lfp_rec_offset, 
        fs = lfp_sf,
        times = lfp_timescale,
        annotations_onset = lfp_rec_offset_annotations_onset
        )

    QMessageBox.information(
        self, "Synchronization",
        "Synchronization done. Both files saved as .SET"
        )



def synchronize_datasets_as_pickles(self):
    """
    This function synchronizes intracranial and external datasets,
    crops them so that they start 1 second before the first artifact,
    and saves them as separate .pkl files. If the external dataset contains multiple streams,
    it processes each stream individually and saves them as separate .pkl files.
    The function also offsets the timestamps of the external streams to align with the intracranial dataset
    and saves them in the specified folder. If no folder is selected, it saves them in the current working directory.
    """ 
    ## Intracranial ##
    # Crop beginning of LFP intracranial recording 1 second before first artifact:
    time_start_LFP_0 = self.dataset_intra.art_start - 1  # 1s before first artifact
    index_start_LFP = time_start_LFP_0 * (self.dataset_intra.sf)

    LFP_array = self.dataset_intra.synced_data.get_data()

    LFP_df_offset = pd.DataFrame(LFP_array.T)
    LFP_df_offset.columns = self.dataset_intra.ch_names
    LFP_timescale_offset_s = self.dataset_intra.times[int(index_start_LFP):] - time_start_LFP_0

    # Save as pickle file:
    LFP_df_offset["sf_LFP"] = self.dataset_intra.sf
    LFP_df_offset["time_stamp"] = LFP_timescale_offset_s
    lfp_title = (
        "SYNCHRONIZED_INTRACRANIAL_" + str(self.dataset_intra.file_name[:-4]) + ".pkl"
        )
    if self.folder_path is not None:
        LFP_filename = join(self.folder_path, lfp_title)
    else: LFP_filename = lfp_title
    # Save the dataset to a pickle file
    with open(LFP_filename, "wb") as file:
        pickle.dump(LFP_df_offset, file)

    ## External##
    stream_names = []
    stream_ids = []
    streams_dict = {}

    filepath = join(self.dataset_extra.file_path, self.dataset_extra.file_name)
    xdf_datas = XdfData(filepath).resolve_streams()

    for streams in range(1, len(xdf_datas['name'])+1, 1):
        stream_names.append(xdf_datas['name'][streams])

    for name in stream_names:
        stream_ids.append(xdf_datas[xdf_datas['name'] == name].index[0])

    for stream_nb in zip(stream_names, stream_ids):
        # create a dictionnary associating each streaming name from 
        # stream_names list to its corresponding stream_id:
        streams_dict[stream_nb[0]] = stream_nb[1]

    streams = {}

    for _, (name, stream_id) in enumerate(streams_dict.items(), start=1):
        stream_name = f"{name}_stream"
        streams[stream_name] = XdfData(filepath).load(stream_id=[stream_id])

    # CREATE GLOBAL VARIABLES FOR EACH STREAM
    for stream_name, stream_data in streams.items():
        globals()[stream_name] = stream_data.data()
        print(stream_name)

    # Convert self.dataset_extra.art_start into the xdf timescale from the BIP data
    art_start_0 = self.dataset_extra.art_start - 1
    # Get the original timestamps from both sources
    timestamps_global = globals()[
        f'{self.dataset_extra.selected_stream}_stream'
        ]['time_stamp']
    times_real = self.dataset_extra.times

    # Find the index in self.dataset_extra.times corresponding to art_start_0
    art_start_index = (times_real >= art_start_0).argmax()

    # Filter the timestamps from the global clock based on this index
    filtered_timestamps_global = timestamps_global[art_start_index:]
    art_start_in_globals = np.array(filtered_timestamps_global.iloc[0])

    # Iterate over the dynamically created variables
    for stream_name in streams.keys():
        # Find the index corresponding to the value of art_start_in_globals
        index = np.argmax(
            globals()[stream_name]['time_stamp'] >= art_start_in_globals
            )
        
        # Crop the stream data from the index onwards
        stream_offset = globals()[stream_name].iloc[index:]
        
        # Create a copy of the cropped stream data
        stream_offset_copy = stream_offset.copy()
        
        # Offset the 'time_stamp' column
        stream_offset_copy['time_stamp'] = stream_offset_copy[
            'time_stamp'] - art_start_in_globals
        
        # Reset the index
        stream_offset_copy.reset_index(drop=True, inplace=True)
        
        # Update the global variable with the modified data
        globals()[stream_name] = stream_offset_copy

    # Create a dictionary to hold the extracted DataFrames
    extracted_streams = {}

    # Iterate over the dynamically created variables
    for stream_name in streams.keys():
        # Extract the current stream DataFrame
        extracted_streams[f"df_{stream_name}"] = globals()[stream_name].copy()

    ## saving as pickle:
    # Iterate over the extracted DataFrames
    for df_name, df_data in extracted_streams.items():
        # Generate the filename
        external_title = (
            f"{df_name}SYNCHRONIZED_EXTERNAL_" + str(self.dataset_extra.file_name[:-4]) + ".pkl"
            )
        # Create the full path to the file
        if self.folder_path is not None:
            filepath = join(self.folder_path, external_title)
        else:
            filepath = external_title
        # Save the DataFrame to a pickle file
        df_data.to_pickle(filepath)
    QMessageBox.information(
        self, "Synchronization", 
        "Synchronization done. All files saved separately as .pickle"
        )



def synchronize_datasets_as_one_pickle(self):
    """
    This function synchronizes intracranial and external datasets,
    crops them so that they start 1 second before the first artifact,
    and saves them as ONE .pkl file. If the external dataset contains multiple streams,
    it processes each stream individually to re-create their own timescale.
    The function saves the sycnhronized dataset in the specified folder.
    If no folder is selected, it saves it in the current working directory.
    """
    ## Intracranial ##
    # Crop beginning of LFP intracranial recording 1 second before first artifact:
    time_start_LFP_0 = self.dataset_intra.art_start - 1  # 1s before first artifact
    index_start_LFP = time_start_LFP_0 * (self.dataset_intra.sf)

    LFP_array = self.dataset_intra.synced_data.get_data()

    LFP_df_offset = pd.DataFrame(LFP_array.T)
    LFP_df_offset.columns = self.dataset_intra.ch_names
    LFP_timescale_offset_s = self.dataset_intra.times[
        int(index_start_LFP):] - time_start_LFP_0

    # Prepare LFP dataframe
    LFP_df_offset["sf_LFP"] = self.dataset_intra.sf
    LFP_df_offset["time_stamp"] = LFP_timescale_offset_s

    ## External ##
    stream_names = []
    stream_ids = []
    streams_dict = {}

    filepath = join(self.dataset_extra.file_path, self.dataset_extra.file_name)
    xdf_datas = XdfData(filepath).resolve_streams()

    for streams in range(1, len(xdf_datas['name'])+1, 1):
        stream_names.append(xdf_datas['name'][streams])

    for name in stream_names:
        stream_ids.append(xdf_datas[xdf_datas['name'] == name].index[0])

    for stream_nb in zip(stream_names, stream_ids):
        # create a dictionnary associating each streaming name from 
        # stream_names list to its corresponding stream_id:
        streams_dict[stream_nb[0]] = stream_nb[1]

    # LOAD ALL STREAMS IN A DICTIONNARY
    streams = {}

    for _, (name, stream_id) in enumerate(streams_dict.items(), start=1):
        stream_name = f"{name}_stream"
        streams[stream_name] = XdfData(filepath).load(stream_id=[stream_id])

    # CREATE GLOBAL VARIABLES FOR EACH STREAM
    for stream_name, stream_data in streams.items():
        globals()[stream_name] = stream_data.data()
        print(stream_name)

    # Convert self.dataset_extra.art_start into the xdf timescale from the BIP data
    art_start_0 = self.dataset_extra.art_start - 1
    # Get the original timestamps from both sources
    timestamps_global = globals()[
        f'{self.dataset_extra.selected_stream}_stream']['time_stamp']
    times_real = self.dataset_extra.times

    # Find the index in self.dataset_extra.times corresponding to art_start_0
    art_start_index = (times_real >= art_start_0).argmax()

    # Filter the timestamps from the global clock based on this index
    filtered_timestamps_global = timestamps_global[art_start_index:]
    art_start_in_globals = np.array(filtered_timestamps_global.iloc[0])

    # Iterate over the dynamically created variables
    for stream_name in streams.keys():
        # Find the index corresponding to the value of art_start_in_globals
        index = np.argmax(
            globals()[stream_name]['time_stamp'] >= art_start_in_globals
            )

        # Crop the stream data from the index onwards
        stream_offset = globals()[stream_name].iloc[index:]

        # Create a copy of the cropped stream data
        stream_offset_copy = stream_offset.copy()

        # Offset the 'time_stamp' column
        stream_offset_copy['time_stamp'] = stream_offset_copy[
            'time_stamp'] - art_start_in_globals
        
        # Reset the index
        stream_offset_copy.reset_index(drop=True, inplace=True)
        
        # Update the global variable with the modified data
        globals()[stream_name] = stream_offset_copy

    # Create a dictionary to hold the extracted DataFrames
    extracted_streams = {}

    # Iterate over the dynamically created variables
    for stream_name in streams.keys():
        # Extract the current stream DataFrame
        extracted_streams[f"df_{stream_name}"] = globals()[stream_name].copy()

    # Iterate over the extracted DataFrames
    for df_name, df_data in extracted_streams.items():
        # Create separate DataFrame variables with specific names
        globals()[df_name] = pd.DataFrame(df_data)

    # Create an empty list to hold the DataFrames
    all_dfs = []

    # Iterate over the extracted DataFrames
    for df_name, df_data in extracted_streams.items():
        # Create a DataFrame with a MultiIndex containing the df_name as the top level
        df = pd.DataFrame(df_data)
        # Add df_name as the top level of column index
        df.columns = pd.MultiIndex.from_product([[df_name], df.columns])  
        all_dfs.append(df)

    # Concatenate all DataFrames in the list along axis 1 (columns)
    LSL_df = pd.concat(all_dfs, axis=1)

    # Assuming LFP_df_offset is your new DataFrame
    # First, adjust its column names to include a MultiIndex with the header 'LFP'
    LFP_df_offset.columns = pd.MultiIndex.from_product([['df_LFP'], LFP_df_offset.columns])

    # Concatenate LFP_df_offset on top of big_df along axis 1 (columns)
    final_df = pd.concat([LFP_df_offset, LSL_df], axis=1)
    
    ## saving as pickle:
    # Generate the filename
    filename = f"{self.dataset_extra.file_name[:-4]}_synchronized_data.pkl"
    # Create the full path to the file
    if self.folder_path is not None:
        filepath = join(self.folder_path, filename)
    else:
        filepath = filename
    # Save the DataFrame to a pickle file
    final_df.to_pickle(filepath)
    print(f"DataFrame {filename} saved as pickle to {filepath}")
    QMessageBox.information(
        self, "Synchronization", 
        "Synchronization done. Everything saved as one .pickle file"
        )

def synchronize_datasets_as_mat(self):
    """
    MIGHT BE OUT OF DATE
    This function synchronizes intracranial and external datasets,
    crops them so that they start 1 second before the first artifact,
    and saves them as two separate .mat files.
    The function crops the beginning of both datasets to ensure they start at the same time,
    and then synchronizes them based on their respective durations.
    The synchronized datasets are saved in the specified folder.
    If no folder is selected, they are saved in the current working directory.
    """
    index_start_LFP = (self.dataset_intra.art_start - 1) * self.dataset_intra.sf
    LFP_array = self.dataset_intra.raw_data.get_data()
    LFP_cropped = LFP_array[:, int(index_start_LFP) :].T

    ## External ##
    # Crop beginning of external recordings 1s before first artifact:
    time_start_external = (self.dataset_extra.art_start) - 1
    index_start_external = time_start_external * self.dataset_extra.sf
    external_file = self.dataset_extra.raw_data.get_data()
    external_cropped = external_file[:, int(index_start_external) :].T

    # Check which recording is the longest,
    # crop it to give it the same duration as the other one:
    LFP_rec_duration = len(LFP_cropped) / self.dataset_intra.sf
    external_rec_duration = len(external_cropped) / self.dataset_extra.sf

    if LFP_rec_duration > external_rec_duration:
        index_stop_LFP = external_rec_duration * self.dataset_intra.sf
        LFP_synchronized = LFP_cropped[: int(index_stop_LFP), :]
        external_synchronized = external_cropped
    elif external_rec_duration > LFP_rec_duration:
        index_stop_external = LFP_rec_duration * self.dataset_extra.sf
        external_synchronized = external_cropped[: int(index_stop_external), :]
        LFP_synchronized = LFP_cropped
    else:
        LFP_synchronized = LFP_cropped
        external_synchronized = external_cropped  

    # save the synchronized data in mat format          
    LFP_df_offset = pd.DataFrame(LFP_synchronized)
    LFP_df_offset.columns = self.dataset_intra.ch_names
    external_df_offset = pd.DataFrame(external_synchronized)
    external_df_offset.columns = self.dataset_extra.ch_names

    lfp_title = (
        "SYNCHRONIZED_INTRACRANIAL_" + str(self.dataset_intra.file_name[:-4]) + ".mat"
        )
    external_title = (
        f"SYNCHRONIZED_EXTERNAL_" + str(self.dataset_extra.file_name[:-6]) + ".mat"
        )
    
    if self.folder_path is not None:
        LFP_filename = join(self.folder_path, lfp_title)
        external_filename = join(self.folder_path, external_title)
    else:
        LFP_filename = lfp_title
        external_filename = external_title

    savemat(
        LFP_filename,
        {
            "data": LFP_df_offset.T,
            "fsample": self.dataset_intra.sf,
            "label": np.array(
                LFP_df_offset.columns.tolist(), dtype=object
            ).reshape(-1, 1),
        },
    )
    savemat(
        external_filename,
        {
            "data": external_df_offset.T,
            "fsample": self.dataset_extra.sf,
            "label": np.array(
                external_df_offset.columns.tolist(), dtype=object
            ).reshape(-1, 1),
        },
    )
    QMessageBox.information(
        self, "Synchronization", "Synchronization done. Both files saved as .mat files"
        )       


def write_set(
    fname: str,
    raw: mne.io.BaseRaw,
    fs: float,
    times: np.ndarray,
    annotations_onset: float | None = None,
):
    """
    Export synchronized recordings to EEGLAB .set files.

    Parameters
    ----------
    fname : str
        Output filename (without .mat extension).
    raw : mne.io.BaseRaw
        MNE Raw object containing data and annotations.
    fs : float
        Sampling frequency.
    times : np.ndarray
        Time vector corresponding to samples.
    annotations_onset : float | None, optional
        Global onset time of annotations (in seconds). If None, events will be omitted.
    """
    data = raw.get_data()
    ch_names = raw.info["ch_names"]
    chanlocs = fromarrays([ch_names], names=["labels"])

    # --- Handle events safely ---
    if hasattr(raw, "annotations") and len(raw.annotations) > 0:
        if annotations_onset is None:
            annotations_onset = 0.0  # default if missing
        events = fromarrays([
            raw.annotations.description,
            annotations_onset * fs + 1,
            raw.annotations.duration * fs
        ], names=["type", "latency", "duration"])
    else:
        # Create empty structured array if no annotations
        events = fromarrays([[], [], []], names=["type", "latency", "duration"])

    # --- Prepare EEGLAB structure ---
    EEG = dict(
        data=data,
        setname=fname,
        nbchan=data.shape[0],
        pnts=data.shape[1],
        trials=1,
        srate=fs,
        xmin=times[0],
        xmax=times[-1],
        chanlocs=chanlocs,
        event=events,
        icawinv=[],
        icasphere=[],
        icaweights=[],
    )

    # --- Save .set file ---
    savemat(fname, {"EEG": EEG}, appendmat=False)

# def write_set(
#         fname: str, 
#         raw: mne.io.BaseRaw, 
#         annotations_onset: float, 
#         fs: float, 
#         times: np.ndarray
#         ):
#     """Export synchronized recordings to EEGLAB .set files."""
#     data = raw.get_data()
#     ch_names = raw.info["ch_names"]
#     chanlocs = fromarrays([ch_names], names=["labels"])
#     events = fromarrays([raw.annotations.description,
#                          annotations_onset * fs + 1,
#                          raw.annotations.duration * fs],
#                         names=["type", "latency", "duration"])
#     savemat(fname, dict(EEG=dict(data=data,
#                                  setname=fname,
#                                  nbchan=data.shape[0],
#                                  pnts=data.shape[1],
#                                  trials=1,
#                                  srate=fs,
#                                  xmin=times[0],
#                                  xmax=times[-1],
#                                  chanlocs=chanlocs,
#                                  event=events,
#                                  icawinv=[],
#                                  icasphere=[],
#                                  icaweights=[])),
#             appendmat=False)


def save_datasets_as_fif(self):
    """
    Save synchronized datasets as .fif files.
    This function synchronizes the intracranial and external datasets,
    crops them so that they start 1 second before the first artifact,
    and saves them as 2 separate .fif files.
    It also extracts events from annotations in the external dataset.
    The events are then transferred to the intracranial dataset.
    The function saves the synchronized datasets in the specified folder.
    If no folder is selected, it saves them in the current working directory.
    """
    print("events from annotations extraction")
    events, _ = mne.events_from_annotations(self.dataset_extra.raw_data)
    inv_dic = {v: str(k) for k, v in _.items()}

    ## offset intracranial recording (crop everything that is more than 1s before the artifact)
    new_start_intracranial = self.dataset_intra.art_start - 1
    lfp_rec_offset = self.dataset_intra.synced_data

    ## offset external recording (crop everything that is more than 1s before the artifact)
    new_start_external = self.dataset_extra.art_start - 1
    TMSi_rec_offset = self.dataset_extra.synced_data

    ## transfer of the events from the external to the intracranial recording
    # create a duplicate of the events to manipulate it without changing the external one
    events_lfp = deepcopy(events)

    # get the events from the external in time instead of samples to account 
    # for the different sampling frequencies
    events_in_time = events[:,0]/self.dataset_extra.sf

    # then offset the events in time to the new start of the external recording
    events_in_time_offset = events_in_time - new_start_external

    if self.dataset_intra.eff_sf is not None:
        lfp_sf = self.dataset_intra.eff_sf
    else:
        lfp_sf = self.dataset_intra.sf

    # convert the events in time offset to samples corresponding to 
    # the sampling frequency of the intracranial recording
    # because the annotations object works with samples, not timings
    events_in_time_offset_lfp = events_in_time_offset * lfp_sf
    events_lfp[:,0] = events_in_time_offset_lfp

    ## create an annotation object for the intracranial recording
    annotations_lfp = mne.annotations_from_events(
        events_lfp, sfreq=lfp_sf, event_desc=inv_dic
        )

    lfp_rec_offset.set_annotations(None) # make sure that no annotations are present
    lfp_rec_offset.set_annotations(annotations_lfp) # set the new annotations

    external_title = (
        "SYNCHRONIZED_EXTERNAL_" + str(self.dataset_extra.file_name[:-4]) + "_raw.fif"
        )

    if self.dataset_intra.flag_cleaned == True:
        lfp_title = (
            "SYNCHRONIZED_INTRACRANIAL_CLEANED_" + str(self.dataset_intra.file_name[:-4]) + "_raw.fif"
            )
    else:
        lfp_title = (
            "SYNCHRONIZED_INTRACRANIAL_" + str(self.dataset_intra.file_name[:-4]) + "_raw.fif"
            )

    if self.folder_path is not None:
        fname_external_out=join(self.folder_path, external_title)
        fname_lfp_out =join(self.folder_path, lfp_title)
    else:
        fname_external_out = external_title
        fname_lfp_out = lfp_title

    TMSi_rec_offset_annotations_onset = TMSi_rec_offset.annotations.onset - new_start_external
    lfp_rec_offset_annotations_onset= lfp_rec_offset.annotations.onset - new_start_intracranial
    
    lfp_timescale = np.linspace(
        0, self.dataset_intra.synced_data.get_data().shape[1]/lfp_sf, 
        self.dataset_intra.synced_data.get_data().shape[1]
        )
    
    TMSi_rec_offset.save(fname_external_out, overwrite=True)
    
    data_lfp = lfp_rec_offset.get_data()
    ch_names = lfp_rec_offset.info['ch_names']
    ch_types = lfp_rec_offset.get_channel_types()

    # Create a new Info object with the correct sampling frequency
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=lfp_sf,
        ch_types='eeg'
    )

    # Create a new Raw object using the corrected info
    lfp_rec_offset_fixed = mne.io.RawArray(data_lfp, info)

    # Reattach annotations and any other metadata
    lfp_rec_offset_fixed.set_annotations(lfp_rec_offset.annotations)

    # Save the corrected Raw
    lfp_rec_offset_fixed.save(fname_lfp_out, overwrite=True)


    QMessageBox.information(
        self, "Synchronization",
        "Synchronization done. Both files saved as .fif"
        )

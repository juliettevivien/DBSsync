"""
This module contains functions to compute the timeshift between
intracranial and extracranial recordings, select artifacts for effective
sampling frequency correction, and compute the effective sampling frequency.

It contains the following functions:
- compute_timeshift: Computes the timeshift between intracranial and extracranial recordings.
- select_first_artifact_intra_eff_sf_correction: Selects the first artifact in the
    intracranial recording for effective sampling frequency correction.
- select_last_artifact_intra_eff_sf_correction: Selects the last artifact in the
    intracranial recording for effective sampling frequency correction.
- select_first_artifact_extra_eff_sf_correction: Selects the first artifact in the
    extracranial recording for effective sampling frequency correction.
- select_last_artifact_extra_eff_sf_correction: Selects the last artifact in the
    extracranial recording for effective sampling frequency correction.
- compute_eff_sf: Computes the effective sampling frequency of the intracranial recording
    based on the selected artifacts in both intracranial and extracranial recordings.   
    
"""
import numpy as np
import scipy
from matplotlib.backend_bases import MouseButton


def compute_timeshift(self):
    """ Compute the timeshift between the intracranial and extracranial recordings
    to check if sampling frequencies are correct and if the two recordings
    are synchronized.
    The timeshift is computed as the difference between the last artifact of the
    intracranial recording and the last artifact of the extracranial recording.
    The timeshift is displayed in milliseconds.
    If the timeshift is too large, it may indicate a problem in the recording,
    such as a packet loss in the intracranial recording.
    If the timeshift is <200ms but still not 0, it may indicate that the 
    sampling frequency of the intracranial recording is not correct. It should
    be corrected by computing the effective sampling frequency of the intracranial
    recording based on the first and last artifacts selected by the user.
    """
    timeshift = (
        self.dataset_extra.last_artifact - self.dataset_intra.last_artifact
        )*1000
    self.label_timeshift.setText(f"Timeshift: {timeshift} ms")


def select_first_artifact_intra_eff_sf_correction(self):
    """ Select the first artifact in the intracranial recording to compute the
    effective sampling frequency of the intracranial recording.
    The user is asked to right-click on the plot to select the start of the first
    artifact. The selected position is displayed on the plot as a black "+" symbol.
    The selected position is stored in the dataset_intra object as the first_art_start_time
    and first_art_start_idx attributes. The effective sampling frequency is computed
    as the difference between the last and first artifacts divided by the time interval
    between them.
    """
    # Check if we're already in selection mode of the last and prevent interference
    if hasattr(self, 'cid_intra_last') and self.cid_intra_last is not None:
        self.canvas_intra_sf.mpl_disconnect(self.cid_intra_last)
        #self.cid_intra_last = None

    pos = []
    self.ax_intra_sf.set_title(
        'Right click on the plot to select the start of the first artifact (shown by the black "+")'
    )

    # Create or update the intracranial "+" symbol
    if not hasattr(self, 'plus_symbol_intra_first'):
        self.plus_symbol_intra_first, = self.ax_intra_sf.plot(
            [], [], "k+", markersize=10
            )

    def onclick(event):
        if event.inaxes is not None:  # Check if the click is inside the axes
            if event.button == MouseButton.RIGHT:
                pos.append([event.xdata, event.ydata])

                # Update the position of the black "+" symbol
                closest_index_x = np.argmin(np.abs(
                    self.dataset_intra.times - event.xdata
                    ))
                closest_value_x = self.dataset_intra.times[closest_index_x]
                closest_value_y = self.dataset_intra.raw_data.get_data()[
                    self.dataset_intra.selected_channel_index][closest_index_x]
                self.plus_symbol_intra_first.set_data(
                    [closest_value_x], [closest_value_y]
                    )
                self.canvas_intra_sf.draw()

                self.dataset_intra.first_art_start_time = closest_value_x
                self.dataset_intra.first_art_start_idx = int(
                    np.round(closest_value_x * self.dataset_intra.sf)
                    )
                self.label_time_select_first_intra.setText(
                    f"Selected Artifact start: {np.round(closest_value_x, decimals=3)} s"
                    )
                self.label_sample_select_first_intra.setText(
                    f"Sample n# {int(self.dataset_intra.first_art_start_idx)}"
                    )
                self.update_compute_eff_sf_button_state()

    self.cid_intra_first = self.canvas_intra_sf.mpl_connect(
        "button_press_event", onclick
        )

def select_last_artifact_intra_eff_sf_correction(self):
    """ Select the last artifact in the intracranial recording to compute the
    effective sampling frequency of the intracranial recording.
    The user is asked to right-click on the plot to select the start of the last
    artifact. The selected position is displayed on the plot as a red "+" symbol.
    The selected position is stored in the dataset_intra object as the last_art_start_time
    and last_art_start_idx attributes. The effective sampling frequency is computed
    as the difference between the last and first artifacts divided by the time interval
    between them.
    """
    # Check if we're already in external selection mode and prevent interference
    if hasattr(self, 'cid_intra_first') and self.cid_intra_first is not None:
        self.canvas_intra_sf.mpl_disconnect(self.cid_intra_first)
        #self.cid_intra_first = None

    pos = []

    self.ax_intra_sf.set_title(
        'Right click on the plot to select the start of the last artifact (shown by the red "+")'
    )
    #self.canvas_intra_sf.draw()

    # Create or update the intracranial "+" symbol
    if not hasattr(self, 'plus_symbol_intra_last'):
        self.plus_symbol_intra_last, = self.ax_intra_sf.plot(
            [], [], "r+", markersize=10
            )

    def onclick(event):
        if event.inaxes is not None:  # Check if the click is inside the axes
            if event.button == MouseButton.RIGHT:
                pos.append([event.xdata, event.ydata])

                # Update the position of the black "+" symbol
                closest_index_x = np.argmin(np.abs(
                    self.dataset_intra.times - event.xdata
                    ))
                closest_value_x = self.dataset_intra.times[closest_index_x]
                closest_value_y = self.dataset_intra.raw_data.get_data()[
                    self.dataset_intra.selected_channel_index][closest_index_x]
                self.plus_symbol_intra_last.set_data(
                    [closest_value_x], [closest_value_y]
                    )
                self.canvas_intra_sf.draw()
                self.dataset_intra.last_art_start_time = closest_value_x
                self.dataset_intra.last_art_start_idx = int(
                    np.round(closest_value_x * self.dataset_intra.sf)
                    )
                self.label_time_select_last_intra.setText(
                    f"Selected Artifact start: {np.round(closest_value_x, decimals=3)} s"
                    )
                self.label_sample_select_last_intra.setText(
                    f"Sample n# {int(self.dataset_intra.last_art_start_idx)}"
                    )
                self.update_compute_eff_sf_button_state()

    self.cid_intra_last = self.canvas_intra_sf.mpl_connect("button_press_event", onclick)   



def select_first_artifact_extra_eff_sf_correction(self):
    """ Select the first artifact in the extracranial recording to compute the
    effective sampling frequency of the intracranial recording.
    The user is asked to right-click on the plot to select the start of the first
    artifact. The selected position is displayed on the plot as a black "+" symbol.
    The selected position is stored in the dataset_extra object as the first_art_start_time
    and first_art_start_idx attributes. The effective sampling frequency is computed
    as the difference between the last and first artifacts divided by the time interval
    between them.
    """
    # Check if we're already in selection mode of the last and prevent interference
    if hasattr(self, 'cid_extra_last') and self.cid_extra_last is not None:
        self.canvas_extra_sf.mpl_disconnect(self.cid_extra_last)
    pos = []
    self.ax_extra_sf.set_title(
        'Right click on the plot to select the start of the first artifact (shown by the black "+")'
    )

    # Create or update the extracranial "+" symbol
    if not hasattr(self, 'plus_symbol_extra_first'):
        self.plus_symbol_extra_first, = self.ax_extra_sf.plot(
            [], [], "k+", markersize=10
            )
    b, a = scipy.signal.butter(1, 0.05, "highpass")
    data = scipy.signal.filtfilt(
        b, a, self.dataset_extra.raw_data.get_data()[
            self.dataset_extra.selected_channel_index
            ])

    def onclick(event):
        if event.inaxes is not None:  # Check if the click is inside the axes
            if event.button == MouseButton.RIGHT:
                pos.append([event.xdata, event.ydata])

                # Update the position of the black "+" symbol
                closest_index_x = np.argmin(np.abs(
                    self.dataset_extra.times - event.xdata
                    ))
                closest_value_x = self.dataset_extra.times[closest_index_x]
                closest_value_y = data[closest_index_x]
                self.plus_symbol_extra_first.set_data(
                    [closest_value_x], [closest_value_y]
                    )
                self.canvas_extra_sf.draw()

                self.dataset_extra.first_art_start_time = closest_value_x
                self.dataset_extra.first_art_start_idx = closest_index_x 
                self.label_time_select_first_extra.setText(
                    f"Selected Artifact start: {np.round(closest_value_x, decimals=8)} s"
                    )
                self.label_sample_select_first_extra.setText(
                    f"Sample n# {int(self.dataset_extra.first_art_start_idx)}"
                    )
                self.update_compute_eff_sf_button_state()

    self.cid_extra_first = self.canvas_extra_sf.mpl_connect(
        "button_press_event", onclick
        )


def select_last_artifact_extra_eff_sf_correction(self):
    """ Select the last artifact in the extracranial recording to compute the
    effective sampling frequency of the intracranial recording.
    The user is asked to right-click on the plot to select the start of the last
    artifact. The selected position is displayed on the plot as a red "+" symbol.
    The selected position is stored in the dataset_extra object as the last_art_start_time
    and last_art_start_idx attributes. The effective sampling frequency is computed
    as the difference between the last and first artifacts divided by the time interval
    between them.
    """
    # Check if we're already in external selection mode and prevent interference
    if hasattr(self, 'cid_extra_first') and self.cid_extra_first is not None:
        self.canvas_extra_sf.mpl_disconnect(self.cid_extra_first)
        #self.cid_intra_first = None

    pos = []

    self.ax_extra_sf.set_title(
        'Right click on the plot to select the start of the last artifact (shown by the red "+")'
    )
    #self.canvas_extra_sf.draw()

    # Create or update the intracranial "+" symbol
    if not hasattr(self, 'plus_symbol_extra_last'):
        self.plus_symbol_extra_last, = self.ax_extra_sf.plot(
            [], [], "r+", markersize=10
            )
    b, a = scipy.signal.butter(1, 0.05, "highpass")
    data = scipy.signal.filtfilt(
        b, a, self.dataset_extra.raw_data.get_data()[
            self.dataset_extra.selected_channel_index
            ])
    
    def onclick(event):
        if event.inaxes is not None:  # Check if the click is inside the axes
            if event.button == MouseButton.RIGHT:
                pos.append([event.xdata, event.ydata])

                # Update the position of the black "+" symbol
                closest_index_x = np.argmin(np.abs(
                    self.dataset_extra.times - event.xdata)
                    )
                closest_value_x = self.dataset_extra.times[closest_index_x]
                closest_value_y = data[closest_index_x]
                self.plus_symbol_extra_last.set_data(
                    [closest_value_x], [closest_value_y]
                    )
                self.canvas_extra_sf.draw()
                self.dataset_extra.last_art_start_time = closest_value_x
                self.dataset_extra.last_art_start_idx = closest_index_x
                self.label_time_select_last_extra.setText(
                    f"Selected Artifact start: {np.round(closest_value_x, decimals=8)} s"
                    )
                self.label_sample_select_last_extra.setText(
                    f"Sample n# {int(self.dataset_extra.last_art_start_idx)}"
                    )
                self.update_compute_eff_sf_button_state()

    self.cid_extra_last = self.canvas_extra_sf.mpl_connect(
        "button_press_event", onclick
        )    

def compute_eff_sf(self):
    """ Compute the effective sampling frequency of the intracranial recording
    based on the first and last artifacts selected by the user in the intracranial
    and extracranial recordings.
    The effective sampling frequency is computed as the difference between the
    last and first artifacts of the intracranial recording, divided by the time 
    interval between them computed from the extracranial recording.
    The effective sampling frequency is stored in the dataset_intra object as the
    eff_sf attribute. The timescale of the intracranial recording is updated
    accordingly.
    """
    time_interval = self.dataset_extra.last_art_start_time - self.dataset_extra.first_art_start_time
    sample_interval = self.dataset_intra.last_art_start_idx - self.dataset_intra.first_art_start_idx
    self.dataset_intra.eff_sf = sample_interval/time_interval
    self.dataset_intra.sf = self.dataset_intra.eff_sf
    self.dataset_intra.times = np.linspace(
        0, 
        self.dataset_intra.raw_data.get_data().shape[1]/self.dataset_intra.sf, 
        self.dataset_intra.raw_data.get_data().shape[1]
        )
    self.label_eff_sf.setText(
        f"The effective sampling frequency of the intracranial recording is actually {self.dataset_intra.eff_sf} and will be used for synchronization.")


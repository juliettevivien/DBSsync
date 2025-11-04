import sys
import matplotlib
matplotlib.use('Qt5Agg')
import PyQt5
from PyQt5.QtWidgets import (
    QLabel, QLineEdit, QComboBox, QApplication, QMainWindow, QPushButton, 
    QVBoxLayout, QHBoxLayout, QWidget, QInputDialog, QMessageBox, 
    QStackedWidget, QDialog, QTableWidget, QTableWidgetItem, QCheckBox, QHeaderView)

from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from os.path import join, dirname, exists
import numpy as np
import mne
import webbrowser
from functools import partial
import json

from functions.io import (
    select_saving_folder,
    load_int_file,
    load_ext_file,
    save_datasets,
    # save_datasets_as_set,
    # save_datasets_as_fif,
    # synchronize_datasets_as_pickles,
    # synchronize_datasets_as_one_pickle,
    # synchronize_datasets_as_mat
    )
from functions.find_artifacts import (
    detect_artifacts_intra,
    manual_selection_intra,
    detect_artifacts_external,
    manual_selection_external
)
from functions.interactive import (
    prompt_channel_name_intra,
    select_channel_extra,
    select_ecg_channel_to_compute_hr_external,
    select_last_artifact_intra,
    select_last_artifact_extra,
    choose_int_channel_for_cleaning,
    choose_ext_channel_for_cleaning,
    validate_filtering
    )
from functions.plotting import (
    plot_channel_intra,
    plot_channel_extra,
    plot_synced_channels,
    plot_overlapped_channels_ecg,
    plot_scatter_channel_extra_sf,
    plot_scatter_channel_intra_sf
    )
from functions.timeshift import (
    compute_timeshift, 
    compute_eff_sf,
    select_first_artifact_intra_eff_sf_correction,
    select_last_artifact_intra_eff_sf_correction,
    select_first_artifact_extra_eff_sf_correction,
    select_last_artifact_extra_eff_sf_correction
    )
from functions.ecg_cleaning import (
    find_r_peaks,
    start_ecg_cleaning_interpolation,
    start_ecg_cleaning_template_sub,
    start_ecg_cleaning_svd,
    manual_override
)
from functions.utils import find_similar_sample
from functions.classes import DataSet, Button, Toolbar


class SyncGUI(QMainWindow):
    def __init__(self):
        #######################################################################
        #                        MAIN LAYOUT OF THE GUI                       #
        #######################################################################
        super().__init__()
        self.folder_path = None
        self.r_peak_polarity_lfp = None  # initialize as None to allow for potential user override
        self.start_cleaning_time = None  # initialize as None to allow for potential user override
        self.end_cleaning_time = None  # initialize as None to allow for potential user override

        # Set up the main window
        self.setWindowTitle("DBSsync GUI")
        self.setWindowIcon(QIcon("logo_resized.png"))
        self.setGeometry(100, 100, 1000, 600)

        # Create a stacked widget to hold multiple pages
        self.stacked_widget = QStackedWidget()

        # Create the pages and add them to the stacked widget
        self.home_page = self.create_home_page()
        self.timeshift_page = self.create_timeshift_page()
        self.effective_sf_page = self.create_effective_sf_page()
        self.ecg_cleaning_page = self.create_ecg_cleaning_page()
        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.timeshift_page)
        self.stacked_widget.addWidget(self.effective_sf_page)
        self.stacked_widget.addWidget(self.ecg_cleaning_page)


        # Create the header with navigation buttons
        header_layout = QHBoxLayout()
        self.menu_label = QLabel("MENU")
        self.btn_home = Button("Home Page","#cd9ddc")
        self.btn_home.clicked.connect(self.show_home_page)
        self.btn_timeshift = Button("Timeshift Analysis", "#cd9ddc")
        self.btn_timeshift.clicked.connect(self.show_timeshift_page)
        self.btn_effective_sf = Button("Effective Sampling Frequency correction", "#cd9ddc")
        self.btn_effective_sf.clicked.connect(partial(self.show_effective_sf_page))
        self.btn_ecg_cleaning = Button("ECG Cleaning", "#cd9ddc")
        self.btn_ecg_cleaning.clicked.connect(self.show_ecg_cleaning_page)        
        self.btn_help = Button("Help", "#cd9ddc")
        self.btn_help.clicked.connect(self.show_help)

        # Add buttons to the header layout
        header_layout.addWidget(self.menu_label)
        header_layout.addWidget(self.btn_home)
        header_layout.addWidget(self.btn_timeshift)
        header_layout.addWidget(self.btn_effective_sf)
        header_layout.addWidget(self.btn_ecg_cleaning)
        header_layout.addWidget(self.btn_help)
        header_layout.addStretch()

        # Create a widget for the header layout
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        header_widget.setStyleSheet("background-color: #d1d1d1; border-top: 2px solid #d3d3d3; border-left: 2px solid #d3d3d3;")
        self.menu_label.setStyleSheet("border: none; font-weight: bold;")

        # Main vertical layout for the entire GUI
        main_layout = QVBoxLayout()
        main_layout.addWidget(header_widget)
        main_layout.addWidget(self.stacked_widget)

        # Create the footer with RESET button
        footer_layout = QHBoxLayout()
        footer_layout.addStretch()

        # Add reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_app)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: "#cd9ddc";
                color: black;
                font-size: 18px;
                border-radius: 10px;
                border: 1px solid lightgrey;
                padding: 5px 30px;
            }
            QPushButton:hover {
                background-color: "lightgray";
            }
        """)
        footer_layout.addWidget(self.reset_button)
        footer_layout.addStretch()

        footer_widget = QWidget()
        footer_widget.setLayout(footer_layout)
        footer_widget.setStyleSheet("background-color: #d1d1d1; border-top: 2px solid #d3d3d3; border-left: 2px solid #d3d3d3;")

        main_layout.addWidget(footer_widget)

        # Central widget setup
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.update_button_styles(self.btn_home)  # Highlight the first button

        # Initialize datasets objects
        self.dataset_intra = DataSet()  # Dataset for the intracranial recording (STN recordings from Percept). Should be .mat file
        self.dataset_extra = DataSet()  # Dataset for the extracranial recording (EEG for example) Should be .xdf or .Poly5 file

        json_filename = 'config.json'
        with open(json_filename, 'r') as f:
            self.config =  json.load(f)
        #######################################################################
        #                        LAYOUT OF EACH GUI PAGE                      #
        #######################################################################

        #######################################################################
        ###########               LAYOUT OF HOME PAGE               ###########
        #######################################################################

    def create_home_page(self):
        # Main vertical layout for the first page
        main_layout = QVBoxLayout()

        # Horizontal layout for intracranial and external recordings panels
        panel_layout = QHBoxLayout()

        # Left panel for intracranial file
        self.panel_intra = self.create_panel_intra()
        panel_layout.addLayout(self.panel_intra)

        # Right panel for .xdf file
        self.xdf_panel = self.create_xdf_panel()
        panel_layout.addLayout(self.xdf_panel)

        # Add the horizontal panel layout to the main layout
        main_layout.addLayout(panel_layout)

        saving_folder_layout = QHBoxLayout()

        # Create a button to select the folder where to save the results
        self.btn_select_folder = Button("Select folder to save results", "lightyellow")
        self.btn_select_folder.clicked.connect(partial(select_saving_folder, self))
        saving_folder_layout.addWidget(self.btn_select_folder)

        self.label_saving_folder = QLabel("No saving folder selected")
        self.label_saving_folder.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        saving_folder_layout.addWidget(self.label_saving_folder)

        self.btn_save = Button("Save datasets", "lightyellow")
        self.btn_save.clicked.connect(partial(save_datasets, self))
        saving_folder_layout.addWidget(self.btn_save)

        main_layout.addLayout(saving_folder_layout)

        # # layout for saving from .xdf files
        # saving_xdf_layout = QHBoxLayout()

        # self.label_saving_xdf = QLabel("If the external file was .xdf, save:")
        # saving_xdf_layout.addWidget(self.label_saving_xdf)

        # self.btn_sync_as_set = Button("separately as .SET files", "lightyellow")
        # self.btn_sync_as_set.setEnabled(False)
        # self.btn_sync_as_set.clicked.connect(partial(save_datasets_as_set, self))
        # saving_xdf_layout.addWidget(self.btn_sync_as_set)

        # self.btn_sync_as_fif = Button("separately as .fif files", "lightyellow")
        # self.btn_sync_as_fif.setEnabled(False)
        # self.btn_sync_as_fif.clicked.connect(partial(save_datasets_as_fif, self))
        # saving_xdf_layout.addWidget(self.btn_sync_as_fif)

        # self.btn_sync_as_pickle = Button("separately as .pkl files", "lightyellow")
        # self.btn_sync_as_pickle.setEnabled(False)
        # self.btn_sync_as_pickle.clicked.connect(partial(synchronize_datasets_as_pickles, self))
        # saving_xdf_layout.addWidget(self.btn_sync_as_pickle)

        # self.btn_all_as_pickle = Button("all as one .pkl", "lightyellow")
        # self.btn_all_as_pickle.setEnabled(False)
        # self.btn_all_as_pickle.clicked.connect(partial(synchronize_datasets_as_one_pickle, self))
        # saving_xdf_layout.addWidget(self.btn_all_as_pickle)

        # main_layout.addLayout(saving_xdf_layout)

        # # layout for saving from .Poly5 files
        # saving_poly5_layout = QHBoxLayout()

        # self.label_saving_poly5 = QLabel("If the external file was .Poly5, save:")
        # saving_poly5_layout.addWidget(self.label_saving_poly5)

        # self.btn_sync_as_mat = Button("separately as .mat files", "lightyellow")
        # self.btn_sync_as_mat.setEnabled(False)
        # self.btn_sync_as_mat.clicked.connect(partial(synchronize_datasets_as_mat, self))
        # saving_poly5_layout.addWidget(self.btn_sync_as_mat)

        # self.btn_sync_as_pickle_from_poly5 = Button("separately as .pkl files", "lightyellow")
        # self.btn_sync_as_pickle_from_poly5.setEnabled(False)
        # #self.btn_sync_as_pickle_from_poly5.clicked.connect(self.synchronize_datasets_as_pickles_from_poly5)
        # saving_poly5_layout.addWidget(self.btn_sync_as_pickle_from_poly5)

        # main_layout.addLayout(saving_poly5_layout)        


        # Create the first page widget and set the layout
        first_page_widget = QWidget()
        first_page_widget.setLayout(main_layout)
        return first_page_widget


    def create_panel_intra(self):
        """Create the left panel for intracranial file processing."""
        layout = QVBoxLayout()

        # File selection button for intracranial
        self.btn_load_file_intra = Button("Load intracranial file (supported format: .mat, .fif, .json)", "lightblue")
        self.btn_load_file_intra.clicked.connect(partial(load_int_file, self))
        layout.addWidget(self.btn_load_file_intra)

        # Create a label to display the selected file name
        self.file_label_intra = QLabel("No file selected")
        self.file_label_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        layout.addWidget(self.file_label_intra)

        # Set up canvas for matplotlib for intracranial data
        self.figure_intra, self.ax_intra = plt.subplots()
        self.canvas_intra = FigureCanvas(self.figure_intra)
        self.canvas_intra.setEnabled(False)  # Initially hidden


        # Create a navigation toolbar and add it to the layout
        self.toolbar_intra = NavigationToolbar(self.canvas_intra, self)
        self.toolbar_intra.setEnabled(False) 
        layout.addWidget(self.toolbar_intra)  # Add the toolbar to the layout
        layout.addWidget(self.canvas_intra)    # Add the canvas to the layout

        # Button layout for intracranial channel selection and plotting
        self.channel_layout_intra = QVBoxLayout()
        self.channel_selection_layout_intra = QHBoxLayout()

        # Channel selection button for intracranial file (Initially hidden)
        self.btn_select_channel_intra = Button("Select Channel", "lightblue")
        self.btn_select_channel_intra.setEnabled(False)  # Initially inactive
        self.btn_select_channel_intra.clicked.connect(partial(prompt_channel_name_intra, self))
        self.channel_selection_layout_intra.addWidget(self.btn_select_channel_intra)

        # Create a label to display the selected channel name
        self.channel_label_intra = QLabel("No channel selected")
        self.channel_label_intra.setEnabled(False) # Initially inactive
        self.channel_selection_layout_intra.addWidget(self.channel_label_intra)  
        self.channel_layout_intra.addLayout(self.channel_selection_layout_intra)      

        # Plot channel button for intracranial files (Initially hidden)
        self.btn_plot_channel_intra = Button("Plot Selected Channel", "lightblue")
        self.btn_plot_channel_intra.setEnabled(False)  # Initially inactive
        self.btn_plot_channel_intra.clicked.connect(partial(plot_channel_intra, self))
        self.channel_layout_intra.addWidget(self.btn_plot_channel_intra)


        self.artifact_layout_intra = QHBoxLayout()
        self.automatic_artifact_layout_intra = QVBoxLayout()
        self.manual_artifact_layout_intra = QVBoxLayout()


        # Plot artifact detection button for intracranial (Initially hidden)
        self.btn_artifact_detect_intra = Button("Automatic detection synchronization artifact", "lightblue")
        self.btn_artifact_detect_intra.setEnabled(False)  # Initially hidden
        self.btn_artifact_detect_intra.clicked.connect(partial(detect_artifacts_intra, self))
        self.automatic_artifact_layout_intra.addWidget(self.btn_artifact_detect_intra)
        self.label_automatic_artifact_time_intra = QLabel("No artifact automatically detected")
        self.label_automatic_artifact_time_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.label_automatic_artifact_time_intra.setVisible(False)  # Initially hidden
        self.automatic_artifact_layout_intra.addWidget(self.label_automatic_artifact_time_intra)

        self.btn_manual_select_artifact_intra = Button("Manual detection synchronization artifact", "lightblue") 
        self.btn_manual_select_artifact_intra.setEnabled(False)
        self.btn_manual_select_artifact_intra.clicked.connect(partial(manual_selection_intra, self))
        self.manual_artifact_layout_intra.addWidget(self.btn_manual_select_artifact_intra)
        self.label_manual_artifact_time_intra = QLabel("No artifact manually selected")
        self.label_manual_artifact_time_intra.setVisible(False)
        self.label_manual_artifact_time_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.manual_artifact_layout_intra.addWidget(self.label_manual_artifact_time_intra)        
        
        self.artifact_layout_intra.addLayout(self.automatic_artifact_layout_intra)
        self.artifact_layout_intra.addLayout(self.manual_artifact_layout_intra)

        # confirm synchronization button:
        layout_confirm_sync = QHBoxLayout()
        self.button_confirm_sync = Button("Confirm synchronization", "lightblue")
        self.button_confirm_sync.clicked.connect(self.confirm_sync)
        self.button_confirm_sync.setEnabled(False)
        self.label_sync_confirmed = QLabel("Synchronization not confirmed")
        self.label_sync_confirmed.setVisible(True)
        layout_confirm_sync.addWidget(self.button_confirm_sync)
        layout_confirm_sync.addWidget(self.label_sync_confirmed)

        # Add channel layout to main layout for intracranial
        layout.addLayout(self.channel_layout_intra)
        layout.addLayout(self.artifact_layout_intra)
        layout.addLayout(layout_confirm_sync)


        return layout


    def create_xdf_panel(self):
        """Create the right panel for .xdf/POLY5 file processing."""
        layout = QVBoxLayout()

        # File selection button for .xdf
        self.btn_load_file_xdf = Button("Load external file (supported formats: .xdf, .fif, .poly5)", "lightgreen")
        self.btn_load_file_xdf.clicked.connect(partial(load_ext_file, self))
        layout.addWidget(self.btn_load_file_xdf)       


        # Create a label to display the selected file name
        self.file_label_xdf = QLabel("No file selected")
        self.file_label_xdf.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        layout.addWidget(self.file_label_xdf)

        # Set up canvas for matplotlib for .xdf
        self.figure_xdf, self.ax_xdf = plt.subplots()
        self.canvas_xdf = FigureCanvas(self.figure_xdf)
        self.canvas_xdf.setEnabled(False)

        # Create a navigation toolbar and add it to the layout
        self.toolbar_xdf = NavigationToolbar(self.canvas_xdf, self)
        self.toolbar_xdf.setEnabled(False) 
        layout.addWidget(self.toolbar_xdf)  # Add the toolbar to the layout
        layout.addWidget(self.canvas_xdf)    # Add the canvas to the layout


        # Button layout for .xdf channel selection and plotting
        self.channel_layout_xdf = QVBoxLayout()
        self.channel_selection_layout_xdf = QHBoxLayout()

        # Channel selection button for .xdf (Initially hidden)
        self.btn_select_channel_xdf = Button("Select Channel", "lightgreen")
        self.btn_select_channel_xdf.setEnabled(False)  # Initially inactive
        self.btn_select_channel_xdf.clicked.connect(partial(select_channel_extra, self))
        self.channel_selection_layout_xdf.addWidget(self.btn_select_channel_xdf)

        # Create a label to display the selected channel name
        self.channel_label_xdf = QLabel("No channel selected")
        self.channel_label_xdf.setEnabled(False) # Initially inactive
        self.channel_selection_layout_xdf.addWidget(self.channel_label_xdf)  
        self.channel_layout_xdf.addLayout(self.channel_selection_layout_xdf)  

        # Plot channel button for .xdf (Initially hidden)
        self.btn_plot_channel_xdf = Button("Plot Selected Channel", "lightgreen")
        self.btn_plot_channel_xdf.setEnabled(False)  # Initially hidden
        self.btn_plot_channel_xdf.clicked.connect(partial(plot_channel_extra, self))
        self.channel_layout_xdf.addWidget(self.btn_plot_channel_xdf)

        self.artifact_layout_xdf = QHBoxLayout()
        self.automatic_artifact_layout_xdf = QVBoxLayout()
        self.manual_artifact_layout_xdf = QVBoxLayout()

        # Plot artifact detection button for .xdf (Initially hidden)
        self.btn_artifact_detect_xdf = Button("Automatic detection synchronization artifact", "lightgreen")
        self.btn_artifact_detect_xdf.setEnabled(False)  # Initially hidden
        self.btn_artifact_detect_xdf.clicked.connect(partial(detect_artifacts_external, self))
        self.automatic_artifact_layout_xdf.addWidget(self.btn_artifact_detect_xdf)
        self.label_automatic_artifact_time_xdf = QLabel("No artifact automatically detected")
        self.label_automatic_artifact_time_xdf.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.label_automatic_artifact_time_xdf.setVisible(False)  # Initially hidden
        self.automatic_artifact_layout_xdf.addWidget(self.label_automatic_artifact_time_xdf)

        self.btn_manual_select_artifact_xdf = Button("Manual detection synchronization artifact", "lightgreen")    
        self.btn_manual_select_artifact_xdf.setEnabled(False)
        self.btn_manual_select_artifact_xdf.clicked.connect(partial(manual_selection_external, self))
        self.manual_artifact_layout_xdf.addWidget(self.btn_manual_select_artifact_xdf)
        self.label_manual_artifact_time_xdf = QLabel("No artifact manually selected")
        self.label_manual_artifact_time_xdf.setVisible(False)
        self.label_manual_artifact_time_xdf.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.manual_artifact_layout_xdf.addWidget(self.label_manual_artifact_time_xdf)

        self.artifact_layout_xdf.addLayout(self.automatic_artifact_layout_xdf)
        self.artifact_layout_xdf.addLayout(self.manual_artifact_layout_xdf)

        self.heart_rate_layout = QHBoxLayout()
        self.btn_select_ecg_channel = Button("Select ECG channel to compute heart rate", "lightgreen")
        self.btn_select_ecg_channel.setEnabled(False)  # Initially inactive
        self.btn_select_ecg_channel.clicked.connect(partial(select_ecg_channel_to_compute_hr_external, self))
        self.heart_rate_layout.addWidget(self.btn_select_ecg_channel)

        # Create a label to display the selected channel name
        self.ecg_channel_label = QLabel("No channel selected")
        self.ecg_channel_label.setEnabled(False) # Initially inactive
        self.heart_rate_layout.addWidget(self.ecg_channel_label)  


        # Add channel layout to main layout for .xdf
        layout.addLayout(self.channel_layout_xdf)
        layout.addLayout(self.artifact_layout_xdf)
        layout.addLayout(self.heart_rate_layout)

        return layout



        #######################################################################
        ###########             LAYOUT OF TIMESHIFT PAGE            ###########
        #######################################################################

    def create_timeshift_page(self):
        """Create the timeshift page layout."""
        # Second page layout
        layout_timeshift_page = QVBoxLayout()

        self.btn_plot_synced_channels = Button("Plot synchronized channels", "lightyellow")
        self.btn_plot_synced_channels.clicked.connect(partial(plot_synced_channels, self))
        self.btn_plot_synced_channels.setEnabled(False)
        layout_timeshift_page.addWidget(self.btn_plot_synced_channels)

        # Set up canvas for matplotlib for synced datasets
        self.figure_synced, self.ax_synced = plt.subplots()
        self.canvas_synced = FigureCanvas(self.figure_synced)
        self.canvas_synced.setEnabled(False)  # Initially hidden

        # Set up the interactive toolbar to plot the synchronized signals together and check for timeshift
        self.toolbar_synced = NavigationToolbar(self.canvas_synced, self)
        self.toolbar_synced.setEnabled(False)
        layout_timeshift_page.addWidget(self.toolbar_synced)
        layout_timeshift_page.addWidget(self.canvas_synced)

        layout_timeshift_page_selection = QHBoxLayout()
        layout_timeshift_page_selection_intra = QVBoxLayout()
        layout_timeshift_page_selection_xdf = QVBoxLayout()

        self.btn_select_last_art_intra = Button("Select last artifact in intracranial recording", "lightblue")
        self.btn_select_last_art_intra.clicked.connect(partial(select_last_artifact_intra, self))
        self.btn_select_last_art_intra.setEnabled(False)
        layout_timeshift_page_selection_intra.addWidget(self.btn_select_last_art_intra)

        self.label_select_last_art_intra = QLabel("No artifact selected")
        self.label_select_last_art_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        layout_timeshift_page_selection_intra.addWidget(self.label_select_last_art_intra)


        self.btn_select_last_art_xdf = Button("Select last artifact in extracranial recording", "lightgreen")
        self.btn_select_last_art_xdf.clicked.connect(partial(select_last_artifact_extra, self))
        self.btn_select_last_art_xdf.setEnabled(False)
        layout_timeshift_page_selection_xdf.addWidget(self.btn_select_last_art_xdf)

        self.label_select_last_art_xdf = QLabel("No artifact selected")
        self.label_select_last_art_xdf.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        layout_timeshift_page_selection_xdf.addWidget(self.label_select_last_art_xdf)

        layout_timeshift_page_selection.addLayout(layout_timeshift_page_selection_intra)
        layout_timeshift_page_selection.addLayout(layout_timeshift_page_selection_xdf)
        layout_timeshift_page.addLayout(layout_timeshift_page_selection)

        layout_timeshift = QHBoxLayout()
        self.btn_compute_timeshift = Button("Compute timeshift", "lightyellow")
        self.btn_compute_timeshift.clicked.connect(partial(compute_timeshift, self))
        self.btn_compute_timeshift.setEnabled(False)
        layout_timeshift.addWidget(self.btn_compute_timeshift)

        self.label_timeshift = QLabel("No timeshift computed")
        layout_timeshift.addWidget(self.label_timeshift)
        layout_timeshift_page.addLayout(layout_timeshift)

        # Create the timeshift page widget
        timeshift_page_widget = QWidget()
        timeshift_page_widget.setLayout(layout_timeshift_page)
        
        return timeshift_page_widget


        #######################################################################
        ###########          LAYOUT OF EFFECTIVE SF PAGE            ###########
        #######################################################################


    def create_effective_sf_page(self):
        """Create the effective sampling frequency page layout."""
        # Main vertical layout for the first page
        layout_effective_sf_page = QVBoxLayout()

        # Horizontal layout for intracranial and external recordings panels
        panel_layout = QHBoxLayout()

        # Left panel for intracranial file
        self.panel_intra_sf_correction = self.create_panel_intra_sf_correction()
        panel_layout.addLayout(self.panel_intra_sf_correction)

        # Right panel for .xdf file
        self.panel_extra_sf_correction = self.create_panel_extra_sf_correction()
        panel_layout.addLayout(self.panel_extra_sf_correction)

        # Add the horizontal panel layout to the main layout
        layout_effective_sf_page.addLayout(panel_layout)

        computing_sf_layout = QHBoxLayout()
        # Compute effective sampling frequency button
        self.btn_compute_eff_sf = Button("Compute effective sampling frequency", "lightyellow")
        self.btn_compute_eff_sf.clicked.connect(partial(compute_eff_sf, self))
        self.btn_compute_eff_sf.setEnabled(False)
        computing_sf_layout.addWidget(self.btn_compute_eff_sf)

        self.label_eff_sf = QLabel("No effective sampling frequency computed yet. Select all requested time points first.")
        computing_sf_layout.addWidget(self.label_eff_sf)
        layout_effective_sf_page.addLayout(computing_sf_layout)

        # Create the  page widget and set the layout
        effective_sf_page_widget = QWidget()
        effective_sf_page_widget.setLayout(layout_effective_sf_page)
        
        return effective_sf_page_widget
    

    def create_panel_intra_sf_correction(self):
        """
        Create the left panel for intracranial file processing and sampling 
        frequency correction.
        """
        layout = QVBoxLayout()

        # Set up canvas for matplotlib for intracranial data
        self.figure_intra_sf, self.ax_intra_sf = plt.subplots()
        self.canvas_intra_sf = FigureCanvas(self.figure_intra_sf)
        self.canvas_intra_sf.setEnabled(False)  # Initially hidden

        # Create a navigation toolbar and add it to the layout
        self.toolbar_intra_sf = NavigationToolbar(self.canvas_intra_sf, self)
        self.toolbar_intra_sf.setEnabled(False) 
        layout.addWidget(self.toolbar_intra_sf)  # Add the toolbar to the layout
        layout.addWidget(self.canvas_intra_sf)    # Add the canvas to the layout

        selection_layout = QHBoxLayout()
        selection_layout_first = QVBoxLayout()
        selection_layout_last = QVBoxLayout()

        self.button_select_first_intra = Button("Select first artifact in intracranial recording", "lightblue")
        self.button_select_first_intra.clicked.connect(partial(select_first_artifact_intra_eff_sf_correction, self))    
        self.button_select_first_intra.setEnabled(False)
        self.label_time_select_first_intra = QLabel()
        self.label_time_select_first_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.label_sample_select_first_intra = QLabel()
        self.label_sample_select_first_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)

        self.button_select_last_intra = Button("Select last artifact in intracranial recording", "lightblue")
        self.button_select_last_intra.clicked.connect(partial(select_last_artifact_intra_eff_sf_correction, self))
        self.button_select_last_intra.setEnabled(False)
        self.label_time_select_last_intra = QLabel()
        self.label_time_select_last_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.label_sample_select_last_intra = QLabel()
        self.label_sample_select_last_intra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)

        selection_layout_first.addWidget(self.button_select_first_intra)
        selection_layout_first.addWidget(self.label_time_select_first_intra)
        selection_layout_first.addWidget(self.label_sample_select_first_intra)
        selection_layout.addLayout(selection_layout_first)

        selection_layout_last.addWidget(self.button_select_last_intra)
        selection_layout_last.addWidget(self.label_time_select_last_intra)
        selection_layout_last.addWidget(self.label_sample_select_last_intra)        
        selection_layout.addLayout(selection_layout_last)
        layout.addLayout(selection_layout)

        return layout

    def create_panel_extra_sf_correction(self):
        """
        Create the left panel for intracranial file processing and sampling 
        frequency correction.
        """
        layout = QVBoxLayout()

        # Set up canvas for matplotlib for intracranial data
        self.figure_extra_sf, self.ax_extra_sf = plt.subplots()
        self.canvas_extra_sf = FigureCanvas(self.figure_extra_sf)
        self.canvas_extra_sf.setEnabled(False)  # Initially hidden


        # Create a navigation toolbar and add it to the layout
        self.toolbar_extra_sf = NavigationToolbar(self.canvas_extra_sf, self)
        self.toolbar_extra_sf.setEnabled(False) 
        layout.addWidget(self.toolbar_extra_sf)  # Add the toolbar to the layout
        layout.addWidget(self.canvas_extra_sf)    # Add the canvas to the layout

        selection_layout = QHBoxLayout()
        selection_layout_first = QVBoxLayout()
        selection_layout_last = QVBoxLayout()

        self.button_select_first_extra = Button("Select first artifact in external recording", "lightgreen")
        self.button_select_first_extra.clicked.connect(partial(select_first_artifact_extra_eff_sf_correction, self))    
        self.button_select_first_extra.setEnabled(False)
        self.label_time_select_first_extra = QLabel()
        self.label_time_select_first_extra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.label_sample_select_first_extra = QLabel()
        self.label_sample_select_first_extra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)

        self.button_select_last_extra = Button("Select last artifact in external recording", "lightgreen")
        self.button_select_last_extra.clicked.connect(partial(select_last_artifact_extra_eff_sf_correction, self))
        self.button_select_last_extra.setEnabled(False)
        self.label_time_select_last_extra = QLabel()
        self.label_time_select_last_extra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)
        self.label_sample_select_last_extra = QLabel()
        self.label_sample_select_last_extra.setAlignment(PyQt5.QtCore.Qt.AlignCenter)

        selection_layout_first.addWidget(self.button_select_first_extra)
        selection_layout_first.addWidget(self.label_time_select_first_extra)
        selection_layout_first.addWidget(self.label_sample_select_first_extra)
        selection_layout.addLayout(selection_layout_first)

        selection_layout_last.addWidget(self.button_select_last_extra)
        selection_layout_last.addWidget(self.label_time_select_last_extra)
        selection_layout_last.addWidget(self.label_sample_select_last_extra)        
        selection_layout.addLayout(selection_layout_last)
        layout.addLayout(selection_layout)

        return layout

        #######################################################################
        ###########          LAYOUT OF ECG CLEANING MAIN PAGE       ###########
        #######################################################################
    def create_ecg_cleaning_page(self):
        """Create the ECG cleaning page layout."""
        layout_ecg_cleaning_page = QVBoxLayout()

        layout_channel_selection_and_plot = QHBoxLayout()
        layout_ecg_channels_selection = QVBoxLayout()

        # Choosing the intracranial channel to clean
        layout_int_channel_selection_cleaning = QHBoxLayout()
        self.btn_choose_int_channel_for_cleaning = Button("Choose intracranial channel to clean", "lightyellow")
        self.btn_choose_int_channel_for_cleaning.clicked.connect(partial(choose_int_channel_for_cleaning, self)) 
        self.btn_choose_int_channel_for_cleaning.setEnabled(False) 
        # add a label to show the selected channel name
        self.label_selected_int_channel = QLabel("No channel selected")
        layout_int_channel_selection_cleaning.addWidget(self.btn_choose_int_channel_for_cleaning)
        layout_int_channel_selection_cleaning.addWidget(self.label_selected_int_channel)

        # Choosing the external ECG channel to help cleaning
        layout_ext_channel_selection_cleaning = QHBoxLayout()
        self.btn_choose_ext_channel_for_cleaning = Button("If available, choose external ECG channel for better cleaning", "lightyellow")
        self.btn_choose_ext_channel_for_cleaning.clicked.connect(partial(choose_ext_channel_for_cleaning, self))
        self.btn_choose_ext_channel_for_cleaning.setEnabled(False) # Should be enabled only when the file is loaded
        # add a label to show the selected channel name
        self.label_selected_ext_channel = QLabel("No channel selected")
        layout_ext_channel_selection_cleaning.addWidget(self.btn_choose_ext_channel_for_cleaning)
        layout_ext_channel_selection_cleaning.addWidget(self.label_selected_ext_channel)        

        # Add the filtering option:
        layout_filtering_before_cleaning = QHBoxLayout()
        # add a label to indicate filtering option:
        self.label_filtering_option = QLabel("Low-pass :")
        self.box_filtering_option = QLineEdit()
        self.box_filtering_option.setEnabled(False)  # enable only when files are loaded
        self.btn_validate_filtering = Button("Apply filter", "lightyellow")
        self.btn_validate_filtering.clicked.connect(partial(validate_filtering, self)) # move to interactive script
        self.btn_validate_filtering.setEnabled(False)  # enabled when int channel is loaded

        layout_filtering_before_cleaning.addWidget(self.label_filtering_option)
        layout_filtering_before_cleaning.addWidget(self.box_filtering_option)
        layout_filtering_before_cleaning.addWidget(self.btn_validate_filtering)
        
        # Add the plot button
        self.btn_confirm_and_plot_channels = Button("Plot selected channels", "lightyellow")
        self.btn_confirm_and_plot_channels.clicked.connect(partial(plot_overlapped_channels_ecg, self))
        self.btn_confirm_and_plot_channels.setEnabled(False)

        # Add to the main layout for channel selection:
        layout_ecg_channels_selection.addLayout(layout_int_channel_selection_cleaning)
        layout_ecg_channels_selection.addLayout(layout_ext_channel_selection_cleaning)
        layout_ecg_channels_selection.addLayout(layout_filtering_before_cleaning)
        layout_ecg_channels_selection.addWidget(self.btn_confirm_and_plot_channels)

        # create the layout for the plotting area
        layout_plot_original_signal = QVBoxLayout()
        # add a matplotlib canvas to visualize the raw/filtered data
        self.figure_overlapped, self.ax_overlapped = plt.subplots()
        self.canvas_overlapped = FigureCanvas(self.figure_overlapped)

        # Set up the interactive toolbar to plot the signal
        self.toolbar_overlapped = Toolbar(self.canvas_overlapped, self)
        self.toolbar_overlapped.setEnabled(False)

        layout_plot_original_signal.addWidget(self.toolbar_overlapped)
        layout_plot_original_signal.addWidget(self.canvas_overlapped)

        layout_channel_selection_and_plot.addLayout(layout_ecg_channels_selection)
        layout_channel_selection_and_plot.addLayout(layout_plot_original_signal)

        # Create a horizontal layout for the methods buttons
        layout_methods_global = QHBoxLayout()
        layout_r_peak_detect = QHBoxLayout()

        # Add a button to detect R-peaks in the selected channel
        self.btn_detect_r_peaks = Button("Detect R-peaks in selected channel", "lightyellow")
        self.btn_detect_r_peaks.clicked.connect(partial(find_r_peaks, self))
        self.btn_detect_r_peaks.setEnabled(False)  # Should be enabled only when the channel is selected
        layout_r_peak_detect.addWidget(self.btn_detect_r_peaks)

        # Add a QComboBox to select the threshold for R-peak detection, with a 
        # default value of 95:
        self.combo_r_peak_threshold = QComboBox()
        self.combo_r_peak_threshold.addItems(["95", "96", "97", "98", "99"])
        self.combo_r_peak_threshold.setCurrentText("95")  # Set default value
        self.combo_r_peak_threshold.setEnabled(False)  # Should be enabled only when the channel is selected
        layout_r_peak_detect.addWidget(self.combo_r_peak_threshold)

        # Add a button for manual override of parameters if needed
        self.btn_manual_override = Button("Manual Override", "lightyellow")
        self.btn_manual_override.clicked.connect(partial(manual_override, self))
        self.btn_manual_override.setEnabled(False)  # Should be enabled only when the channel is selected
        layout_r_peak_detect.addWidget(self.btn_manual_override)

        layout_methods = QHBoxLayout()

        # Insert button for the interpolation method
        self.btn_start_ecg_cleaning_interpolation = Button("Interpolation method", "lightyellow")
        self.btn_start_ecg_cleaning_interpolation.clicked.connect(partial(start_ecg_cleaning_interpolation, self))
        self.btn_start_ecg_cleaning_interpolation.setEnabled(False) # Should be enabled only when the file is loaded

        # Insert button for the template substraction method
        self.btn_start_ecg_cleaning_template_sub = Button("Template substraction method", "lightyellow")
        self.btn_start_ecg_cleaning_template_sub.clicked.connect(partial(start_ecg_cleaning_template_sub, self))
        self.btn_start_ecg_cleaning_template_sub.setEnabled(False)

        # Insert button for the SVD method
        self.btn_start_ecg_cleaning_svd = Button("Singular Value Decomposition method", "lightyellow")
        self.btn_start_ecg_cleaning_svd.clicked.connect(partial(start_ecg_cleaning_svd, self))
        self.btn_start_ecg_cleaning_svd.setEnabled(False)

        layout_methods.addWidget(self.btn_start_ecg_cleaning_interpolation)
        layout_methods.addWidget(self.btn_start_ecg_cleaning_template_sub)
        layout_methods.addWidget(self.btn_start_ecg_cleaning_svd)

        layout_methods_global.addLayout(layout_r_peak_detect)
        layout_methods_global.addLayout(layout_methods)

        # Layout for the first row of plots
        layout_plots1 = QHBoxLayout()
        layout_plots1_left = QVBoxLayout()
        layout_plots1_right = QVBoxLayout()

        # Add a canvas to visualize the detected peaks
        self.figure_detected_peaks, self.ax_detected_peaks = plt.subplots()
        self.canvas_detected_peaks = FigureCanvas(self.figure_detected_peaks)

        # Add a toolbar to visualize the detected peaks
        self.toolbar_detected_peaks = Toolbar(self.canvas_detected_peaks, self)
        self.toolbar_detected_peaks.setEnabled(False)

        layout_plots1_left.addWidget(self.toolbar_detected_peaks)
        layout_plots1_left.addWidget(self.canvas_detected_peaks)  

        # Add a canvas to visualize the raw and cleaned signals overlapped
        # add another matplotlib canvas to visualize the cleaned data
        self.figure_ecg_clean, self.ax_ecg_clean = plt.subplots()
        self.canvas_ecg_clean = FigureCanvas(self.figure_ecg_clean)

        # Set up the interactive toolbar to plot the cleaned signal
        self.toolbar_ecg_clean = Toolbar(self.canvas_ecg_clean, self)
        self.toolbar_ecg_clean.setEnabled(False)

        layout_plots1_right.addWidget(self.toolbar_ecg_clean)
        layout_plots1_right.addWidget(self.canvas_ecg_clean)

        layout_plots1.addLayout(layout_plots1_left)
        layout_plots1.addLayout(layout_plots1_right)

        # Layout for the second row of plots
        layout_plots2 = QHBoxLayout()
        layout_plots2_left = QVBoxLayout()
        layout_plots2_middle = QVBoxLayout()
        layout_plots2_right = QVBoxLayout()

        # add a matplotlib canvas to visualize the ECG artifact detected
        self.figure_ecg_artifact, self.ax_ecg_artifact = plt.subplots()
        self.canvas_ecg_artifact = FigureCanvas(self.figure_ecg_artifact)
        self.toolbar_ecg_artifact = Toolbar(self.canvas_ecg_artifact, self)
        self.toolbar_ecg_artifact.setEnabled(False)
        layout_plots2_left.addWidget(self.toolbar_ecg_artifact)
        layout_plots2_left.addWidget(self.canvas_ecg_artifact)

        # add a matplotlib canvas to visualize the power spectrum of the raw and cleaned channel
        self.figure_psd, self.ax_psd = plt.subplots()
        self.canvas_psd = FigureCanvas(self.figure_psd)
        self.toolbar_psd = Toolbar(self.canvas_psd, self)
        self.toolbar_psd.setEnabled(False)
        layout_plots2_middle.addWidget(self.toolbar_psd)
        layout_plots2_middle.addWidget(self.canvas_psd)

        # add a summary of the heart rate found, and a button to confirm the cleaning
        # Display the computed heart rate:
        self.label_heart_rate_lfp = QLabel('Heart Rate:')

        # Create a button to confirm the cleaning and continue with the synchronization
        self.btn_confirm_cleaning = Button("Confirm cleaning and keep channel", "lightyellow")
        self.btn_confirm_cleaning.clicked.connect(self.confirm_cleaning)
        self.btn_confirm_cleaning.setEnabled(False) # Should be enabled only when self.dataset_intra.ecg is not None

        layout_plots2_right.addWidget(self.label_heart_rate_lfp)
        layout_plots2_right.addWidget(self.btn_confirm_cleaning)

        layout_plots2.addLayout(layout_plots2_left)
        layout_plots2.addLayout(layout_plots2_middle)
        layout_plots2.addLayout(layout_plots2_right)

        layout_ecg_cleaning_page.addLayout(layout_channel_selection_and_plot)
        layout_ecg_cleaning_page.addLayout(layout_methods_global)
        layout_ecg_cleaning_page.addLayout(layout_plots1)
        layout_ecg_cleaning_page.addLayout(layout_plots2)

        # Create the first page widget and set the layout
        ecg_cleaning_page_widget = QWidget()
        ecg_cleaning_page_widget.setLayout(layout_ecg_cleaning_page)

        return ecg_cleaning_page_widget



        #######################################################################
        #                     ORDERING AND CALLING GUI PAGES                  #
        #######################################################################

    def show_home_page(self):
        self.stacked_widget.setCurrentIndex(0)
        self.update_button_styles(self.btn_home)

    def show_timeshift_page(self):
        self.stacked_widget.setCurrentIndex(1)
        self.update_button_styles(self.btn_timeshift)
        self.update_plot_sync_channels_state()

    def show_effective_sf_page(self):
        self.stacked_widget.setCurrentIndex(2)
        self.update_button_styles(self.btn_effective_sf)
        if self.dataset_intra.selected_channel_index is not None:
            plot_scatter_channel_intra_sf(self)
            self.button_select_first_intra.setEnabled(True)
            self.button_select_last_intra.setEnabled(True)
        if self.dataset_extra.selected_channel_index is not None:
            plot_scatter_channel_extra_sf(self)
            self.button_select_first_extra.setEnabled(True)
            self.button_select_last_extra.setEnabled(True)        

    def show_ecg_cleaning_page(self):
        self.stacked_widget.setCurrentIndex(3)
        self.update_button_styles(self.btn_ecg_cleaning)

    def show_help(self):
        # Path to the HTML file stored in the GUI folder
        help_file_folder = join(dirname(__file__), "help")
        help_file_path = join(help_file_folder, 'info.html')
        if exists(help_file_path):
            webbrowser.open(f'file://{help_file_path}')
        else:
            print("Help file not found.")


        #######################################################################
        #                           CHOOSING FUNCTIONS                        #
        #######################################################################

    def choose_channel_for_cleaning(self):
        """Prompt for channel name selection for intracranial file."""
        if self.dataset_intra.raw_data:
            try:
                channel_names = self.dataset_intra.ch_names  # List of channel names
                channel_name, ok = QInputDialog.getItem(self, "Channel Selection", "Select a channel:", channel_names, 0, False)

                if ok and channel_name:  # Check if a channel was selected
                    self.dataset_intra.selected_channel_name_ecg = channel_name
                    self.dataset_intra.selected_channel_index_ecg = channel_names.index(channel_name)  # Get the index of the selected channel
                    self.label_selected_channel.setText(f"Selected Channel: {channel_name}")     
                    self.btn_confirm_and_plot_channels.setEnabled(True)            

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to select channel: {e}")

        # Plot the selected channel data from the intracranial file.
        if self.dataset_intra.raw_data and self.dataset_intra.selected_channel_index_ecg is not None:
            self.canvas_ecg.setEnabled(True)
            self.toolbar_ecg.setEnabled(True)
            self.ax_ecg.clear()
            channel_data = self.dataset_intra.raw_data.get_data()[self.dataset_intra.selected_channel_index_ecg]
            times = self.dataset_intra.times
            self.ax_ecg.plot(times, channel_data)
            self.ax_ecg.set_title(f"Channel {self.dataset_intra.selected_channel_index_ecg} data - {self.dataset_intra.selected_channel_name_ecg}")
            self.ax_ecg.set_xlabel("Time (s)")
            self.ax_ecg.set_ylabel("Amplitude")
            self.canvas_ecg.draw()
            self.box_start_time.setEnabled(True)
            self.box_end_time.setEnabled(True)
            self.box_thresh_ecg.setEnabled(True)
            self.btn_validate_start_end_time.setEnabled(True)
            self.box_filtering_option.setEnabled(True)


        #######################################################################
        #                          VALIDATION FUNCTIONS                       #
        #######################################################################

    def confirm_cleaning(self):
        """Confirm the cleaning and replace the raw data with the cleaned data."""
        self.dataset_intra.flag_cleaned = True
        if self.dataset_intra.selected_channel_index_ecg == 0:
            # Replace the corresponding channel's data with the cleaned data
            if self.config['NoSync'] == True:
                self.dataset_intra.raw_data._data[0,:] = self.dataset_intra.cleaned_ecg_left
                QMessageBox.information(self, "ECG cleaning", "Cleaning confirmed. Replacing raw data with cleaned data in the left channel.")
            else:
                self.dataset_intra.synced_data._data[0,:] = self.dataset_intra.cleaned_ecg_left
                QMessageBox.information(self, "ECG cleaning", "Cleaning confirmed. Replacing synced data with cleaned data in the left channel.")


        elif self.dataset_intra.selected_channel_index_ecg == 1:
            # Replace the corresponding channel's data
            if self.config['NoSync'] == True:
                self.dataset_intra.raw_data._data[1,:] = self.dataset_intra.cleaned_ecg_right
                QMessageBox.information(self, "ECG cleaning", "Cleaning confirmed. Replacing raw data with cleaned data in the right channel.")
            else:
                self.dataset_intra.synced_data._data[1,:] = self.dataset_intra.cleaned_ecg_right
                QMessageBox.information(self, "ECG cleaning", "Cleaning confirmed. Replacing synced data with cleaned data in the right channel.")

    # def confirm_cleaning_with_ext(self):
    #     if self.dataset_intra.selected_channel_index_ecg == 0:
    #         # Replace the corresponding channel's data with the cleaned data
    #         self.dataset_intra.synced_data._data[0,:] = self.dataset_intra.cleaned_ecg_left
    #         QMessageBox.information(self, "ECG cleaning", "Cleaning confirmed. Replacing raw data with cleaned data in the left channel.")

    #     elif self.dataset_intra.selected_channel_index_ecg == 1:   
    #         # Replace the corresponding channel's data
    #         self.dataset_intra.synced_data._data[1,:] = self.dataset_intra.cleaned_ecg_right
    #         QMessageBox.information(self, "ECG cleaning", "Cleaning confirmed. Replacing raw data with cleaned data in the right channel.")



    # def validate_start_end_time(self):
    #     """Validate the start and end times for cleaning."""
    #     try:
    #         start_time = int(self.box_start_time.text())
    #         end_time = int(self.box_end_time.text())

    #         if start_time < 0 or end_time < 0:
    #             raise ValueError("Start and end times must be positive.")

    #         if start_time >= end_time:
    #             raise ValueError("Start time must be less than end time.")

    #         if self.dataset_intra.selected_channel_index_ecg == 0:
    #             self.dataset_intra.start_time_left = start_time
    #             self.dataset_intra.end_time_left = end_time
    #         elif self.dataset_intra.selected_channel_index_ecg == 1:
    #             self.dataset_intra.start_time_right = start_time
    #             self.dataset_intra.end_time_right = end_time

    #         self.dataset_intra.ecg_thresh = int(self.box_thresh_ecg.currentText())
    #         self.label_start_time.setText(f"Start time: {start_time} s")
    #         self.label_end_time.setText(f"End time: {end_time} s")
    #         self.btn_start_ecg_cleaning.setEnabled(True)  # Enable the button after validation
    #         self.btn_start_ecg_cleaning_template_sub.setEnabled(True)

    #         if self.radio_button_down.isChecked():
    #             self.dataset_intra.artifact_polarity = "down"
    #         elif self.radio_button_up.isChecked():
    #             self.dataset_intra.artifact_polarity = "up"

    #         if self.box_filtering_option.text() != "":
    #             h_freq = float(self.box_filtering_option.text())
    #             self.dataset_intra.raw_data = self.dataset_intra.raw_data.copy().filter(l_freq=0, h_freq=h_freq, picks=[self.dataset_intra.raw_data.ch_names[0], self.dataset_intra.raw_data.ch_names[1]])

    #     except ValueError as e:
    #         QMessageBox.warning(self, "Invalid Input, please enter an integer", str(e))


    # def validate_filtering_with_ext(self):
    #     """Validate the start and end times for cleaning."""
    #     try:    
    #         if self.box_filtering_option_with_ext.text() != "":
    #             # 1. Extract the original channel before you overwrite it
    #             raw_new_left = self.dataset_intra.synced_data.copy().pick_channels([self.dataset_intra.synced_data.ch_names[0]])
    #             raw_new_left.rename_channels({raw_new_left.ch_names[0]: 'RAW_Left_STN'})
    #             raw_new_right = self.dataset_intra.synced_data.copy().pick_channels([self.dataset_intra.synced_data.ch_names[1]])
    #             raw_new_right.rename_channels({raw_new_right.ch_names[0]: 'RAW_Right_STN'})

    #             # Add channels to existing Raw object
    #             self.dataset_intra.synced_data.add_channels([raw_new_left, raw_new_right])
    #             h_freq = float(self.box_filtering_option_with_ext.text())
    #             self.dataset_intra.synced_data.filter(l_freq=0, h_freq=h_freq, picks=[self.dataset_intra.synced_data.ch_names[0], self.dataset_intra.synced_data.ch_names[1]])
    #             self.label_filter_display_with_ext.setText(f"Low-pass filter applied at {h_freq} Hz to both left and right STN channels")

    #     except ValueError as e:
    #         QMessageBox.warning(self, "Invalid Input, please enter an integer", str(e))

    def confirm_sync(self):
        print('sync confirmed')
        ## offset intracranial recording (crop everything that is more than 1s before the artifact)
        tmax_lfp = max(self.dataset_intra.raw_data.times)
        new_start_intracranial = self.dataset_intra.art_start - 1
        lfp_rec_offset = self.dataset_intra.raw_data.copy().crop(tmin=new_start_intracranial, tmax=tmax_lfp)
        # store the synced intracranial channels in a new object
        self.dataset_intra.synced_data = lfp_rec_offset

        ## offset external recording (crop everything that is more than 1s before the artifact)
        tmax_external = max(self.dataset_extra.times)
        new_start_external = self.dataset_extra.art_start - 1
        TMSi_rec_offset = self.dataset_extra.raw_data.copy().crop(tmin=new_start_external, tmax=tmax_external)
        # store the synced external channels in a new object
        self.dataset_extra.synced_data = TMSi_rec_offset

        # Handle cases where xdf files contain non-continuous data streams:
        if self.dataset_extra.file_name.endswith(".xdf"):
            # extract the timestamp from LSL global timescale, corresponding to the art_start - 1
            # find the index of new_start_external in self.dataset_extra.times:
            self.dataset_extra.art_start_LSL_idx = np.where(self.dataset_extra.times == new_start_external) 
            print(self.dataset_extra.art_start_LSL_idx)
            
        self.label_sync_confirmed.setText("Confirmed, you can now save directly, or clean ECG artifacts")
        self.btn_choose_int_channel_for_cleaning.setEnabled(True)
        self.btn_choose_ext_channel_for_cleaning.setEnabled(True)

        # if self.dataset_extra.file_name.endswith(".xdf"):
        #     self.btn_sync_as_set.setEnabled(True)
        #     self.btn_sync_as_fif.setEnabled(True)
        #     self.btn_sync_as_pickle.setEnabled(True)
        #     self.btn_all_as_pickle.setEnabled(True)
        # elif self.dataset_extra.file_name.endswith(".Poly5"):
        #     self.btn_sync_as_mat.setEnabled(True)


        #######################################################################
        #                       STATE UPDATE FUNCTIONS                        #
        #######################################################################

    def reset_app(self):
        # Close current instance of the window
        self.close()
        
        # Open a new instance of SyncGUI
        self.new_window = SyncGUI()  # Create a new window instance
        self.new_window.show()  # Show the new window

    
    def update_compute_eff_sf_button_state(self):
        if (
            self.dataset_intra.first_art_start_idx is not None
            and self.dataset_intra.last_art_start_idx is not None
            and self.dataset_extra.first_art_start_time is not None
            and self.dataset_extra.last_art_start_time is not None
        ):
            self.btn_compute_eff_sf.setEnabled(True)
        else:
            self.btn_compute_eff_sf.setEnabled(False)


    def update_timeshift_button_state(self):
        """Enable or disable the timeshift button based on artifact selection."""
        if self.dataset_intra.last_artifact is not None and self.dataset_extra.last_artifact is not None:
            self.btn_compute_timeshift.setEnabled(True)
        else:
            self.btn_compute_timeshift.setEnabled(False)

    def update_plot_sync_channels_state(self):
        if self.dataset_intra.art_start is not None and self.dataset_extra.art_start is not None:
            self.btn_plot_synced_channels.setEnabled(True)
        else:
            self.btn_plot_synced_channels.setEnabled(False)

    def update_button_styles(self, active_button):
        # Set custom property to determine which button is active
        buttons = [self.btn_home, self.btn_effective_sf, self.btn_timeshift, self.btn_ecg_cleaning]
        for button in buttons:
            button.setProperty("active", button == active_button)
            button.style().unpolish(button)
            button.style().polish(button)

    def update_synchronize_button_state(self):
        """Enable or disable the synchronize button based on file selection."""
        if self.dataset_intra.art_start is not None and self.dataset_extra.art_start is not None:
            self.button_confirm_sync.setEnabled(True)
        else:
            self.button_confirm_sync.setEnabled(False)
            # self.btn_sync_as_set.setEnabled(False)
            # self.btn_sync_as_fif.setEnabled(False)
            #self.btn_sync_as_pickle.setEnabled(False)
            #self.btn_all_as_pickle.setEnabled(False)
            #self.btn_sync_as_mat.setEnabled(False)



        #######################################################################
        #                   SECONDARY WINDOW FUNCTIONS                        #
        #######################################################################

    def process_value_from_plot(self, value):
        k = int(value)

        # Reconstruct the data using a the number of components decided by the user
        self.X_ecg = (self.U[:, :k] @ np.diag(self.S[:k]) @ self.Vh[:k, :])

        # Plot the reconstructed data and average artifact:
        self.ax_ecg_artifact.clear()
        reconstructed_epochs = []
        times = np.linspace(self.window[0], self.window[1], self.epoch_length)
        for r in range(self.X_ecg.shape[1]):
            self.ax_ecg_artifact.plot(times, self.X_ecg[:, r], color= 'lightgrey')
            reconstructed_epochs.append(self.X_ecg[:, r])

        reconstructed_epochs = np.array(reconstructed_epochs)
        mean_reconstructed_artifact = np.mean(reconstructed_epochs, axis=0)
        self.ax_ecg_artifact.plot(times, mean_reconstructed_artifact, label=f'Mean Reconstructed Artifact using {k} components', color='black', linewidth=2)
        self.ax_ecg_artifact.set_xlabel("Time (seconds)")
        self.ax_ecg_artifact.set_ylabel("Amplitude")
        self.ax_ecg_artifact.legend()
        self.canvas_ecg_artifact.draw()

        clean_data = np.copy(self.full_data)
        #artifact_data = np.copy(self.full_data)
        pre_samples = int(abs(self.window[0]) * self.dataset_intra.sf)
        post_samples = int(self.window[1] * self.dataset_intra.sf)

        for i, peak in enumerate(self.final_peaks):
            raw_epoch_1 = self.full_data[(peak - pre_samples):(peak + post_samples)]
            reconstructed_i = reconstructed_epochs[i]
            # 1. Prepare design matrix for linear fit (scale + offset)
            X_template = np.vstack([reconstructed_i , np.ones_like(reconstructed_i )]).T  # Shape: (template_len, 2)
            # 2. Solve for optimal scale (a) and offset (b) using least squares
            coeffs, _, _, _ = np.linalg.lstsq(X_template, raw_epoch_1, rcond=None)
            a, b = coeffs

            # 3. Build fitted template and subtract
            fitted_template = a * reconstructed_i + b
            complex_qrs_template, start_idx, end_idx = find_similar_sample(fitted_template, tails=30)
            start = (peak - pre_samples) + start_idx
            end = (peak - pre_samples) + end_idx
            raw_epoch_2 = self.full_data[start:end]

            assert len(raw_epoch_2) == len(complex_qrs_template), "Raw epoch length does not match complex QRS template length"

            clean_data[start:end] -= complex_qrs_template
            #artifact_data[start:end] = complex_qrs_template

        clean_data_full = clean_data
        #artifact_data_full = artifact_data

        if self.dataset_intra.selected_channel_index_ecg == 0:
            self.dataset_intra.cleaned_ecg_left = clean_data_full
            print("Left channel cleaned")

        elif self.dataset_intra.selected_channel_index_ecg == 1:
            self.dataset_intra.cleaned_ecg_right = clean_data_full
            print("Right channel cleaned")

        try :
            if self.config['NoSync'] == True:
                timescale = np.linspace(0, self.dataset_intra.raw_data.get_data().shape[1]/self.dataset_intra.sf, self.dataset_intra.raw_data.get_data().shape[1])
            else:
                timescale = np.linspace(0, self.dataset_intra.synced_data.get_data().shape[1]/self.dataset_intra.sf, self.dataset_intra.synced_data.get_data().shape[1])
            # plot an overlap of the raw and cleaned data
            self.canvas_ecg_clean.setEnabled(True)
            self.toolbar_ecg_clean.setEnabled(True)
            self.ax_ecg_clean.clear()
            self.ax_ecg_clean.set_title("Cleaned ECG Signal")
            self.ax_ecg_clean.plot(timescale, self.full_data, label='Raw data')
            self.ax_ecg_clean.plot(timescale, clean_data_full, label='Cleaned data')
            #self.ax_ecg_clean.plot(timescale, artifact_data_full, label='Artifact data', color='red', alpha=0.5)
            self.ax_ecg_clean.set_xlabel("Time (s)")
            self.ax_ecg_clean.set_ylabel("Amplitude")
            self.ax_ecg_clean.legend()
            self.canvas_ecg_clean.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clean ECG: {e}")

        try :
            # Plot an overlap of the power spectrum using welch's method:
            n_fft = int(round(self.dataset_intra.sf))
            n_overlap=int(round(self.dataset_intra.sf)/2)

            psd_raw, freqs_raw = mne.time_frequency.psd_array_welch(
                self.full_data,self.dataset_intra.sf,fmin=0,
                fmax=125,n_fft=n_fft,
                n_overlap=n_overlap)
            
            psd_clean, freqs_clean = mne.time_frequency.psd_array_welch(
                clean_data_full,self.dataset_intra.sf,fmin=0,
                fmax=125,n_fft=n_fft,
                n_overlap=n_overlap)

            self.canvas_psd.setEnabled(True)
            self.toolbar_psd.setEnabled(True)
            self.ax_psd.clear()
            self.ax_psd.plot(freqs_raw, np.log(psd_raw), color='blue', label='PSD raw channel')
            self.ax_psd.plot(freqs_clean, np.log(psd_clean), color = 'orange', label='PSD cleaned channel')
            self.ax_psd.legend()
            self.canvas_psd.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to clean ECG: {e}")

        self.btn_confirm_cleaning.setEnabled(True)  # Enable the button after cleaning



    def show_stream_selection_dialog(self, streamings_df):
        class StreamSelectionDialog(QDialog):
            def __init__(self, df):
                super().__init__()
                self.setWindowTitle("Select Streams to Load")
                self.df = df
                self.selected_streams = []
                self.resize(800, 400)
                
                layout = QVBoxLayout()
                self.setLayout(layout)
                
                # Table
                self.table = QTableWidget()
                self.table.setColumnCount(len(df.columns) + 1)  # Extra column for checkbox
                self.table.setHorizontalHeaderLabels(['Select'] + list(df.columns))
                self.table.setRowCount(len(df))
                
                for i, row in df.iterrows():
                    # Checkbox
                    checkbox = QCheckBox()
                    checkbox_widget = QWidget()
                    h_layout = QHBoxLayout()
                    h_layout.addWidget(checkbox)
                    #h_layout.setAlignment(checkbox, 1)  # center
                    #h_layout.setAlignment(Qt.AlignCenter)
                    h_layout.setContentsMargins(0,0,0,0)
                    checkbox_widget.setLayout(h_layout)
                    self.table.setCellWidget(i, 0, checkbox_widget)
                    
                    # Fill rest of the columns
                    for j, col in enumerate(df.columns):
                        item = QTableWidgetItem(str(row[col]))
                        item.setFlags(item.flags() & ~2)  # make read-only
                        self.table.setItem(i, j+1, item)
                
                self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
                layout.addWidget(self.table)
                
                # OK button
                ok_btn = QPushButton("Load Selected Streams")
                ok_btn.clicked.connect(self.accept)
                layout.addWidget(ok_btn)

            def get_selected_streams(self):
                selected = []
                for i in range(self.table.rowCount()):
                    checkbox_widget = self.table.cellWidget(i, 0)
                    checkbox = checkbox_widget.layout().itemAt(0).widget()
                    if checkbox.isChecked():
                        selected.append(self.df.iloc[i]['Streaming id'])
                return selected

        dialog = StreamSelectionDialog(streamings_df)
        if dialog.exec_() == QDialog.Accepted:
            selected_streams = dialog.get_selected_streams()
            print("User selected streams:", selected_streams)
            return selected_streams
        return []

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SyncGUI()
    window.show()
    sys.exit(app.exec_())

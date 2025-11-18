"""
This module contains the classes used in the main application:
- DataSet: A class to hold the data and metadata for the application.
- Button: A custom QPushButton with a specific style.
- Toolbar: A custom NavigationToolbar for matplotlib with a specific style.
- PlotWindow: A QWidget that displays SVD components for ECG cleaning and allows user input for 'k'.

"""

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QPushButton, QMessageBox



class DataSet:
    def __init__(self, raw_data=None):
        self.raw_data = raw_data
        self.file_path = None
        self.file_name = None
        self.selected_channel_index = None
        self.selected_channel_name = None
        self.ch_names = None
        self.sf = None  
        self.art_start= None 
        self.times = None
        self.last_artifact = None
        self.reset_timescale = None
        self.reset_data = None
        self.max_y_value = None
        self.first_art_start_time = None
        self.last_art_start_time = None
        self.first_art_start_idx = None
        self.last_art_start_idx = None
        self.eff_sf = None
        self.selected_channel_index_ecg = None
        self.selected_channel_name_ecg = None
        self.flag_cleaned = None
        self.selected_stream = None
        self.list_of_streams = None
        self.selected_json_stream_name = None


class Button(QPushButton):
    def __init__(self, text, color, parent=None):
        super().__init__(text, parent)
        
        # Apply a common style (rounded corners and light grey border)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};  /* Color provided dynamically */
                color: black;
                font-size: 14px;
                border-radius: 10px;  /* Rounded corners */
                border: 1px solid lightgrey;  /* Light grey border */
                padding: 5px 10px;  /* Padding for a nicer look */
            }}
            QPushButton[active="true"] {{
                font-weight: bold;  /* Bold font for active button */
            }}
            QPushButton:hover {{
                background-color: lightgray;  /* Hover effect */
            }}
            QPushButton:disabled {{
                background-color: lightgray; 
                color: gray;}}
        """)
        
class Toolbar(NavigationToolbar):
    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        # Apply smaller styling
        self.setStyleSheet("""
            QToolBar {
                spacing: 2px;
            }
            QToolButton {
                width: 16px;
                height: 16px;
                icon-size: 12px 12px;
                padding: 0px;
                margin: 0px;
            }
        """)


class PlotWindow(QWidget):
    """A QWidget that displays SVD components and allows user input for 'k'.
    This window is used to visualize the first 4 SVD components and their
    respective variance explained, and to allow the user to input a value
    for 'k', which is the number of components to keep for the SVD
    reconstruction of the ECG artifact.
    """
    def __init__(self, callback, U, S, window, epoch_length):
        super().__init__()
        self.setWindowTitle("SVD Components")
        self.callback = callback

        # Save inputs
        self.U = U
        self.window = window
        self.epoch_length = epoch_length
        self.S = S

        # === Create the matplotlib figure and canvas === #
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.plot_svd_components()

        # === User input for 'k' === #
        self.k_input = QLineEdit()
        self.k_input.setPlaceholderText("Enter k")

        self.submit_btn = QPushButton("Submit")
        self.submit_btn.clicked.connect(self.submit_k)

        # === Layout === #
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.k_input)
        layout.addWidget(self.submit_btn)
        self.setLayout(layout)

    def plot_svd_components(self):
        """
        Plot the first 4 SVD components, with their respective variance explained.
        """
        n_components = 4
        t = np.linspace(self.window[0], self.window[1], self.epoch_length)
        self.figure.clear()
        variance_explained = (self.S**2) / np.sum(self.S**2) * 100
        for i in range(n_components):
            ax = self.figure.add_subplot(n_components, 1, i + 1)
            ax.plot(t, self.U[:, i], label=f'SVD Component {i+1}, explains {variance_explained[i]:.2f}% of variance')
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right")
            if i == 0:
                ax.set_title("First 4 SVD Temporal Components")
        ax.set_xlabel("Time (seconds)")
        self.canvas.draw()

    def submit_k(self):
        """Handle the submission of 'k' value, i.e., the number of components to
        keep for the SVD reconstruction of the ECG artifact.
        Send the value of k back to the main window.
        """
        try:
            k = int(self.k_input.text())
            self.callback(k)  # Send back to main window
            self.close()
        except ValueError:
             QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer for k.")



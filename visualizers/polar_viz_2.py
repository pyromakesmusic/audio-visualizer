import sys
import matplotlib
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

"""
Functions
"""

#  Interpolation function
def interpolate_list(original_list, new_length):
    # Calculate the step size for interpolation
    step = (len(original_list) - 1) / (new_length - 1)

    # Initialize the new list
    new_list = []

    # Perform linear interpolation
    for i in range(new_length):
        # Calculate the index of the two closest points in the original list
        idx1 = int(np.floor(i * step))
        idx2 = int(np.ceil(i * step))

        # Calculate the fractional distance between the two points
        frac = i * step - idx1
        # Perform linear interpolation between the two points
        interpolated_value = (1 - frac) * original_list[idx1] + frac * original_list[idx2]
        # Add the interpolated value to the new list
        new_list.append(interpolated_value)

    return new_list


"""
MAIN
"""

class AudioVis:
    def __init__(self):
        self.frequencies = None
        self.spectrum = None

        # Set the Matplotlib backend to Qt
        plt.switch_backend('Qt5Agg')
        # Get the Qt widget associated with the figure
        self.window = plt.get_current_fig_manager().window

        # Maximize figure window and position it to the top-left corner
        self.window.showMaximized()  # Maximize window

        # Remove margins at the top and left
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # Remove menu bar
        self.window.setMenuBar(None)

        """
        GLOBALS
        """
        # Parameters for audio capture
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024

        self.ROLLING_WINDOW_RATIO = 0.05  # Ratio of the chunk size for the rolling window

        # Define frequency bins
        self.LOW_FREQ = (20, 200)  # Low frequency range
        self.MID_FREQ = (200, 2000)  # Middle frequency range
        self.HIGH_FREQ = (2000, 20000)  # High frequency range

        # Visual parameters
        self.T = 0.001  # Decay parameter varies from 0 to 1, is a fraction of total opacity
        self.W = 0.01
        self.MAX_FRAMES = 20

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()

        # Open audio stream
        self.stream = self.p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)

        # Create plot
        plt.rcParams["figure.figsize"] = (20, 20)
        plt.rcParams["figure.facecolor"] = "black"
        # fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.fig = plt.figure()

        # ax = plt.subplot(polar=True)  # This is the one I want to normally use, with a polar projection

        self.ax = plt.subplot(polar=True)
        self.fig.patch.set_facecolor("black")
        self.ax.set_facecolor("black")

        self.ax.set_xticks([])  # Hide x-axis ticks
        self.ax.set_yticks([])  # Hide y-axis ticks

        plt.xlim(0, 1.564)
        plt.ylim(0, 1000)  # Experimenting with different y-lim, having this change dynamically would be cool

        x = np.arange(0, 2 * self.CHUNK, 2)

        self.lines_low, = self.ax.plot(x, np.random.rand(self.CHUNK), alpha=0.9, color="red", linestyle=":")
        self.lines_mid, = self.ax.plot(x, np.random.rand(self.CHUNK), alpha=0.9, color="blue", linestyle="-")
        self.lines_high, = self.ax.plot(x, np.random.rand(self.CHUNK), alpha=0.4, color="green", linestyle="--")

        self.lines = [self.lines_low, self.lines_mid, self.lines_high]

        # Grab initial data
        self.data = np.frombuffer(self.stream.read(self.CHUNK), dtype=np.int16)

        # Rectify the data
        self.rectified_data = np.abs(self.data)
        # Rectify the audio data by taking absolute value
        self.log_data = np.log(np.abs(self.data))

        # Initialize rolling average
        self.ROLLING_WINDOW_SIZE = int(self.ROLLING_WINDOW_RATIO * self.CHUNK)

        # Build a set of rolling windows
        self.rolling_window = np.zeros(self.ROLLING_WINDOW_SIZE)

        # Making a rolling average the length of the window
        self.rolling_average_low = np.zeros(self.ROLLING_WINDOW_SIZE)
        self.rolling_average_mid = np.zeros(self.ROLLING_WINDOW_SIZE)
        self.rolling_average_high = np.zeros(self.ROLLING_WINDOW_SIZE)

    # Update function
    def update_plot(self, frame):
        self.data = np.frombuffer(self.stream.read(self.CHUNK), dtype=np.int16)
        # Rectify the data
        self.rectified_data = np.abs(self.data)
        # Rectify the audio data by taking absolute value
        self.log_data = np.log(np.abs(self.data))

        # Calculate frequency bins
        self.frequencies = np.fft.fftfreq(self.CHUNK, 1 / self.RATE)
        self.spectrum = np.fft.fft(self.log_data)

        # Calculate average amplitudes in frequency bins
        low_mask = (self.frequencies >= self.LOW_FREQ[0]) & (self.frequencies <= self.LOW_FREQ[1])
        mid_mask = (self.frequencies >= self.MID_FREQ[0]) & (self.frequencies <= self.MID_FREQ[1])
        high_mask = (self.frequencies >= self.HIGH_FREQ[0]) & (self.frequencies <= self.HIGH_FREQ[1])

        self.rolling_average_low = interpolate_list(low_mask, 1024)
        self.rolling_average_mid = interpolate_list(mid_mask, 1024)
        self.rolling_average_high = interpolate_list(high_mask, 1024)

        # Calculate rolling average
        for i in range(len(self.data)):
            self.rolling_window[:-1] = self.rolling_window[1:]  # Shift values to the left
            self.rolling_window[-1] = self.log_data[i]

            # Calculate rolling average of the last 50 samples
            self.rolling_average_low[:-1] = self.rolling_average_low[1:]
            self.rolling_average_low[i - 1] = self.np.mean(np.abs(self.spectrum[low_mask]))

            self.rolling_average_mid[:-1] = self.rolling_average_mid[1:]
            self.rolling_average_mid[i - 1] = np.mean(np.abs(self.spectrum[mid_mask]))

            self.rolling_average_high[:-1] = self.rolling_average_high[1:]
            self.rolling_average_high[i - 1] = np.mean(np.abs(self.spectrum[high_mask]))

        # Update lines

        self.lines_low.set_ydata(self.log_data)
        self.lines_mid.set_ydata(self.rectified_data)
        self.lines_high.set_ydata(self.rolling_average_low)

        plt.ylim(0, max(self.rectified_data))
        # Choosing the rolling average low because it should change slowest

        red_colors = (min((self.rolling_average_low[-1] / 20), 0.8), min((self.rolling_average_mid[-1] / 150), 0.8),
                      min((self.rolling_average_high[-1] / 40), 0.2), 0.4)
        # These divisors should be on sliders and eventually continuous input

        green_colors = (min((self.rolling_average_high[-1] / 50), 0.2), min((self.rolling_average_mid[-1] / 20), 0.2),
                        min((self.rolling_average_low[-1] / 60), 0.2), 0.4)

        blue_colors = (min((self.rolling_average_mid[-1] / 30), 0.2), min((self.rolling_average_low[-1] / 20), 0.3),
                       min((self.rolling_average_high[-1] / 100), 0.2), 0.4)

        # Update line colors
        self.lines_low.set_color(red_colors)
        self.lines_mid.set_color(green_colors)
        self.lines_high.set_color(blue_colors)

        return self.lines_low, self.lines_mid, self.lines_high

    def start_animation(self):
        # Start animation
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=5, blit=True, cache_frame_data=False)

        # Set the animation window to be borderless
        fig_manager = plt.get_current_fig_manager()

        plt.show()
        # Close the stream and terminate PyAudio
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        plt.close()

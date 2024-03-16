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

# Set the Matplotlib backend to Qt
plt.switch_backend('Qt5Agg')
# Get the Qt widget associated with the figure
window = plt.get_current_fig_manager().window

# Maximize figure window and position it to the top-left corner
window.showMaximized()  # Maximize window

# Remove margins at the top and left
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Remove menu bar
window.setMenuBar(None)


"""
GLOBALS
"""
# Parameters for audio capture
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
ROLLING_WINDOW = 50  # Size of the rolling window for calculating the rolling average
ROLLING_WINDOW_RATIO = 0.05  # Ratio of the chunk size for the rolling window


# Define frequency bins
LOW_FREQ = (20, 200)  # Low frequency range
MID_FREQ = (200, 2000)  # Middle frequency range
HIGH_FREQ = (2000, 20000)  # High frequency range

# Visual parameters
T = 0.001  # Decay parameter varies from 0 to 1, is a fraction of total opacity
W = 0.01
MAX_FRAMES = 20

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Create plot
plt.rcParams["figure.figsize"] = (20,20)
plt.rcParams["figure.facecolor"] = "black"
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
fig = plt.figure()

# ax = plt.subplot(polar=True)  # This is the one I want to normally use, with a polar projection

ax = plt.subplot(polar=True)
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

ax.set_xticks([])  # Hide x-axis ticks
ax.set_yticks([])  # Hide y-axis ticks

plt.xlim(0, 1.564)
plt.ylim(0, 1000)  # Experimenting with different y-lim, having this change dynamically would be cool

x = np.arange(0, 2 * CHUNK, 2)

lines_low, = ax.plot(x, np.random.rand(CHUNK), alpha=0.9, color="red", linestyle=":")
lines_mid, = ax.plot(x, np.random.rand(CHUNK), alpha=0.9, color="blue", linestyle="-")
lines_high, = ax.plot(x, np.random.rand(CHUNK), alpha=0.4, color="green", linestyle="--")

lines = [lines_low, lines_mid, lines_high]

# Grab initial data
data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

# Rectify the data
rectified_data = np.abs(data)

# Initialize rolling average
rolling_window_size = int(ROLLING_WINDOW_RATIO * CHUNK)

# Build a set of rolling windows
rolling_window = np.zeros(rolling_window_size)

# Making a rolling average the length of the window
rolling_average_low = np.zeros(rolling_window_size)
rolling_average_mid = np.zeros(rolling_window_size)
rolling_average_high = np.zeros(rolling_window_size)

# Update function
def update_plot(frame):
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    # Rectify the data
    rectified_data = np.abs(data)
    # Rectify the audio data by taking absolute value
    log_data = np.log(np.abs(data))

    # Calculate frequency bins
    frequencies = np.fft.fftfreq(CHUNK, 1 / RATE)
    spectrum = np.fft.fft(log_data)

    # Calculate average amplitudes in frequency bins
    low_mask = (frequencies >= LOW_FREQ[0]) & (frequencies <= LOW_FREQ[1])
    mid_mask = (frequencies >= MID_FREQ[0]) & (frequencies <= MID_FREQ[1])
    high_mask = (frequencies >= HIGH_FREQ[0]) & (frequencies <= HIGH_FREQ[1])

    rolling_average_low = interpolate_list(low_mask, 1024)
    rolling_average_mid = interpolate_list(mid_mask, 1024)
    rolling_average_high = interpolate_list(high_mask, 1024)


    # Calculate rolling average
    for i in range(len(data)):
        rolling_window[:-1] = rolling_window[1:]  # Shift values to the left
        rolling_window[-1] = log_data[i]

        # Calculate rolling average of the last 50 samples
        rolling_average_low[:-1] = rolling_average_low[1:]
        rolling_average_low[i-1] = np.mean(np.abs(spectrum[low_mask]))

        rolling_average_mid[:-1] = rolling_average_mid[1:]
        rolling_average_mid[i-1] = np.mean(np.abs(spectrum[mid_mask]))

        rolling_average_high[:-1] = rolling_average_high[1:]
        rolling_average_high[i-1] = np.mean(np.abs(spectrum[high_mask]))

    # Update lines

    lines_low.set_ydata(log_data)
    lines_mid.set_ydata(rectified_data)
    lines_high.set_ydata(rolling_average_low)


    plt.ylim(0, max(rectified_data))
    # Choosing the rolling average low because it should change slowest


    red_colors = (min((rolling_average_low[-1]/20), 0.8),min((rolling_average_mid[-1]/150), 0.8),
                  min((rolling_average_high[-1]/40), 0.2), 0.4)
    # These divisors should be on sliders and eventually continuous input

    green_colors = (min((rolling_average_high[-1]/50), 0.2), min((rolling_average_mid[-1]/20), 0.2),
                    min((rolling_average_low[-1]/60), 0.2), 0.4)

    blue_colors = (min((rolling_average_mid[-1]/30), 0.2), min((rolling_average_low[-1]/20), 0.3),
                   min((rolling_average_high[-1]/100), 0.2), 0.4)


    # Update line colors
    lines_low.set_color(red_colors)
    lines_mid.set_color(green_colors)
    lines_high.set_color(blue_colors)

    return lines_low, lines_mid, lines_high

# Start animation
ani = FuncAnimation(fig, update_plot, interval=5, blit=True, cache_frame_data=False)

# Set the animation window to be borderless
fig_manager = plt.get_current_fig_manager()

plt.show()
# Close the stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()
plt.close()
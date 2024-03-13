import matplotlib
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from PyQt5 import QtWidgets


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

# Parameters for audio capture
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
ROLLING_WINDOW = 50  # Size of the rolling window for calculating the rolling average


# Visual parameters
T = 0.01
W = 0.3

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Create plot
matplotlib.rcParams["figure.figsize"] = (20,20)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#
fig.patch.set_visible(False)  # Hide figure background
ax.set_xticks([])  # Hide x-axis ticks
ax.set_yticks([])  # Hide y-axis ticks

x = np.arange(0, 2 * CHUNK, 2)
# lines = ax.plot(x, np.random.rand(CHUNK), alpha=0.8, color="red")
lines = ax.plot(x, np.random.rand(CHUNK), alpha=0.8, color="red")

# Initialize rolling average
rolling_average = np.zeros(ROLLING_WINDOW)

# Update function
def update_plot(frame):
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

    # Calculate rolling average
    rolling_average[:-1] = rolling_average[1:]  # Shift values to the left
    rolling_average[-1] = np.mean(np.abs(data))  # Calculate new rolling average


    for line in lines:
        # Set data for each line
        line.set_ydata(rolling_average)  # Using this as a filter

        # Adjust opacity
        alpha = line.get_alpha()
        if alpha > 0:
            line.set_alpha(max(0, alpha - T / 10.0))  # Decaying opacity
        else:
            line.set_alpha(0)  # Ensure opacity doesn't go negative

        # Rotate line
        angle = np.deg2rad(W * frame)  # Convert degrees to radians
        x_data, y_data = line.get_xdata(), line.get_ydata()
        x_rotated = x_data * np.cos(angle) - y_data * np.sin(angle)
        y_rotated = x_data * np.sin(angle) + y_data * np.cos(angle)
        line.set_data(x_rotated, y_rotated)
        return lines

# Start animation
ani = FuncAnimation(fig, update_plot, interval=8, blit=True, cache_frame_data=False)
plt.show()
# Close the stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()
plt.close()
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

# Visual parameters
T = 0.2
W = 1

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Create plot
matplotlib.rcParams["figure.figsize"] = (15,12)
fig, ax = plt.subplots()
#
fig.patch.set_visible(False)  # Hide figure background
ax.set_xticks([])  # Hide x-axis ticks
ax.set_yticks([])  # Hide y-axis ticks

x = np.arange(0, 2 * CHUNK, 2)
lines = ax.plot(x, np.random.rand(CHUNK), alpha=1.0)

# Update function
def update_plot(frame):
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    for line in lines:
        # Set data for each line
        line.set_ydata(data)

        # Adjust opacity
        alpha = line.get_alpha()
        if alpha > 0:
            line.set_alpha(max(0, alpha - T / 100.0))  # Decaying opacity
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
ani = FuncAnimation(fig, update_plot, blit=True)
plt.show()
# Close the stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()
plt.close()
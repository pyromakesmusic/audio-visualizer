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
ROLLING_WINDOW_RATIO = 0.05  # Ratio of the chunk size for the rolling window


# Define frequency bins
LOW_FREQ = (20, 200)  # Low frequency range
MID_FREQ = (200, 2000)  # Middle frequency range
HIGH_FREQ = (2000, 20000)  # High frequency range

# Visual parameters
T = 0.4  # Decay parameter varies from 0 to 1, is a fraction of total opacity
W = 0.8
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
plt.axes().set_facecolor('black')  # Set plot background color

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_facecolor("black")
#
fig.patch.set_visible(False)  # Hide figure background
ax.set_xticks([])  # Hide x-axis ticks
ax.set_yticks([])  # Hide y-axis ticks

x = np.arange(0, 2 * CHUNK, 2)
lines_0, = ax.plot(x, np.random.rand(CHUNK), alpha=0.9, color="red")
lines_low, = ax.plot(x, np.random.rand(CHUNK), alpha=0.9, color="red")
lines_mid, = ax.plot(x, np.random.rand(CHUNK), alpha=0.9, color="green")
lines_high, = ax.plot(x, np.random.rand(CHUNK), alpha=0.9, color="blue")

lines = [lines_0, lines_low, lines_mid, lines_high]

# Initialize rolling average
rolling_window_size = int(ROLLING_WINDOW_RATIO * CHUNK)

# Grab initial data
data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

# Rectify the data
rectified_data = np.abs(data)

rolling_window = np.zeros(rolling_window_size)

rolling_average = np.zeros(len(rectified_data))

rolling_average_low = np.zeros(rolling_window_size)
rolling_average_mid = np.zeros(rolling_window_size)
rolling_average_high = np.zeros(rolling_window_size)

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

# Update function
def update_plot(frame):
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

    # Rectify the audio data by taking absolute value
    rectified_data = np.abs(data)

    # Perform FFT
    spectrum = np.fft.fft(rectified_data)

    # Calculate frequency bins
    frequencies = np.fft.fftfreq(CHUNK, 1 / RATE)

    # Calculate average amplitudes in frequency bins
    low_mask = (frequencies >= LOW_FREQ[0]) & (frequencies <= LOW_FREQ[1])
    mid_mask = (frequencies >= MID_FREQ[0]) & (frequencies <= MID_FREQ[1])
    high_mask = (frequencies >= HIGH_FREQ[0]) & (frequencies <= HIGH_FREQ[1])

    rolling_average_low = interpolate_list(low_mask, 1024)
    rolling_average_mid = interpolate_list(mid_mask, 1024)
    rolling_average_high = interpolate_list(high_mask, 1024)


    # Calculate rolling average
    for i in range(len(rectified_data)):
        # Perform FFT
        spectrum = np.fft.fft(rectified_data)

        # Calculate frequency bins
        frequencies = np.fft.fftfreq(CHUNK, 1 / RATE)

        # Calculate average amplitudes in frequency bins
        low_mask = (frequencies >= LOW_FREQ[0]) & (frequencies <= LOW_FREQ[1])
        mid_mask = (frequencies >= MID_FREQ[0]) & (frequencies <= MID_FREQ[1])
        high_mask = (frequencies >= HIGH_FREQ[0]) & (frequencies <= HIGH_FREQ[1])

        # Add current sample to rolling window
        rolling_window[:-1] = rolling_window[1:]  # Shift values to the left
        rolling_window[-1] = rectified_data[i]

        # Calculate rolling average of the last 50 samples
        rolling_average[i] = np.mean(rolling_window)

        rolling_average_low[:-1] = rolling_average_low[1:]
        rolling_average_low[i-1] = np.mean(np.abs(spectrum[low_mask]))

        rolling_average_mid[:-1] = rolling_average_mid[1:]
        rolling_average_mid[i-1] = np.mean(np.abs(spectrum[mid_mask]))

        rolling_average_high[:-1] = rolling_average_high[1:]
        rolling_average_high[i-1] = np.mean(np.abs(spectrum[high_mask]))

    # Update lines

    lines_0.set_ydata(rectified_data)
    lines_low.set_ydata(rolling_average_low)
    lines_mid.set_ydata(rolling_average_mid)
    lines_high.set_ydata(rolling_average_high)
    return lines

# Start animation
ani = FuncAnimation(fig, update_plot, interval=4, blit=True, cache_frame_data=False)
plt.show()
# Close the stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()
plt.close()
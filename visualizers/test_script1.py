import matplotlib
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters for audio capture
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

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
line, = ax.plot(x, np.random.rand(CHUNK))

# Maximize figure window and position it to the top-left corner
plt.get_current_fig_manager().window.state('zoomed')  # Maximize window
plt.get_current_fig_manager().window.geometry("+0+0")  # Move window to top-left corner

# Remove margins at the top and left
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)


# Update function
def update_plot(frame):
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    line.set_ydata(data)
    return line,

# Start animation
ani = FuncAnimation(fig, update_plot, blit=True)
plt.show()


# Close the stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()
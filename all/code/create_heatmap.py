import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Sample 2D array
data = [[89.8989898989899, 88.88888888888889, 91.91919191919192, 90.9090909090909, 88.88888888888889, 86.86868686868688, 89.8989898989899],
[93.93939393939394, 92.92929292929293, 92.92929292929293, 92.92929292929293, 93.93939393939394, 90.9090909090909, 92.92929292929293],
[90.9090909090909, 89.8989898989899, 86.86868686868688, 89.8989898989899, 89.8989898989899, 86.86868686868688, 84.84848484848484],
[86.86868686868688, 86.86868686868688, 84.84848484848484, 83.83838383838383, 86.86868686868688, 84.84848484848484, 85.85858585858585]]


data = np.array(data)

# Function to plot a heatmap
def plot_heatmap(data, title):
    plt.figure(figsize=(6, 6))
    norm = Normalize(vmin=np.min(data), vmax=np.max(data))
    plt.imshow(data, cmap='RdYlGn', norm=norm)
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(False)
    plt.show()

# Plot heatmap with data's min and max
plot_heatmap(data, "Heatmap with Data's Min and Max")

# Rescale data to range 0-100
rescaled_data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 100

# Plot heatmap with range 0 to 100
plot_heatmap(rescaled_data, "Heatmap Rescaled to Range 0-100")

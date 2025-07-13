#----------------------------------------- Import libraries-------------------------------------------#
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

#----------------------------------------- Output dir -------------------------------------------------#

output_dir = "C:/Users/ardeh/OneDrive/Desktop/ARD/Plots/partB"
os.makedirs(output_dir, exist_ok=True)

#----------------------------------------- Load and read DarkFrame ------------------------------------#

dark_frames_path = "C:/Users/ardeh/OneDrive/Desktop/ARD/Darks.mat"  
dark_data = scipy.io.loadmat(dark_frames_path)
dark_frames_key = [key for key in dark_data.keys() if not key.startswith("__")][0]
dark_frames = dark_data[dark_frames_key] 

#----------------------------------- Plot first sample of DarkFrame -----------------------------------#

first_dark_frame = dark_frames[0]
fig, ax = plt.subplots(figsize=(6, 6))
plt.rcParams.update({'font.size': 14})  
im = ax.imshow(first_dark_frame, cmap='viridis', origin='upper')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label("Intensity", fontsize=14)  
ax.set_title("First Dark Frame", fontsize=14)
ax.set_xlabel("X Pixels", fontsize=14)
ax.set_ylabel("Y Pixels", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12) 
cbar.ax.tick_params(labelsize=12)  
plt.savefig(os.path.join(output_dir, "Dark_Frame_sample1.png"), dpi=200, bbox_inches='tight')
plt.show()

#----------------------------------- Plot all 10 DarkFrames------------------------------------------#

rows, cols = 2, 5
fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
vmin, vmax = np.min(dark_frames[:10]), np.max(dark_frames[:10])
for i, ax in enumerate(axes.flatten()):
    im = ax.imshow(dark_frames[i], cmap='gray', origin='upper', vmin=vmin, vmax=vmax)
    ax.set_title(f"Dark Frame {i+1}")
    ax.set_xlabel("X Pixels")
    ax.set_ylabel("Y Pixels")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
cbar_ax = fig.add_axes([0.9999, 0.05, 0.02, 0.9])
fig.colorbar(im, cax=cbar_ax, label="Pixel Intensity")
plt.savefig(os.path.join(output_dir, "Dark_Frames_grids.png"), dpi=200, bbox_inches='tight')
plt.show()

#---------------------- statistics of each dark frame in dataframe format using pandas----------------#

stats = {
    "Image Index": range(1, 11),
    "Mean": [round(np.mean(dark_frames[i]), 2) for i in range(10)],
    "Median": [round(np.median(dark_frames[i]), 2) for i in range(10)],
    "Standard Deviation": [round(np.std(dark_frames[i]), 2) for i in range(10)]
}

stats_df = pd.DataFrame(stats)
print(stats_df)


#-------------------------- Statistic and plot mean, median, std for each dark frame------------------#

#---------------------------------------- plot mean-------------------------------------------------#

plt.figure(figsize=(8, 4))
plt.rcParams.update({'font.size': 14})
plt.plot(stats_df["Image Index"], stats_df["Mean"], marker='.', linestyle='-')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.title("Mean Intensity of Each Dark Frame")
plt.xlabel("Dark Frame Index")
plt.ylabel("Mean Intensity")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Dark_Frame_Mean.png"), dpi=200, bbox_inches='tight')
plt.show()

#----------------------------------------- plot median----------------------------------------------#

plt.figure(figsize=(8, 4))
plt.rcParams.update({'font.size': 14})
plt.plot(stats_df["Image Index"], stats_df["Median"], marker='.', linestyle='-', color='g')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.title("Median Intensity of Each Dark Frame")
plt.xlabel("Dark Frame Index")
plt.ylabel("Median Intensity")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Dark_Frame_Median.png"), dpi=200, bbox_inches='tight')
plt.show()

#------------------------------------------- plot STD----------------------------------------------#

plt.figure(figsize=(8, 4))
plt.rcParams.update({'font.size': 14})
plt.plot(stats_df["Image Index"], stats_df["Standard Deviation"], marker='.', linestyle='-', color='r')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.title("Standard Deviation of Each Dark Frame")
plt.xlabel("Dark Frame Index")
plt.ylabel("Standard Deviation")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Dark_Frame_StdDev.png"), dpi=200, bbox_inches='tight')
plt.show()

#----------------------------------- plot bias and noise------------------------------------------#

bias_map = np.mean(dark_frames, axis=0, dtype=np.float32)
noise_map = np.sqrt(np.mean((dark_frames - bias_map) ** 2, axis=0, dtype=np.float32))

fig, ax = plt.subplots(figsize=(8, 8))
plt.rcParams.update({'font.size': 14})
im = ax.imshow(bias_map, cmap="viridis", origin="upper")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label("Bias Level")
ax.set_title("Bias Map")
ax.set_xlabel("Pixel X")
ax.set_ylabel("Pixel Y")
plt.savefig(os.path.join(output_dir, "Bias_Map.png"), dpi=200, bbox_inches='tight')
plt.show()



fig, ax = plt.subplots(figsize=(8, 8))
plt.rcParams.update({'font.size': 14})
im = ax.imshow(noise_map, cmap="inferno", origin="upper")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label("Noise Level")
ax.set_title("Noise Map")
ax.set_xlabel("Pixel X")
ax.set_ylabel("Pixel Y")
plt.savefig(os.path.join(output_dir, "Noise_Map.png"), dpi=200, bbox_inches='tight')
plt.show()


#----------------------------------- plot bias and noise histogram------------------------------------------#
plt.figure(figsize=(12, 6))
plt.rcParams.update({'font.size': 14})
plt.hist(bias_map.flatten(), bins=100, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel("Bias Level")
plt.ylabel("Frequency")
plt.title("Histogram of Bias Levels (Dark Frames)")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Histogram_bias_levels.png"), dpi=200, bbox_inches='tight')
plt.show()


plt.figure(figsize=(15, 6))
plt.rcParams.update({'font.size': 14})
plt.hist(dark_frames.flatten(), bins=100, alpha=0.7, edgecolor='black')
plt.xlabel("Bias Level")
plt.ylabel("Frequency")
plt.title("Histogram of Dark Frames")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Histogram_allframe.png"), dpi=200, bbox_inches='tight')
plt.show()


plt.figure(figsize=(12, 6))
plt.rcParams.update({'font.size': 14})
plt.hist(noise_map.flatten(), bins=100, color='green', alpha=0.7, edgecolor='black')
plt.xlabel("Noise Level")
plt.ylabel("Number of Pixels")
plt.title("Histogram of Noise Levels (Dark Frames)")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Histogram_Noise_Levels.png"), dpi=200, bbox_inches='tight')
plt.show()


#----------------------------------- statistics of bias and noise in dataframe------------------------------------------#

bias_map = np.mean(dark_frames, axis=0, dtype=np.float32)
noise_map = np.std(dark_frames, axis=0, dtype=np.float32)

stats = {
    "Statistic": ["Mean Bias", "Median Bias", "Std Bias", "Mean Noise", "Median Noise", "Std Noise"],
    "Value": [
        round(np.mean(bias_map), 2), round(np.median(bias_map), 2), round(np.std(bias_map), 2),
        round(np.mean(noise_map), 2), round(np.median(noise_map), 2), round(np.std(noise_map), 2)
    ]
}

stats_df_bias_noise = pd.DataFrame(stats)
pd.options.display.float_format = "{:.2f}".format  
print(stats_df_bias_noise)
#--------------------------------- Finish task B (DarkFrame statistics and plotting)------------------------------------------#
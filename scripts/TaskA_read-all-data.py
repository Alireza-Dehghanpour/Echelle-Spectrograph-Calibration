###############################
###authur: ARD##
###############################

#----------------------------------------- Import libraries-------------------------------------------#
import scipy.io
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import ace_tools as tools  # For displaying data neatly

#----------------------------------------- Data Path -------------------------------------------------#

file_paths = {
    "Darks": "C:/Users/ardeh/OneDrive/Desktop/ARD/Darks.mat",
    "Whitelamp": "C:/Users/ardeh/OneDrive/Desktop/ARD/Whitelamp.mat",
    "Flat_Frame": "C:/Users/ardeh/OneDrive/Desktop/ARD/Flat_Frame.mat",
    "CalibrationLamp": "C:/Users/ardeh/OneDrive/Desktop/ARD/CalibrationLamp.mat",
    "Calibration_Frame": "C:/Users/ardeh/OneDrive/Desktop/ARD/Calibration_Frame.mat"
}
output_dir = "C:/Users/ardeh/OneDrive/Desktop/ARD/Plots/partA"   # output dir for plots
os.makedirs(output_dir, exist_ok=True)

#----------------------------------------- Read Data -------------------------------------------------#


data_contents = {}
for key, path in file_paths.items():
    try:
        data_contents[key] = scipy.io.loadmat(path)
        print(f"Successfully loaded: {key}")
    except Exception as e:
        data_contents[key] = str(e)
        print(f"Failed to load {key}: {e}")
data_info = {}
for key, content in data_contents.items():
    if isinstance(content, dict):
        data_info[key] = {k: v.shape for k, v in content.items() if isinstance(v, np.ndarray)}
    else:
        data_info[key] = f"Error loading: {content}"

#----------------------------------- use dataframe (pandas library)------------------------------------------#

df_info = pd.DataFrame.from_dict(data_info, orient='index').round(2)
#tools.display_dataframe_to_user(name="Data Information", dataframe=df_info
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None) 
pd.set_option('display.width', 200)  
pd.set_option('display.float_format', '{:.2f}'.format) 
df_info.head()
print(df_info)

dark_frames = data_contents.get("Darks", {}).get("detector_darks", None)
flat_frame = data_contents.get("Flat_Frame", {}).get("flatfield", None)
calibration_frame = data_contents.get("Calibration_Frame", {}).get("calibration_frame", None)
white_spectrum = data_contents.get("Whitelamp", {}).get("white_spectrum", None)
white_wavelength = data_contents.get("Whitelamp", {}).get("white_wavelength", None)
calib_spectrum = data_contents.get("CalibrationLamp", {}).get("calib_spectrum", None)
calib_wavelength = data_contents.get("CalibrationLamp", {}).get("calib_wavelength", None)


if white_spectrum is not None and white_wavelength is not None:
    white_spectrum = white_spectrum.flatten()
    white_wavelength = white_wavelength.flatten()

if calib_spectrum is not None and calib_wavelength is not None:
    calib_spectrum = calib_spectrum.flatten()
    calib_wavelength = calib_wavelength.flatten()

#----------------------------------- Functions for statistics ------------------------------------------#
def compute_statistics(frame):
    if frame is None:
        return "No data available"
    return {
        "Mean": np.mean(frame),
        "Median": np.median(frame),
      #  "Std Dev": np.std(frame),
        "Min": np.min(frame),
        "Max": np.max(frame)
    }

dark_stats = compute_statistics(np.mean(dark_frames, axis=0) if dark_frames is not None else None)
flat_stats = compute_statistics(flat_frame)
calib_stats = compute_statistics(calibration_frame)

stats_df = pd.DataFrame({
    "Dark Frame (Avg)": dark_stats,
    "Flat Frame": flat_stats,
    "Calibration Frame": calib_stats
}).round(2)
stats_df.head()
print(stats_df)
#tools.display_dataframe_to_user(name="Statistical Analysis", dataframe=stats_df)

#----------------------------------- Functions for plots ------------------------------------------#

def plot_2d(frame, title, filename, figsize=(6, 6), cmap='gray'):
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(frame, cmap=cmap, origin='upper')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Pixel Intensity")
    ax.set_title(title)
    ax.set_xlabel("X Pixels")
    ax.set_ylabel("Y Pixels")
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved {title} plot as {save_path}")

#----------------------------------- Functions for 1D plots  ------------------------------------------#

def plot_1D(x, y, title, filename, figsize=(18, 7), legend_loc='upper right'):
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, label=title)
    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    #ax.legend(loc=legend_loc, frameon=True, bbox_to_anchor=(0, 1), borderaxespad=1)
    ax.legend()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved {title} plot as {save_path}")


#plot_and_save(np.mean(dark_frames, axis=0) if dark_frames is not None else None, "Averaged Dark Frame", "dark_frame_avg.png")
plot_2D(dark_frames[0], "First Dark Frame", "dark_frame_1.png", cmap="viridis")
plot_2D(flat_frame, "Flat Frame", "flat_frame.png", cmap="viridis")
plot_2D(calibration_frame, "Calibration Frame", "calibration_frame.png", cmap="viridis")
plot_1D(white_wavelength, white_spectrum, "White Lamp Spectrum", "white_lamp_spectrum.png", figsize=(18, 7))
plot_1D(calib_wavelength, calib_spectrum, "Calibration Lamp Spectrum", "calibration_lamp_spectrum.png", figsize=(18, 7), legend_loc='upper right')


#----------------------------------- Finish task A (read all data and have some info and statistic)  ------------------------------------------#
# ---------------------------------- Import Required Libraries ---------------------------------- #
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import json
import os

# --------------------------------------- Define Output Directory ------------------------------------------ #

sensitivity_output_dir = "C:/Users/ardeh/OneDrive/Desktop/ARD/Plots/partE"
os.makedirs(sensitivity_output_dir, exist_ok=True)

# ----------------------------------------- Load Flat Frame Data ------------------------------------------- #

flat_frame_file = "C:/Users/ardeh/OneDrive/Desktop/ARD/Flat_Frame.mat"
flat_frame_data = scipy.io.loadmat(flat_frame_file)
flat_frame = flat_frame_data['flatfield']

# ----------------------------------------- Plot Flat Frame Data -------------------------------------------- #

plt.rcParams.update({'font.size': 14})  
plt.figure(figsize=(8, 6))
plt.imshow(flat_frame, cmap='viridis', origin='upper', aspect='auto')
plt.colorbar(label="Intensity")
plt.title("Flat Frame ")
plt.xlabel("X-axis (Pixels)")
plt.ylabel("Y-axis (Pixels)")
plt.savefig(os.path.join(sensitivity_output_dir, "Flat_Frame_Original.png"), dpi=200, bbox_inches='tight')
plt.show()

# ---------------------------------- Load white lamp spectrum data and plot---------------------------------- #

whitelamp_file = "C:/Users/ardeh/OneDrive/Desktop/ARD/Whitelamp.mat"
whitelamp_data = scipy.io.loadmat(whitelamp_file)
white_wavelength = whitelamp_data['white_wavelength'].flatten()
white_spectrum = whitelamp_data['white_spectrum'].flatten()

plt.figure(figsize=(12, 6))
plt.plot(white_wavelength, white_spectrum, color='blue', lw=1, marker = '.')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.title("White Lamp Spectrum")
plt.grid(True)
plt.savefig(os.path.join(sensitivity_output_dir, "White_Lamp_Spectrum.png"), dpi=200, bbox_inches='tight')
plt.show()

# ---------------------------------- Load Wavelength Solution from Step G ---------------------------------- #

wavelength_solution_file = "C:/Users/ardeh/OneDrive/Desktop/ARD/data/wavelength_solution.json"
with open(wavelength_solution_file, "r") as file:
    wavelength_solution = json.load(file)

interp_white_spectrum = interp1d(white_wavelength, white_spectrum, kind='linear', fill_value="extrapolate")

# -------------------------------------- Compute Pixel Sensitivity ------------------------------------------ #

# sensitivity_map = np.zeros_like(flat_frame, dtype=np.float32)
# for order, mapping in wavelength_solution.items():
#     for pixel, wavelength in mapping.items():
#         expected_intensity = interp_white_spectrum(wavelength)  
#         if expected_intensity > 0:  
#             sensitivity_map[:, int(pixel)] = flat_frame[:, int(pixel)] / expected_intensity

# max_sensitivity = np.max(sensitivity_map)
# sensitivity_map /= max_sensitivity


sensitivity_map = np.zeros_like(flat_frame, dtype=np.float32)
for order, mapping in wavelength_solution.items():
    for pixel, wavelength in mapping.items():
        expected_intensity = interp_white_spectrum(wavelength)  
        if expected_intensity > 0:  
            sensitivity_map[:, int(pixel)] = flat_frame[:, int(pixel)] / expected_intensity
max_sensitivity = np.nanmax(sensitivity_map) 
if max_sensitivity > 0:
    sensitivity_map /= max_sensitivity

# ----------------------------------------- Plot Sensitivity Map ------------------------------------------- #

plt.figure(figsize=(8, 6))
plt.imshow(sensitivity_map, cmap='viridis', origin='upper', aspect='auto')
plt.colorbar(label="Relative Sensitivity")
plt.title("Sensitivity Map (Normalized)")
plt.xlabel("X-axis (Pixels)")
plt.ylabel("Y-axis (Pixels)")
plt.savefig(os.path.join(sensitivity_output_dir, "Sensitivity_Map.png"), dpi=200, bbox_inches='tight')
plt.show()

sensitivity_mean = np.mean(sensitivity_map)
sensitivity_min = np.min(sensitivity_map)
sensitivity_max = np.max(sensitivity_map)
sensitivity_std = np.std(sensitivity_map)


print(f"Sensitivity Statistics:")
print(f"Mean Sensitivity: {sensitivity_mean:.5f}")
print(f"Min Sensitivity: {sensitivity_min:.5f}")
print(f"Max Sensitivity: {sensitivity_max:.5f}")
print(f"Standard Deviation: {sensitivity_std:.5f}")

# -------------------------------------- Save Sensitivity Data --------------------------------------------- #

# -----------------.mat------------- #

sensitivity_mat_file = os.path.join(sensitivity_output_dir, "sensitivity_map.mat")
scipy.io.savemat(sensitivity_mat_file, {"sensitivity_map": sensitivity_map})

# -----------------json-------------- #
sensitivity_json_file = os.path.join(sensitivity_output_dir, "sensitivity_map.json")
sensitivity_list = sensitivity_map.tolist()
with open(sensitivity_json_file, "w") as json_file:
    json.dump(sensitivity_list, json_file)
#----------------------------------------------------- Finish task E-------------------------------------------------------#
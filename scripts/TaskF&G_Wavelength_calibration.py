
# ------------------------------------------ Import Libraries ------------------------------------------------ #

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import pandas as pd
import random

# ------------------------------------- Load Calibration Lamp Spectrum ---------------------------------------- #

calib_lamp_file = "C:/Users/ardeh/OneDrive/Desktop/ARD/CalibrationLamp.mat"
calib_lamp_data = scipy.io.loadmat(calib_lamp_file)
calib_wavelength = calib_lamp_data['calib_wavelength'].flatten()
calib_spectrum = calib_lamp_data['calib_spectrum'].flatten()


output_dir = "C:/Users/ardeh/OneDrive/Desktop/ARD/Plots/partF&G"
output_dir2 = "C:/Users/ardeh/OneDrive/Desktop/ARD/data"

# --------------------------------------- Plot Calibration Lamp Spectrum --------------------------------------- #

plt.figure(figsize=(10, 5))
plt.plot(calib_wavelength, calib_spectrum, color='blue', lw=1)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (Counts)")
plt.title("Calibration Lamp Spectrum")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Calibration Lamp Spectrum.png"), dpi=200, bbox_inches='tight')
plt.show()

# ---------------------------------- Detect Peaks in Calibration Lamp Spectrum ---------------------------------- #

peaks, properties = find_peaks(calib_spectrum, height=0.1 * np.max(calib_spectrum), prominence=0.05 * np.max(calib_spectrum))
plt.figure(figsize=(10, 5))
plt.plot(calib_wavelength, calib_spectrum, color='blue', lw=1, label="Calibration Spectrum")
plt.scatter(calib_wavelength[peaks], calib_spectrum[peaks], color='red', s=20, label="Detected Peaks")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (Counts)")
plt.title("Detected Peaks in Calibration Lamp Spectrum")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Detected PeaksSpectrum.png"), dpi=200, bbox_inches='tight')
plt.show()


#---------------------method 2----------------#

# calib_peaks, _ = find_peaks(calib_spectrum, prominence=10)
# detected_wavelengths = calib_wavelength[calib_peaks]
# detected_intensities = calib_spectrum[calib_peaks]

# plt.figure(figsize=(12, 6))
# plt.plot(calib_wavelength, calib_spectrum, linestyle="-", color="blue", label="Calibration Spectrum")
# plt.scatter(detected_wavelengths, detected_intensities, color="red", marker="o", label="Detected Peaks")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Intensity")
# plt.title("Detected Peaks in Calibration Lamp Spectrum")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(output_dir, "Detected_Peaks.png"), dpi=200, bbox_inches='tight')
# plt.show()



# ---------------------------------- Gaussian Fitting for Peak Refinement ---------------------------------- #

def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
fitted_params = []
plt.figure(figsize=(10, 5))
plt.plot(calib_wavelength, calib_spectrum, color='blue', lw=1, label="Calibration Spectrum")
first_fit = True 
for peak in peaks:
    fit_range = (calib_wavelength > calib_wavelength[peak] - 0.5) & (calib_wavelength < calib_wavelength[peak] + 0.5)
    x_fit = calib_wavelength[fit_range]
    y_fit = calib_spectrum[fit_range]
    if len(x_fit) < 3:  
        fitted_params.append([calib_spectrum[peak], calib_wavelength[peak], 0.2])  
        continue
    p0 = [np.max(y_fit), calib_wavelength[peak], 0.2]
    try:
        popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0)
        fitted_params.append(popt)
        label = "Gaussian Fit" if first_fit else None
        plt.plot(x_fit, gaussian(x_fit, *popt), linestyle="--", linewidth=1.5, color='red', label=label)
        first_fit = False
        
        #plt.plot(x_fit, gaussian(x_fit, *popt), linestyle="--", linewidth=1.5, label="Gaussian Fit")
    except RuntimeError:
        fitted_params.append([calib_spectrum[peak], calib_wavelength[peak], 0.2]) 
        
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (Counts)")
plt.title("Gaussian Fitting of Peaks in Calibration Lamp Spectrum")
plt.legend(fontsize=8, loc="upper right")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Gaussian Fitting.png"), dpi=200, bbox_inches='tight')
plt.show()


#-----method2-----------------------------------#

# def gaussian(x, A, mu, sigma):
#     return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# refined_wavelengths = []
# refined_intensities = []
# fit_window = 5

# for peak_idx in calib_peaks:
#     lower_bound = max(0, peak_idx - fit_window)
#     upper_bound = min(len(calib_wavelength), peak_idx + fit_window)
#     x_data = calib_wavelength[lower_bound:upper_bound]
#     y_data = calib_spectrum[lower_bound:upper_bound]
#     sigma_estimate = (x_data[-1] - x_data[0]) / 6
#     initial_guess = [max(y_data), calib_wavelength[peak_idx], sigma_estimate]
    
#     try:
#         popt, _ = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
#         refined_wavelengths.append(popt[1])
#         refined_intensities.append(popt[0])
#     except RuntimeError:
#         refined_wavelengths.append(calib_wavelength[peak_idx])
#         refined_intensities.append(calib_spectrum[peak_idx])

# refined_wavelengths = np.array(refined_wavelengths)
# refined_intensities = np.array(refined_intensities)

# plt.figure(figsize=(12, 6))
# plt.plot(calib_wavelength, calib_spectrum, linestyle="-", color="blue", label="Calibration Spectrum")
# plt.scatter(detected_wavelengths, detected_intensities, color="red", marker="o", label="Findpeaks (Integer)")
# plt.scatter(refined_wavelengths, refined_intensities, color="green", marker="x", label="Gaussian Fit (Sub-Pixel)")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Intensity")
# plt.title("Calibration Lamp Peaks: Findpeaks vs. Gaussian Fit")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(output_dir, "Gaussian_Fit_Peaks.png"), dpi=200, bbox_inches='tight')
# plt.show()


#----------------------method 3-------------------#

# def gaussian(x, amp, mean, sigma):
#     return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

# fitted_params = []
# plt.figure(figsize=(10, 5))
# plt.plot(calib_wavelength, calib_spectrum, color='blue', lw=1, label="Calibration Spectrum")

# for peak in peaks:
#     fit_range = (calib_wavelength > calib_wavelength[peak] - 0.5) & (calib_wavelength < calib_wavelength[peak] + 0.5)
#     x_fit = calib_wavelength[fit_range]
#     y_fit = calib_spectrum[fit_range]
#     p0 = [np.max(y_fit), calib_wavelength[peak], 0.2]
#     try:
#         popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0)
#         fitted_params.append(popt)
#         plt.plot(x_fit, gaussian(x_fit, *popt), linestyle="--", linewidth=1.5, label=f"Fit {popt[1]:.2f} nm")
#     except RuntimeError:
#         print(f"Could not fit peak at {calib_wavelength[peak]:.2f} nm")


# ---------------------------------- Load Calibration Frame and Orders ---------------------------------- #

calib_frame_file = "C:/Users/ardeh/OneDrive/Desktop/ARD/Calibration_Frame.mat"
calib_frame_data = scipy.io.loadmat(calib_frame_file)
calib_frame = calib_frame_data['calibration_frame']
plt.figure(figsize=(10, 6))
plt.imshow(calib_frame, cmap='viridis', origin='upper', aspect='auto')
plt.colorbar(label="Intensity")
plt.title("Calibration Frame")
plt.xlabel("X-axis (Pixels)")
plt.ylabel("Y-axis (Pixels)")
plt.savefig(os.path.join(output_dir, "Calibration Frame.png"), dpi=200, bbox_inches='tight')
plt.show()

# ------------------------------- read parameter from json file saved from part C -------------------------#

order_fits_file = "C:/Users/ardeh/OneDrive/Desktop/ARD/data/order_fits.json"
with open(order_fits_file, "r") as file:
    order_fits = json.load(file)
order_fits = {int(k.replace("order_", "")): np.array(v) for k, v in order_fits.items()}

x_vals = np.arange(calib_frame.shape[1])
plt.figure(figsize=(10, 6))
plt.imshow(calib_frame, cmap='viridis', origin='upper', aspect='auto')
for i, coeffs in order_fits.items():
    y_vals = np.polyval(coeffs, x_vals)
    plt.plot(x_vals, y_vals, linestyle="--", linewidth=1.2, label=f"Order {i+1}")
plt.colorbar(label="Intensity")
plt.title("Detected Orders in Calibration Frame")
plt.xlabel("X-axis (Pixels)")
plt.ylabel("Y-axis (Pixels)")
plt.legend(fontsize=8, loc="upper right")
plt.savefig(os.path.join(output_dir, "Detected Orders.png"), dpi=200, bbox_inches='tight')
plt.show()

# ---------------------------------- Detect Peaks in Calibration Frame ---------------------------------- #
# order_peaks = {}
# for i, coeffs in order_fits.items():
#     y_vals = np.polyval(coeffs, x_vals).astype(int)
#     y_vals = np.clip(y_vals, 0, calib_frame.shape[0] - 1)
#     intensity_profile = calib_frame[y_vals, x_vals]
#     peaks, _ = find_peaks(intensity_profile, height=0.1 * np.max(intensity_profile), prominence=0.05 * np.max(intensity_profile))
#     order_peaks[i] = peaks


order_peaks = {}
for i, coeffs in order_fits.items():
    y_vals = np.polyval(coeffs, x_vals).astype(int)
    y_vals = np.clip(y_vals, 0, calib_frame.shape[0] - 1)
    intensity_profile = calib_frame[y_vals, x_vals]
    std_dev_intensity = np.std(intensity_profile) 
    prominence = max(np.percentile(intensity_profile, 90) * 0.05, 2 * std_dev_intensity)
    peaks, _ = find_peaks(intensity_profile, height=0.1 * np.max(intensity_profile), prominence=prominence)
  #  order_peaks[i] = peaks
   # order_peaks[i] = [p for p in peaks if y_vals.min() <= p <= y_vals.max()]
    
    order_peaks[i] = [p for p in peaks if 0 <= p < len(intensity_profile)]



# ---------------------------------- Map Pixels to Wavelengths ---------------------------------- #
# pixel_to_wavelength = {}
# for i, peaks in order_peaks.items():
#     if len(peaks) > len(fitted_params):
#         peaks = peaks[:len(fitted_params)]
#     wavelength_mapping = dict(zip(peaks, [p[1] for p in fitted_params]))
#     pixel_to_wavelength[i] = wavelength_mapping


pixel_to_wavelength = {}
for i, peaks in order_peaks.items():
    peaks = sorted(peaks)  
    matched_wavelengths = []
    for j, peak in enumerate(peaks):
        if j < len(fitted_params):  
            order_factor = (i + 1) / len(order_peaks) 
            adjusted_wavelength = fitted_params[j][1] * (1 + order_factor * 0.005)  
        matched_wavelengths.append((peak, adjusted_wavelength))
    pixel_to_wavelength[i] = {pixel: wavelength for pixel, wavelength in matched_wavelengths}


# ---------------------------------- Estimate Precision of Wavelength Solution ---------------------------------- #

# ---------------------------------- Residual ---------------------------------- #

residuals = []
for order, mapping in pixel_to_wavelength.items():
    print(f"Order {order}: {list(mapping.items())[:5]}")
    
 #   for pixel in mapping:
 #       mapping[pixel] += random.uniform(-0.00001, 0.00001) 
    
    for pixel, mapped_wavelength in mapping.items():
        closest_expected_wavelength = min([p[1] for p in fitted_params], key=lambda x: abs(x - mapped_wavelength))
        residuals.append(mapped_wavelength - closest_expected_wavelength)


# ---------------------- MSE,MAE,RMSE and R2 ------------------------------- #


residuals = np.array(residuals)
mae = mean_absolute_error(np.zeros_like(residuals), residuals)
rmse = np.sqrt(mean_squared_error(np.zeros_like(residuals), residuals))
std_dev = np.std(residuals)
expected_wavelengths = np.array([p[1] for p in fitted_params[:len(residuals)]])
mapped_wavelengths = expected_wavelengths + residuals[:len(expected_wavelengths)]

#r2 = r2_score(expected_wavelengths, mapped_wavelengths)
mae_precise = np.round(mae, 3)
rmse_precise = np.round(rmse, 3)
std_dev_precise = np.round(std_dev, 3)
#r2_precise = np.round(r2, 3)


# ------------------------ Plot Residual ---------------------------------- #

plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=20, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(0, color='red', linestyle="--", linewidth=1.5, label="Zero Error")
plt.xlabel("Residual Error (nm)")
plt.ylabel("Frequency")
plt.title("Histogram of Wavelength Residuals")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Histogram of Wavelength Residuals.png"), dpi=200, bbox_inches='tight')
plt.show()

# -------- Save wavelenght solution to use in step E(Sensitivity)----------------- #

pixel_to_wavelength_fixed = {
    int(order): {int(pixel): float(wavelength) for pixel, wavelength in mapping.items()}
    for order, mapping in pixel_to_wavelength.items()
}

wavelength_solution_json = os.path.join(output_dir2, "wavelength_solution.json")
with open(wavelength_solution_json, "w") as json_file:
    json.dump(pixel_to_wavelength_fixed, json_file)

wavelength_solution_mat = os.path.join(output_dir2, "wavelength_solution.mat")
scipy.io.savemat(wavelength_solution_mat, {"wavelength_solution": pixel_to_wavelength_fixed})

#--------------------- Print and save Metric ------------------------------------#

precision_metrics_csv = os.path.join(output_dir, "precision_metrics_parfF&G.csv")
with open(precision_metrics_csv, "w") as f:
    f.write(f"Mean Absolute Error (MAE),{format(mae_precise, '.3f')}\n")
    f.write(f"Root Mean Squared Error (RMSE),{format(rmse_precise, '.3f')}\n")
    f.write(f"Standard Deviation of Residuals,{format(std_dev_precise, '.3f')}\n")
   # f.write(f"R² Score,{format(r2_precise, '.3f')}\n")


print(f" Wavelength solution saved successfully!\n"
      f"- JSON File: {wavelength_solution_json}\n"
      f"- MAT File: {wavelength_solution_mat}\n"
      f"- Precision Metrics CSV: {precision_metrics_csv}")

precision_metrics = pd.DataFrame({
    "Metric": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "Standard Deviation of Residuals"],
    "Value": [mae_precise, rmse_precise, std_dev_precise]
})
print(precision_metrics.to_string(index=False, float_format='{:,.5f}'.format))


print("Residuals (first 10 values):", residuals[:10])
print("Max residual:", np.max(residuals))
print("Min residual:", np.min(residuals))
print("Residual Standard Deviation:", np.std(residuals))
print("First 5 Expected Wavelengths:", expected_wavelengths[:5])
print("First 5 Mapped Wavelengths:", mapped_wavelengths[:5])

#----------------------------------------------------- Finish task F and G-------------------------------------------------------#
#--------------------------------------------- Import libraries---------------------------------------------#

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#----------------------------------------- Data Path and load-------------------------------------------------#
output_dir = "C:/Users/ardeh/OneDrive/Desktop/ARD/Plots/partF"
output_dir2 = "C:/Users/ardeh/OneDrive/Desktop/ARD/data"
os.makedirs(output_dir, exist_ok=True)

calibration_lamp_file = "C:/Users/ardeh/OneDrive/Desktop/ARD/CalibrationLamp.mat"

calibration_lamp_data = scipy.io.loadmat(calibration_lamp_file)
calib_wavelength = calibration_lamp_data['calib_wavelength'].squeeze()
calib_spectrum = calibration_lamp_data['calib_spectrum'].squeeze()

#---------------------------------------- Plot the Calibration Lamp Spectrum----------------------------------#

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(12, 6))
plt.plot(calib_wavelength, calib_spectrum, linestyle="-", color="blue", label="Calibration Spectrum")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.title("Calibration Lamp Spectrum")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Calibration_Lamp_Spectrum.png"), dpi=200, bbox_inches='tight')
plt.show()

#------------------------------------ Find peak using find_peak and plot---------------------------------------#

calib_peaks, _ = find_peaks(calib_spectrum, prominence=10)
detected_wavelengths = calib_wavelength[calib_peaks]
detected_intensities = calib_spectrum[calib_peaks]

plt.figure(figsize=(12, 6))
plt.plot(calib_wavelength, calib_spectrum, linestyle="-", color="blue", label="Calibration Spectrum")
plt.scatter(detected_wavelengths, detected_intensities, color="red", marker="o", label="Detected Peaks")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.title("Detected Peaks in Calibration Lamp Spectrum")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Detected_Peaks.png"), dpi=200, bbox_inches='tight')
plt.show()

#--------------------------------Find peak using gaussian for higher accuracy-----------------------------------#

def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

refined_wavelengths = []
refined_intensities = []
fit_window = 5

for peak_idx in calib_peaks:
    lower_bound = max(0, peak_idx - fit_window)
    upper_bound = min(len(calib_wavelength), peak_idx + fit_window)
    x_data = calib_wavelength[lower_bound:upper_bound]
    y_data = calib_spectrum[lower_bound:upper_bound]
    sigma_estimate = (x_data[-1] - x_data[0]) / 6
    initial_guess = [max(y_data), calib_wavelength[peak_idx], sigma_estimate]
    
    try:
        popt, _ = curve_fit(gaussian, x_data, y_data, p0=initial_guess)
        refined_wavelengths.append(popt[1])
        refined_intensities.append(popt[0])
    except RuntimeError:
        refined_wavelengths.append(calib_wavelength[peak_idx])
        refined_intensities.append(calib_spectrum[peak_idx])

refined_wavelengths = np.array(refined_wavelengths)
refined_intensities = np.array(refined_intensities)

plt.figure(figsize=(12, 6))
plt.plot(calib_wavelength, calib_spectrum, linestyle="-", color="blue", label="Calibration Spectrum")
plt.scatter(detected_wavelengths, detected_intensities, color="red", marker="o", label="Findpeaks (Integer)")
plt.scatter(refined_wavelengths, refined_intensities, color="green", marker="x", label="Gaussian Fit (Sub-Pixel)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.title("Calibration Lamp Peaks: Findpeaks vs. Gaussian Fit")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Gaussian_Fit_Peaks.png"), dpi=200, bbox_inches='tight')
plt.show()

#----------------------------------statistics and comparison between find peak and gaussian---------------------------------------#

metrics_df = pd.DataFrame({
    "Metric": ["Mean Wavelength", "Standard Deviation", "Min Wavelength", "Max Wavelength"],
    "Findpeaks (Integer)": [
        np.mean(detected_wavelengths), np.std(detected_wavelengths),
        np.min(detected_wavelengths), np.max(detected_wavelengths)
    ],
    "Gaussian Fit (Sub-Pixel)": [
        np.mean(refined_wavelengths), np.std(refined_wavelengths),
        np.min(refined_wavelengths), np.max(refined_wavelengths)
    ]
})

#-------------------------------------------------Save Metrics, -------------------------------------------------------------------#

metrics_df.to_csv(os.path.join(output_dir, "Peak_Detection_Statistics.csv"), index=False)


peaks_df = pd.DataFrame({
    "Refined Wavelength (nm)": refined_wavelengths,
    "Refined Intensity": refined_intensities
})

peaks_df.to_csv(os.path.join(output_dir, "Refined_Calibration_Peaks.csv"), index=False)


scipy.io.savemat(os.path.join(output_dir, "Refined_Calibration_Peaks.mat"), {
    "refined_wavelengths": refined_wavelengths,
    "refined_intensities": refined_intensities
})

print("Refined peaks saved successfully.")
print(f"All figures and statistics saved in: {output_dir}")
#----------------------------------------------------- Finish task F-------------------------------------------------------#
##############################################################################################################################
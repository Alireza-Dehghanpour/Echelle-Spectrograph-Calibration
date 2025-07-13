#----------------------------------------- Import libraries-------------------------------------------#
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams.update({'font.size': 14})

#----------------------------------------- Data Path and load-------------------------------------------------#

flat_file_path = "C:/Users/ardeh/OneDrive/Desktop/Flat_Frame.mat"
output_dir = "C:/Users/ardeh/OneDrive/Desktop/Plots/partD"
output_dir2 = "C:/Users/ardeh/OneDrive/Desktop/data"

os.makedirs(output_dir, exist_ok=True)

flat_mat = scipy.io.loadmat(flat_file_path)
flat_data = flat_mat['flatfield']

detector_shape = flat_data.shape
y_pixels = np.arange(detector_shape[0])

#-------------------------------------------- Mean Intensity and plot--------------------------------------------------#

mean_intensity = np.mean(flat_data, axis=1)
fig1 = plt.figure(figsize=(10, 5))
plt.plot(y_pixels, mean_intensity, color='blue', label="Mean Intensity Along Slit")
plt.xlabel("Pixel Position (Slit Direction - Y-axis)")
plt.ylabel("Mean Intensity")
plt.title("Mean Intensity Profile Along the Slit")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "Mean_Intensity.png"), dpi=200, bbox_inches='tight')
plt.show()


#-------------------------------------------- Sum Intensity and plot--------------------------------------------------#

summed_intensity_y = np.sum(flat_data, axis=1)  
plt.figure(figsize=(10, 5))
plt.plot(summed_intensity_y, color='red', lw=1)
plt.xlabel("Pixel Position (Slit Direction - Y-axis)")
plt.ylabel("Sum Intensity")
plt.title("Sum Intensity Profile Along the Slit")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Sum_Intensity.png"), dpi=200, bbox_inches='tight')
plt.show()

#-------------------------------------------- Intensity plot at specific Y--------------------------------------------------#

y_positions = [300, 1000, 1950]
for y in y_positions:
    fig = plt.figure(figsize=(12, 6))
    plt.plot(np.arange(detector_shape[1]), flat_data[y, :], color='blue', label=f"Intensity at Y = {y}")
    plt.xlabel("Pixel Position (Order Direction - X-axis)")
    plt.ylabel("Intensity")
    plt.title(f"Intensity Profile at Y = {y}")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, f"intensity_profile_Y{y}.png"), dpi=200, bbox_inches='tight')
    plt.show()

#-------------------------------------------- Fit spline and plot--------------------------------------------------#

spline_fit = UnivariateSpline(y_pixels, mean_intensity, s=5e6)
spline_curve = spline_fit(y_pixels)


fig2 = plt.figure(figsize=(10, 5))
plt.plot(y_pixels, mean_intensity, color='blue', label="Observed Mean Intensity")
plt.plot(y_pixels, spline_curve, color='magenta', linestyle='dashed', label="Spline Fit")
plt.xlabel("Pixel Position (Slit Direction - Y-axis)")
plt.ylabel("Mean Intensity")
plt.title("Spline Fit to Slit Mean Intensity Profile")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, f"spline_fit.png"), dpi=200, bbox_inches='tight')
plt.show()

#---------------------------------------evaluate spline: Metrics and csv save and plot---------------------------------#

residuals = mean_intensity - spline_curve
rmse = np.sqrt(mean_squared_error(mean_intensity, spline_curve))
r2 = r2_score(mean_intensity, spline_curve)
metrics_df = pd.DataFrame({
    "Metric": ["Mean Residual", "Standard Deviation", "Max Residual", "RMSE", "R²"],
    "Value": [np.mean(residuals), np.std(residuals), np.max(np.abs(residuals)), rmse, r2]
})

print(metrics_df)
metrics_df.to_csv(os.path.join(output_dir, "spline_fit_metrics.csv"), index=False)

#------------------------------------------------ Plot residuals-----------------------------------------------------------#

fig3 = plt.figure(figsize=(10, 5))
plt.plot(y_pixels, residuals, color='red', label="Residuals (Observed - Fitted)")
plt.axhline(0, color='black', linestyle='dashed', linewidth=1)
plt.xlabel("Pixel Position (Slit Direction - Y-axis)")
plt.ylabel("Residual Intensity")
plt.title("Residuals of Spline Fit")
plt.legend()
plt.grid()
plt.savefig(os.path.join(output_dir, "Spline_fit_residual.png"), dpi=200, bbox_inches='tight')
plt.show()

#------------------------------------------------Save to json and mat---------------------------------------------------#

spline_data = {"y_pixels": y_pixels.tolist(), "spline_fitted_values": spline_curve.tolist()}
json_path = os.path.join(output_dir2, "spline_fit.json")
with open(json_path, "w") as json_file:
    json.dump(spline_data, json_file)

mat_path = os.path.join(output_dir2, "spline_fit.mat")
scipy.io.savemat(mat_path, {"y_pixels": y_pixels, "spline_fitted_values": spline_curve})

print(f"Spline fit and plots saved in {output_dir2}")
#----------------------------------------------------- Finish task D-------------------------------------------------------#
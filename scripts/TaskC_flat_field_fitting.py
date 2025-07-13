#--------------------------------------------- Import libraries----------------------------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import os
import json
from scipy.signal import find_peaks
from scipy.io import savemat
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#----------------------------------------- Data Path and load-------------------------------------------------#

flat_file_path = "C:/Users/ardeh/OneDrive/Desktop/ARD/Flat_Frame.mat"

flat_data = scipy.io.loadmat(flat_file_path)
output_dir = "C:/Users/ardeh/OneDrive/Desktop/ARD/Plots/partc"

os.makedirs(output_dir, exist_ok=True)
flat_frame = flat_data.get('flatfield', None)

#if flat_frame is None:
#   raise ValueError("Error: Flat frame data not found in the .mat file.")

#----------------------------------- Plot first Flat frame -----------------------------------------#

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(8, 6))
plt.imshow(flat_frame, cmap='viridis', origin='upper', aspect='auto')
plt.title("Flat Frame")
plt.xlabel("X-axis (Pixels)")
plt.ylabel("Y-axis (Pixels)")
plt.colorbar(label="Intensity")
plt.savefig(os.path.join(output_dir, "Flat_Frame_Original.png"), dpi=200, bbox_inches='tight')
plt.show()

#----------------------------------- Plot Mean Intensity Along X -----------------------------------------#

mean_intensity_x = np.mean(flat_frame, axis=0)
plt.figure(figsize=(10, 5))
plt.plot(mean_intensity_x, color='blue', lw=1)
plt.xlabel("X Pixels")
plt.ylabel("Mean Intensity Along Y-Axis")
plt.title("Mean Intensity Profile Along X-Axis")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "Mean_Intensity_X.png"), dpi=200, bbox_inches='tight')
plt.show()

#----------------------------------- Plot intensity at specific Y -----------------------------------------#

y_values = [30, 1000, 1950]
for y in y_values:
    plt.figure(figsize=(10, 6))
    intensity_profile = flat_frame[y, :]
    plt.plot(intensity_profile, color='blue', lw=2)
    plt.xlabel("X Pixels")
    plt.ylabel("Intensity")
    plt.title(f"Intensity Profile at Y = {y}")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"Intensity_Profile_Y_{y}.png"), dpi=200, bbox_inches='tight')
    plt.show()

#------------------------- detect the orders (take max of every 10 pixel in x) -----------------------------#

x_samples = np.arange(0, flat_frame.shape[1], 10)
order_profiles = {x: [] for x in x_samples}

for x in x_samples:
    intensity_profile = flat_frame[:, x]
    peaks, _ = find_peaks(intensity_profile, prominence=5)
    order_profiles[x] = peaks

valid_x_samples = [x for x in x_samples if len(order_profiles[x]) == 17]
order_count = 17

orders = [[] for _ in range(order_count)]
for x in valid_x_samples:
    for i, y in enumerate(order_profiles[x]):
        orders[i].append((x, y))
        
#----------------------------------- Plot detected orders-----------------------------------------#

plt.figure(figsize=(8, 6))
plt.imshow(flat_frame, cmap='viridis', origin='upper', aspect='auto')
for i, order in enumerate(orders):
    x_vals, y_vals = zip(*order)
    plt.scatter(x_vals, y_vals, s=4, label=f"Order {i+1}")
plt.title("Detected Spectral Orders in Flat Frame")
plt.xlabel("X-axis (Pixels)")
plt.ylabel("Y-axis (Pixels)")
plt.legend(loc="upper right", fontsize=8)
plt.savefig(os.path.join(output_dir, "Flat_Frame_Detected_Orders.png"), dpi=200, bbox_inches='tight')
plt.show()

#--------------------------------- Fit poly function to each order---------------------------------#

order_fits = {}
for i, order in enumerate(orders):
    x_vals, y_vals = zip(*order)
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    if len(x_vals) > 2:
        poly_coeffs = np.polyfit(x_vals, y_vals, deg=2)
        order_fits[i] = poly_coeffs
        
#------------------------------------ Plot Fitted orders-------------------------------------------#

plt.figure(figsize=(8, 6))
plt.imshow(flat_frame, cmap='gray', origin='upper', aspect='auto')
#x_fit = np.linspace(0, flat_frame.shape[1], 200)
x_fit = np.linspace(0, flat_frame.shape[1] - 1, 200) 
for i, coeffs in order_fits.items():
    y_fit = np.polyval(coeffs, x_fit)
    plt.plot(x_fit, y_fit, linestyle="--", linewidth=1.5, label=f"Order {i+1} Fit")
plt.title("Fitted Order in Flat Frame")
plt.xlabel("X-axis (Pixels)")
plt.ylabel("Y-axis (Pixels)")
plt.legend(loc="upper right", fontsize=8)
plt.savefig(os.path.join(output_dir, "Flat_Frame_Fitted_Orders.png"), dpi=200, bbox_inches='tight')
plt.show()

#---------------------------- Evaluate the fitted functions and orders------------------------------------#

#----------------------- Reconstruct the flat frame using fitted functions--------------------------------#

reconstructed_flat_frame = np.zeros_like(flat_frame)
for i, coeffs in order_fits.items():
    x_fit = np.arange(flat_frame.shape[1])
    y_fit = np.polyval(coeffs, x_fit).astype(int)
    y_fit = np.clip(y_fit, 0, flat_frame.shape[0] - 1)
    for x, y in zip(x_fit, y_fit):
        reconstructed_flat_frame[y, x] = np.mean(flat_frame[:, x])

#---------------------------------- Compare original & reconstructed--------------------------------------#

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(flat_frame, cmap='gray', origin='upper', aspect='auto')
axs[0].set_title("Original Flat Frame")
axs[0].set_xlabel("X-axis (Pixels)")
axs[0].set_ylabel("Y-axis (Pixels)")
axs[1].imshow(reconstructed_flat_frame, cmap='gray', origin='upper', aspect='auto')
axs[1].set_title("Reconstructed Flat Frame")
axs[1].set_xlabel("X-axis (Pixels)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Flat_Frame_Reconstruction.png"), dpi=200, bbox_inches='tight')
plt.show()

#-------------------------- Metrics, save the fitted functions in mat and json-----------------------------#

metrics = []
for i, coeffs in order_fits.items():
    x_fit = np.array(valid_x_samples)
    y_fit = np.polyval(coeffs, x_fit).astype(int)
    y_original = np.array([order_profiles[x][i] for x in valid_x_samples])

    if len(y_original) == len(y_fit):
        mse = mean_squared_error(y_original, y_fit)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_original, y_fit)
        r2 = r2_score(y_original, y_fit)
    #    metrics.append((i, mse, rmse, mae, r2))

        metrics.append((i + 1, mse, rmse, mae, r2))
        
#--------------------------------------------- metrics in pandas--------------------------------------------#

metrics_df = pd.DataFrame(metrics, columns=["Order", "MSE", "RMSE", "MAE", "R² Score"])
metrics_df.to_csv(os.path.join(output_dir, "Flat_Frame_Fit_Metrics.csv"), index=False)
print(metrics_df)


order_fits_dict = {f"order_{i}": coeffs.tolist() for i, coeffs in order_fits.items()}

#---------------------------------------------- Save as JSON-------------------------------------------------#

json_path = "C:/Users/ardeh/OneDrive/Desktop/ARD/data/order_fits.json"
with open(json_path, "w") as json_file:
    json.dump(order_fits_dict, json_file)

#---------------------------------------------- Save as .mat-------------------------------------------------#

mat_path = "C:/Users/ardeh/OneDrive/Desktop/ARD/data/order_fits.mat"
savemat(mat_path, order_fits_dict)

print(f"Saved fitted order functions to:\n- {json_path}\n- {mat_path}")

#--------------------------------- Finish task C (FlatFrame, fitting, statistics and plotting)------------------------------------------#
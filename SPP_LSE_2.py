import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation



def ecef2lla(x, y, z):
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e2))
    for _ in range(10):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))
    return np.degrees(lat), np.degrees(lon), h

def enu2ecef(east, north, up, ref_lat, ref_lon, ref_alt=0.0):
    """
    Convert ENU (East, North, Up) coordinates to ECEF (X, Y, Z).
    
    Parameters:
        east, north, up : array-like or scalar
            Local ENU coordinates in meters.
        ref_lat, ref_lon : float
            Reference point latitude and longitude in **degrees**.
        ref_alt : float, optional
            Reference point altitude in meters (default: 0.0).
    
    Returns:
        x, y, z : ndarray or scalar
            ECEF coordinates in meters.
    """
    lat = np.radians(ref_lat)
    lon = np.radians(ref_lon)
  
    a = 6378137.0      
    e2 = 6.69437999014e-3  


    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = a / np.sqrt(1 - e2 * sin_lat**2)  

    x0 = (N + ref_alt) * cos_lat * cos_lon
    y0 = (N + ref_alt) * cos_lat * sin_lon
    z0 = (N * (1 - e2) + ref_alt) * sin_lat

    dX = -sin_lon * east - cos_lon * sin_lat * north + cos_lon * cos_lat * up
    dY =  cos_lon * east - sin_lon * sin_lat * north + sin_lon * cos_lat * up
    dZ =                   cos_lat * north          + sin_lat * up

    x = x0 + dX
    y = y0 + dY
    z = z0 + dZ

    return x, y, z
# LSE_SPP
def lse_spp(pseudoranges, sat_positions, initial_pos=None, max_iter=15, tol=1e-3):
    N = len(pseudoranges)
    if N < 4:
        return np.array([np.nan, np.nan, np.nan, np.nan]), np.full(N, np.nan), False

    if initial_pos is None:
        initial_pos = np.array([0.0, 0.0, 6371000.0])
    state = np.concatenate([initial_pos, [0.0]])  # [x, y, z, b]

    for _ in range(max_iter):
        pos = state[:3]
        b = state[3]

        geometric_ranges = np.linalg.norm(sat_positions - pos, axis=1)
        predicted_pr = geometric_ranges + b
        residuals = pseudoranges - predicted_pr

        # Jacobian: ∂ρ/∂x = (x - xs)/r
        J = np.zeros((N, 4))
        for j in range(N):
            dx = pos - sat_positions[j]
            r = geometric_ranges[j]
            if r < 1e-6:
                raise ValueError("Satellite too close to user estimate.")
            J[j, :3] = dx / r
            J[j, 3] = 1.0

        # Normal equations (equal weights → W = I)
        JTJ = J.T @ J
        JTr = J.T @ residuals

        try:
            delta = np.linalg.solve(JTJ, JTr)
        except np.linalg.LinAlgError:
            return np.array([np.nan, np.nan, np.nan, np.nan]), residuals, False

        state += delta
        if np.linalg.norm(delta[:3]) < tol:
            return state, residuals, True

    return state, residuals, False
def lla2ecef(lat, lon, alt):
        # WGS84 constants
        a = 6378137.0
        e2 = 6.69437999014e-3
        lat = np.radians(lat)
        lon = np.radians(lon)
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        x = (N + alt) * np.cos(lat) * np.cos(lon)
        y = (N + alt) * np.cos(lat) * np.sin(lon)
        z = (N * (1 - e2) + alt) * np.sin(lat)
        return np.stack([x, y, z], axis=-1)

def ecef2enu(ecef, ref_ecef, ref_lat, ref_lon):
    lat = np.radians(ref_lat)
    lon = np.radians(ref_lon)
    R = np.array([
        [-np.sin(lon),            np.cos(lon),           0],
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]
    ])
    return (R @ (ecef - ref_ecef).T).T
def compute_pdop(sat_positions, user_pos):
    N = sat_positions.shape[0]
    if N < 4:
        return np.nan

    diff = sat_positions - user_pos
    ranges = np.linalg.norm(diff, axis=1)
    if np.any(ranges < 1e-6):
        return np.nan

    los = diff / ranges[:, None]  # unit vector from user to satellite
    G = np.hstack([-los, np.ones((N, 1))])  # geometry matrix

    try:
        Q = np.linalg.inv(G.T @ G)
        pdop = np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2])
        return pdop
    except np.linalg.LinAlgError:
        return np.nan
# KML Writer
def write_kml(latlon_list, filename="LSE_SPP_trajectory.kml"):
    with open(filename, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
        f.write('<Document>\n<Placemark>\n<name>SPP Trajectory</name>\n<LineString><coordinates>\n')
        for lat, lon, h in latlon_list:
            f.write(f"{lon:.6f},{lat:.6f},{h:.3f}\n")
        f.write('</coordinates></LineString>\n</Placemark>\n</Document>\n</kml>\n')
    print(f"KML saved to {filename}")


# Main
if __name__ == "__main__":
    data_dir = "C:/Users/ZHANG/OneDrive - The Hong Kong Polytechnic University/AAAlessons/4203lab/SPP_LSE/"

    # Load data
    pseudo_df = pd.read_csv(os.path.join(data_dir, "pseudoranges_meas.csv"), header=None)
    clock_df = pd.read_csv(os.path.join(data_dir, "satellite_clock_bias.csv"), header=None)
    iono_df = pd.read_csv(os.path.join(data_dir, "ionospheric_delay.csv"), header=None)
    tropo_df = pd.read_csv(os.path.join(data_dir, "tropospheric_delay.csv"), header=None)
    satpos_df = pd.read_csv(os.path.join(data_dir, "satellite_positions.csv"), header=None)
    truth_df = pd.read_csv(os.path.join(data_dir, "NAV-HPPOSECEF.csv"))

    # Reshape satellite positions: (max_sats, 3*num_epochs) → (num_epochs, max_sats, 3)
    max_sats = pseudo_df.shape[0]
    num_epochs = pseudo_df.shape[1]
    satpos_3d = satpos_df.values.reshape((max_sats, num_epochs, 3), order='C').transpose(1, 0, 2)
    # print(satpos_3d)
    # Align truth
    truth_times = truth_df["iTOW"] / 1000.0
    # print(truth_times)
    obs_start_time = 92315.0
    obs_times = obs_start_time + np.arange(num_epochs)
    # print(obs_times)
    truth_ecef = np.full((num_epochs, 3), np.nan)

    for i, t in enumerate(obs_times):
        idx = (np.abs(truth_times - t)).argmin()
        if abs(truth_times.iloc[idx] - t) < 1.0:
            truth_ecef[i] = [
                truth_df.iloc[idx]["ecefX"] / 100.0,
                truth_df.iloc[idx]["ecefY"] / 100.0,
                truth_df.iloc[idx]["ecefZ"] / 100.0,
            ]
    # print(truth_ecef)
    # SPP Loop
    estimated_ecef = []
    converged_epochs = 0
    last_pos = None
    pdop_list = []
    for epoch in range(num_epochs):
        # if epoch == 1:
        #     break
        sat_xyz = satpos_3d[epoch]
        # print(f"Epoch {epoch}, {sat_xyz}")
        pr = pseudo_df.iloc[:, epoch].values
        clk = clock_df.iloc[:, epoch].values
        ion = iono_df.iloc[:, epoch].values
        trp = tropo_df.iloc[:, epoch].values

        valid = ~np.isnan(pr) & ~np.isnan(clk) & ~np.isnan(ion) & ~np.isnan(trp) & ~np.isnan(sat_xyz).any(axis=1)
        if valid.sum() < 4:
            estimated_ecef.append([np.nan, np.nan, np.nan])
            continue

        pr = pr[valid]
        clk = clk[valid]
        ion = ion[valid]
        trp = trp[valid]
        sat_xyz = sat_xyz[valid]

        pr_corrected = pr + clk - ion - trp  # apply corrections

        if last_pos is not None:
            x0_pos = last_pos[:3]
        else:
            x0_pos = np.array([0, 0, 0])  #zero initial guess

        state, residuals, converged = lse_spp(
            pseudoranges=pr_corrected,
            sat_positions=sat_xyz,
            initial_pos=x0_pos,
            max_iter=20,
            tol=1e-7
        )

        if converged:
            x_est = state[:3]
            norm = np.linalg.norm(x_est)
            if 6e6 < norm < 7e6:
                estimated_ecef.append(x_est)
                converged_epochs += 1
                last_pos = state.copy()
                # print(f"Epoch {epoch}: converged")
            else:
                # print(f"Unrealistic position at epoch {epoch}, norm = {norm:.0f} m")
                estimated_ecef.append([np.nan, np.nan, np.nan])
        else:
            # print(f"Epoch {epoch}: LSE failed to converge")
            estimated_ecef.append([np.nan, np.nan, np.nan])


        pdop_val = np.nan
        if valid.sum() >= 4 and not np.isnan(truth_ecef[epoch]).any():
            # Use true position for PDOP
            pdop_val = compute_pdop(sat_xyz, truth_ecef[epoch])
        pdop_list.append(pdop_val)
        # print(f"PDOP: {pdop_val:.2f}")
        # print(pdop_list)

    estimated_ecef = np.array(estimated_ecef)
    valid_mask = ~np.isnan(truth_ecef).any(axis=1)
    ref_ecef = truth_ecef[valid_mask][0]  
    ref_lla = ecef2lla(ref_ecef[0], ref_ecef[1], ref_ecef[2])
    # print("Reference Position:", ref_lla)
    estimated_enu = ecef2enu(estimated_ecef[valid_mask],ref_ecef, ref_lla[0],ref_lla[1])
    truth_enu = ecef2enu(truth_ecef[valid_mask],ref_ecef, ref_lla[0],ref_lla[1])
    # print('truth_enu',truth_enu)
    # print('estimated_enu',estimated_enu)
    temp = np.array(truth_enu - estimated_enu)
    # print('temp',temp)
    error_2D = np.sqrt(temp[:,0]**2 + temp[:,1]**2)
    # print('2D',error_2D)
    error_3D = np.sqrt(temp[:,0]**2 + temp[:,1]**2 +temp[:,2]**2)
    # print('3D',error_3D)
    valid_mask_2D = ~np.isnan(error_2D)
    valid_mask_3D = ~np.isnan(error_3D)
    error_2D_valid = error_2D[valid_mask_2D]
    error_3D_valid = error_3D[valid_mask_3D]

    print(f"2D Error Mean: {np.mean(error_2D_valid):.2f} m, RMS: {np.sqrt(np.mean(error_2D_valid**2)):.2f} m")
    print(f"3D Error Mean: {np.mean(error_3D_valid):.2f} m, RMS: {np.sqrt(np.mean(error_3D_valid**2)):.2f} m")

    # KML output
    spp_llh = []
    for ecef in estimated_ecef:
        if not np.isnan(ecef).any():
            lat, lon, h = ecef2lla(*ecef)
            spp_llh.append([lat, lon, h])
            # print(f"SPP LLH: {lat:.6f}, {lon:.6f}, {h:.3f}")
        else:
            spp_llh.append([np.nan, np.nan, np.nan])
        
    valid_llh = [p for p in spp_llh if not np.isnan(p[0])]
    if valid_llh:
        write_kml(valid_llh, os.path.join(data_dir, "LSE_SPP_trajectory.kml"))

    print("\nDone.")

    # After the loop
    pdop_array = np.array(pdop_list)
    epochs_all = np.arange(num_epochs)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_all, pdop_array, 'm.-', label='PDOP')
    plt.xlabel('Epoch')
    plt.ylabel('PDOP')
    plt.title('PDOP Computed Using Ground Truth Position')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "pdop_vs_epoch_truth.png"), dpi=300)
    plt.show()

    # Stats
    valid_pdop = pdop_array[~np.isnan(pdop_array)]
    if len(valid_pdop) > 0:
        print(f"\nPDOP (truth-based):")
        print(f"Mean: {np.mean(valid_pdop):.2f}")
        print(f"Min:  {np.min(valid_pdop):.2f}")
        print(f"Max:  {np.max(valid_pdop):.2f}")
    #visualization
    # truth_enu = ecef2enu(truth_ecef[valid_mask], ref_ecef, ref_lla[0], ref_lla[1])
    # estimated_enu = ecef2enu(estimated_ecef[valid_mask], ref_ecef, ref_lla[0], ref_lla[1])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(truth_enu[:, 0], truth_enu[:, 1], truth_enu[:, 2],
            'g-', label='Ground Truth (ENU)', linewidth=2)
    ax.plot(estimated_enu[:, 0], estimated_enu[:, 1], estimated_enu[:, 2],
            'r--', label='SPP Estimate (ENU)', linewidth=1.5)

    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    ax.set_title('3D ENU Trajectory: SPP vs Ground Truth')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "enu_trajectory_3d.png"), dpi=300)
    plt.show()

    # Visualization 2: Positioning Error vs Time (Epoch)
    epochs_valid = np.where(valid_mask)[0]
    if len(epochs_valid) > 0:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_valid, error_2D, 'b.-', label='2D Error')
        plt.xlabel('Epoch')
        plt.ylabel('Error (m)')
        plt.title('2D Positioning Error vs Epoch')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_valid, error_3D, 'r.-', label='3D Error')
        plt.xlabel('Epoch')
        plt.ylabel('Error (m)')
        plt.title('3D Positioning Error vs Epoch')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "error_vs_epoch.png"), dpi=300)
        plt.show()


    # Visualization 3: Error Histograms
    if len(error_2D) > 0:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(error_2D, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('2D Error (m)')
        plt.ylabel('Frequency')
        plt.title(f'2D Error Distribution\nMean: {np.mean(error_2D):.2f} m, RMS: {np.sqrt(np.mean(error_2D**2)):.2f} m')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.hist(error_3D, bins=20, color='lightcoral', edgecolor='black')
        plt.xlabel('3D Error (m)')
        plt.ylabel('Frequency')
        plt.title(f'3D Error Distribution\nMean: {np.mean(error_3D):.2f} m, RMS: {np.sqrt(np.mean(error_3D**2)):.2f} m')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "error_histograms.png"), dpi=300)
        plt.show()
        error_enu = truth_enu - estimated_enu  # shape: (N_valid, 3)
    error_east = error_enu[:, 0]   # East error
    error_north = error_enu[:, 1]  # North error
    error_up = error_enu[:, 2]     # Up error

    # 创建第一个大图：4个子图
    fig1, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle('Positioning Error Analysis', fontsize=16)

    # 子图 1: East Error
    ax1 = axes[0, 0]
    ax1.plot(epochs_valid, error_east, 'b-', linewidth=1.5, label='East Error')
    ax1.fill_between(epochs_valid, 0, error_east, color='lightblue', alpha=0.5)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax1.set_title('East Positioning Error')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('East Error (m)')
    ax1.grid(True, alpha=0.3)

    # 子图 2: North Error
    ax2 = axes[0, 1]
    ax2.plot(epochs_valid, error_north, 'r-', linewidth=1.5, label='North Error')
    ax2.fill_between(epochs_valid, 0, error_north, color='lightcoral', alpha=0.5)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_title('North Positioning Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('North Error (m)')
    ax2.grid(True, alpha=0.3)

    # 子图 3: Up Error
    ax3 = axes[1, 0]
    ax3.plot(epochs_valid, error_up, 'g-', linewidth=1.5, label='Up Error')
    ax3.fill_between(epochs_valid, 0, error_up, color='lightgreen', alpha=0.5)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax3.set_title('Up Positioning Error')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Up Error (m)')
    ax3.grid(True, alpha=0.3)

    # 子图 4: 2D & 3D Error (与之前的图类似，但颜色和样式更匹配)
    ax4 = axes[1, 1]
    ax4.plot(epochs_valid, error_2D, 'purple', linewidth=2, label='2D Error')
    ax4.plot(epochs_valid, error_3D, 'orange', linewidth=2, label='3D Error')
    ax4.fill_between(epochs_valid, 0, error_2D, color='plum', alpha=0.3)
    ax4.fill_between(epochs_valid, 0, error_3D, color='peachpuff', alpha=0.3)
    ax4.set_title('2D and 3D Positioning Errors')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Error (m)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "error_by_direction.png"), dpi=300)
    plt.show()

    # 创建第二个大图：2个子图
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 7))
    fig2.suptitle('Positioning Error Evolution and Distribution', fontsize=16)

    # 子图 1: Positioning Error Evolution (带平滑曲线)
    ax5 = axes2[0]

    # 平滑处理 (使用简单的移动平均)
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    window_size = 5  # 可根据需要调整
    if len(error_2D) >= window_size and len(error_3D) >= window_size:
        error_2D_smoothed = moving_average(error_2D, window_size)
        error_3D_smoothed = moving_average(error_3D, window_size)
        epochs_smoothed = epochs_valid[window_size//2 : -(window_size//2)] if window_size % 2 == 1 else epochs_valid[window_size//2 : -window_size//2]
    else:
        error_2D_smoothed = error_2D.copy()
        error_3D_smoothed = error_3D.copy()
        epochs_smoothed = epochs_valid.copy()

    # 绘制原始曲线
    ax5.plot(epochs_valid, error_2D, 'lightgrey', linewidth=1, label='2D Raw')
    ax5.plot(epochs_valid, error_3D, 'wheat', linewidth=1, label='3D Raw')

    # 绘制平滑曲线
    ax5.plot(epochs_smoothed, error_2D_smoothed, 'purple', linewidth=2.5, label='2D Smoothed')
    ax5.plot(epochs_smoothed, error_3D_smoothed, 'orange', linewidth=2.5, label='3D Smoothed')

    ax5.set_title('Positioning Error Evolution')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Positioning Error (m)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 子图 2: East vs North Error Distribution
    ax6 = axes2[1]

    # 使用散点图，并用 epoch 作为颜色映射
    scatter = ax6.scatter(error_east, error_north, c=epochs_valid, cmap='viridis', s=30, edgecolors='black', linewidth=0.5)
    ax6.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax6.axvline(x=0, color='k', linestyle='--', linewidth=0.8)
    ax6.set_title('East vs North Error Distribution')
    ax6.set_xlabel('East Error (m)')
    ax6.set_ylabel('North Error (m)')
    ax6.grid(True, alpha=0.3)

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax6, shrink=0.9)
    cbar.set_label('Epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "error_evolution_and_distribution.png"), dpi=300)
    plt.show()
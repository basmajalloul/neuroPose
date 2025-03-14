import streamlit as st
import os
import time
import queue
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import mne
import numpy as np
import pandas as pd
from datetime import datetime
import neurokit2 as nk
from scipy.stats import zscore
import scipy.signal as signal
from scipy.signal import welch

st.set_page_config(layout="wide")

# Constants
SCENARIOS = ["Baseline Resting", "Cognitive Load", "Stress Induction", 
             "Motor Task", "Fatigue", "Dual Task"]
HEALTH_STATUSES = ["Healthy", "MCI"]
DATA_DIR = "generated_sessions"
os.makedirs(DATA_DIR, exist_ok=True)

experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define the sampling rate
SAMPLING_RATES = {"Pose": 30}  # Hz

# Joint groups
UPPER_BODY_JOINTS = ["Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
                      "Left Wrist", "Right Wrist"]
LOWER_BODY_JOINTS = ["Left Hip", "Right Hip", "Left Knee", "Right Knee",
                      "Left Ankle", "Right Ankle"]

def compute_band_power(signal, fs, band):
    """
    Compute the power of a signal within a specified frequency band using Welch's method.

    Parameters:
    - signal: EEG time series data (1D numpy array)
    - fs: Sampling rate (Hz)
    - band: Tuple (low_freq, high_freq) defining the frequency band

    Returns:
    - Power value for the given frequency band
    """
    f, Pxx = welch(signal, fs=fs, nperseg=min(256, len(signal)))  # Compute Power Spectral Density (PSD)
    idx_band = np.logical_and(f >= band[0], f <= band[1])  # Select frequencies in the desired band
    return np.trapz(Pxx[idx_band], f[idx_band])  # Integrate power over the selected band

def detect_eeg_anomalies(eeg_df):
    """
    Identify anomalies in EEG data and return a structured list with timestamps.
    """
    anomalies = []
    amplitude_threshold = 3  # Z-score threshold for outliers

    sampling_rate = 256  # Hz
    window_size = 2 * sampling_rate  # 2 seconds
    step_size = sampling_rate  # 1-second step
    num_windows = (len(eeg_df) - window_size) // step_size + 1

    band_ranges = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (13, 30),
        "Gamma": (30, 45),
    }

    structured_anomalies = {}  # Store anomalies with time tracking

    print("DEBUG: Checking EEG anomalies...")

    for col in eeg_df.columns[1:]:  # Skip Timestamp
        signal_data = eeg_df[col].values

        for i in range(num_windows):
            start_time = i  # In seconds
            start, end = i * step_size, i * step_size + window_size
            segment = signal_data[start:end]

            for band, r in band_ranges.items():
                power = compute_band_power(segment, sampling_rate, r)

                # Debug: Check computed power values
                #print(f"DEBUG: {col} - {band} Power = {power:.5f}")

                # Define threshold dynamically
                anomaly_threshold = 0.05  # Adjust this threshold if needed
                if power > anomaly_threshold:
                    anomaly_type = f"EEG {col} - {band} Power too high"

                    if anomaly_type not in structured_anomalies:
                        structured_anomalies[anomaly_type] = [start_time, start_time]
                    else:
                        structured_anomalies[anomaly_type][1] = start_time  # Extend range

                if power <= anomaly_threshold:
                    anomaly_type = f"EEG {col} - {band} Power too low"

                    if anomaly_type not in structured_anomalies:
                        structured_anomalies[anomaly_type] = [start_time, start_time]
                    else:
                        structured_anomalies[anomaly_type][1] = start_time  # Extend range

    # Debug: Print detected anomalies before returning
    #print(f"DEBUG: Detected EEG Anomalies: {structured_anomalies}")

    # Format anomalies with start-end timestamps
    for anomaly_type, (start_time, end_time) in structured_anomalies.items():
        if anomaly_type not in structured_anomalies:
            structured_anomalies[anomaly_type] = []

        structured_anomalies[anomaly_type].append([start_time, end_time])

    return structured_anomalies

def compute_hrv_metrics(hrv_series):
    """
    Compute HRV metrics: SDNN, RMSSD, LF/HF ratio.
    """
    if len(hrv_series) < 2:
        return None  # Not enough data to compute

    # Compute SDNN (Standard deviation of NN intervals)
    sdnn = np.std(hrv_series)

    # Compute RMSSD (Root Mean Square of Successive Differences)
    diffs = np.diff(hrv_series)
    rmssd = np.sqrt(np.mean(diffs**2))

    # Approximate LF/HF ratio (requires frequency domain analysis, simplifying here)
    lf_hf_ratio = np.var(hrv_series) / (rmssd + 1e-6)  # Avoid divide by zero

    return {
        "HRV_SDNN": sdnn,
        "HRV_RMSSD": rmssd,
        "HRV_LFHF": lf_hf_ratio
    }

def detect_hrv_anomalies(hrv_df):
    """
    Detect HRV anomalies and return structured results with timestamps.
    """
    anomalies = []

    if hrv_df.empty or "HRV" not in hrv_df.columns:
        return [("HRV Data", "No HRV data available.", None, None)]

    # Compute HRV metrics
    hrv_metrics = compute_hrv_metrics(hrv_df["HRV"])
    if hrv_metrics is None:
        return [("HRV Data", "Insufficient HRV data to compute metrics.", None, None)]

    sdnn, rmssd, lf_hf_ratio = hrv_metrics["HRV_SDNN"], hrv_metrics["HRV_RMSSD"], hrv_metrics["HRV_LFHF"]

    print(f"DEBUG: HRV_SDNN={sdnn:.2f}, RMSSD={rmssd:.2f}, LF/HF={lf_hf_ratio:.2f}")

    # Define thresholds for detecting anomalies
    sdnn_threshold = 20  # Flag SDNN below 20 ms
    rmssd_threshold = 15  # Flag RMSSD below 15 ms
    lf_hf_threshold = 0.5  # Abnormal LF/HF Ratio

    # Iterate over HRV time series and detect when anomalies occur
    anomaly_ranges = {}

    for i in range(len(hrv_df)):
        timestamp = hrv_df["Timestamp"].iloc[i]
        hrv_value = hrv_df["HRV"].iloc[i]

        # Check for SDNN anomaly
        if sdnn < sdnn_threshold:
            anomaly_type = "HRV_SDNN is too low"
            if anomaly_type not in anomaly_ranges:
                anomaly_ranges[anomaly_type] = [timestamp, timestamp]  # Start and end times
            else:
                anomaly_ranges[anomaly_type][1] = timestamp  # Update end time

        # Check for RMSSD anomaly
        if rmssd < rmssd_threshold:
            anomaly_type = "HRV_RMSSD is too low"
            if anomaly_type not in anomaly_ranges:
                anomaly_ranges[anomaly_type] = [timestamp, timestamp]
            else:
                anomaly_ranges[anomaly_type][1] = timestamp

        # Check for LF/HF anomaly
        if lf_hf_ratio < lf_hf_threshold:
            anomaly_type = "HRV_LF/HF Ratio is too low"
            if anomaly_type not in anomaly_ranges:
                anomaly_ranges[anomaly_type] = [timestamp, timestamp]
            else:
                anomaly_ranges[anomaly_type][1] = timestamp

    # Format anomalies into structured output with timestamps
    for description, (start, end) in anomaly_ranges.items():
        anomalies.append((description, f"{start:.2f}s - {end:.2f}s", start, end))

    return anomalies

def detect_signal_anomalies(data_df, signal_type="EEG"):
    """
    Identifies sudden peaks, dips, or variability in EEG or HRV signals.
    Returns a structured list of detected anomalies with timestamps.
    """
    anomalies = []
    if 'Timestamp' not in data_df.columns:
        raise ValueError(f"Expected 'Timestamp' column in {signal_type} data.")

    numeric_columns = data_df.select_dtypes(include=['number']).columns.difference(['Timestamp'])
    for channel in numeric_columns:
        mean_val = data_df[channel].mean()
        std_val = data_df[channel].std()
        threshold_high = mean_val + 3 * std_val
        threshold_low = mean_val - 3 * std_val

        for idx, value in data_df[channel].items():
            timestamp = data_df.loc[idx, 'Timestamp']
            if value > threshold_high:
                anomalies.append((f"{signal_type} {channel} - Peak Detected", timestamp, timestamp))
            elif value < threshold_low:
                anomalies.append((f"{signal_type} {channel} - Dip Detected", timestamp, timestamp))

    return anomalies

def detect_pose_anomalies(pose_df):
    anomalies = []

    for index, row in pose_df.iterrows():
        timestamp = row["Timestamp"]

        for joint in pose_df.columns[1:]:  # Skip "Timestamp" column
            joint_value = row[joint]

            # Example condition: detecting sudden drops/spikes in movement
            if index > 0:
                prev_value = pose_df.at[index - 1, joint]
                change = abs(joint_value - prev_value)

                if change > 10:  # Define an appropriate threshold
                    anomalies.append(
                        f"{timestamp}s - Sudden movement in {joint} detected"
                    )

    return anomalies


def generate_unique_filename(prefix, scenario, health_status, trial_number, extension="csv"):
    """Ensure unique filenames using timestamps with microseconds and trial number."""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{scenario}_{health_status}_trial{trial_number}_{timestamp_str}.{extension}"

def generate_eeg_mne(output_dir, scenario, health_status, duration, trial_number, data_queue):
    """
    Generate EEG data using MNE-Python with realistic signals based on empirical benchmarks.
    """
    sampling_rate = 256  # Standard EEG sampling rate (Hz)
    num_samples = duration * sampling_rate
    timestamps = np.linspace(0, duration, num_samples)

    # Define EEG channels (Matching the original format)
    eeg_channels = ["Timestamp", "Fp1", "Fp2", "AF3", "AF4", "F3", "F4", "F7", "F8", "C3", "C4", "P3", "P4", "O1", "O2"]
    ch_types = ["eeg"] * (len(eeg_channels) - 1)  # Excluding timestamp
    ch_names = eeg_channels[1:]

    # Create EEG info structure
    info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)

    # Generate EEG signal (white noise + realistic power spectrum)
    rng = np.random.default_rng()
    base_eeg = rng.normal(0, 1, (len(ch_names), num_samples))

    # Apply frequency-specific power distributions based on scenario and health status
    def apply_band_modulation(signal, band, factor):
        freqs = np.fft.rfftfreq(num_samples, d=1/sampling_rate)
        spectrum = np.fft.rfft(signal)

        # Apply modulation based on target frequency band
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        spectrum[band_mask] *= factor

        return np.fft.irfft(spectrum, n=num_samples)

    # Define frequency modulations based on scenario and health status benchmarks
    band_mods = {}

    if health_status == "Healthy":
        if scenario == "Motor Task":
            band_mods = {"alpha": 0.5, "beta": 1.5}  # Reduce alpha by 50%, increase beta by 50%
        elif scenario == "Stress Induction":
            band_mods = {"alpha": 0.5, "beta": 1.7, "theta": 1.3}  # Stronger alpha suppression, increased beta and theta activity
        elif scenario == "Cognitive Load":
            band_mods = {"theta": 1.2, "alpha": 0.6, "beta": 1.6, "gamma": 1.3}  # Increase Theta, Reduce Alpha, Increase Beta and Gamma
        elif scenario == "Fatigue":
            band_mods = {"alpha": 1.2, "beta": 0.6, "theta": 1.5, "delta": 1.3}  # Increased Alpha and Theta, reduced Beta, moderate Delta increase
        elif scenario == "Dual Task":
            band_mods = {"theta": 1.3, "alpha": 0.5, "beta": 1.8, "gamma": 1.4}  # Increased Theta, Strong Alpha suppression, high Beta and Gamma increase
    elif health_status == "MCI":
        if scenario == "Baseline Resting":
            band_mods = {"delta": 1.3, "theta": 1.3, "alpha": 0.7, "beta": 0.7}  # Increase delta/theta, reduce alpha/beta
        elif scenario == "Motor Task":
            band_mods = {"alpha": 0.4, "beta": 2.0, "theta": 1.0}  # Further decrease alpha, increase beta more significantly, slight theta boost
        elif scenario == "Stress Induction":
            band_mods = {"alpha": 0.4, "beta": 1.8, "theta": 1.5}  # Reduce alpha more under stress, increase beta and theta activity significantly
        elif scenario == "Cognitive Load":
            band_mods = {"theta": 1.8, "alpha": 0.4, "beta": 1.3, "gamma": 1.7}  # Higher Theta and Gamma, Stronger Alpha Suppression, Moderate Beta increase
        elif scenario == "Fatigue":
            band_mods = {"alpha": 1.5, "beta": 0.5, "theta": 1.7, "delta": 1.5}  # Higher Alpha, strong Beta suppression, elevated Theta and Delta
        elif scenario == "Dual Task":
            band_mods = {"theta": 1.5, "alpha": 0.4, "beta": 1.6, "gamma": 1.6}  # High Theta, Strong Alpha suppression, Increased Beta and Gamma

    # Apply modulations
    for band, factor in band_mods.items():
        if band == "delta":
            band_range = (0.5, 4)
        elif band == "theta":
            band_range = (4, 7)
        elif band == "alpha":
            band_range = (8, 12)
        elif band == "beta":
            band_range = (13, 30)
        elif band == "gamma":
            band_range = (30, 50)
        base_eeg = np.array([apply_band_modulation(ch, band_range, factor) for ch in base_eeg])

    # Create MNE Raw object
    raw = mne.io.RawArray(base_eeg, info)

    # Generate unique filename
    eeg_filename = generate_unique_filename("eeg_data", scenario, health_status, trial_number)

    # Save EEG data to CSV in correct format
    eeg_df = pd.DataFrame(base_eeg.T, columns=ch_names)
    eeg_df.insert(0, "Timestamp", np.linspace(0, duration, num_samples))
    eeg_df.to_csv(os.path.join(output_dir, eeg_filename), index=False)

    # Put EEG data into the queue with the expected key format
    data_queue.put(("eeg", eeg_df))

    #print("DEBUG: Sample EEG data before saving:", eeg_df.head())

    print(f"✅ EEG Data Generated Successfully for {scenario} - {health_status} saved to {eeg_filename}")

    # 🔍 DEBUG: Compute Band Power Before Saving
    freq_bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (13, 30),
        "Gamma": (30, 45),
    }

    band_powers = {band: [] for band in freq_bands}
    
    for ch in ch_names:
        signal_data = eeg_df[ch].values
        for band, r in freq_bands.items():
            power = compute_band_power(signal_data, sampling_rate, r)
            band_powers[band].append(power)

    # Compute Mean Band Power Across Windows
    band_powers = {band: np.mean(values) for band, values in band_powers.items()}

    # 🚨 DEBUG LOG: Print Band Power Before Saving
    # for band, power in band_powers.items():
    #     print(f"DEBUG (GENERATION): Band Power for {band}: {power:.5f}")
   
def generate_hrv(output_dir, scenario, health_status, duration, trial_number, data_queue):
    """
    Generate HRV data with proper length synchronization and apply scenario-based modifications.
    """
    try:
        sampling_rate = 250  # Standard HRV sampling rate

        # === 🩺 STEP 1: Simulate ECG Signal === #
        print("🔹 Step 1: Simulating ECG signal...")
        ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=70)

        # === 🧹 STEP 2: Clean ECG & Detect R-Peaks === #
        print("🔹 Step 2: Cleaning ECG signal...")
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)

        print("🔹 Step 3: Detecting R-peaks...")
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)

        if "ECG_R_Peaks" not in rpeaks or len(rpeaks["ECG_R_Peaks"]) < 3:
            raise ValueError("❌ Not enough R-peaks detected in ECG signal!")

        print(f"✅ Found {len(rpeaks['ECG_R_Peaks'])} R-peaks.")

        # === ⏳ STEP 4: Extract RR Intervals (Before HRV Calculation) === #
        rr_intervals = np.diff(rpeaks["ECG_R_Peaks"]) / sampling_rate * 1000  # Convert to ms
        hrv_timestamps = np.cumsum(np.insert(rr_intervals, 0, 0)) / 1000  # Convert to seconds

        if len(rr_intervals) == 0:
            raise ValueError("❌ RR intervals are empty!")

        print(f"✅ HRV Data Generated: {len(hrv_timestamps)} samples.")

        # === 🔧 STEP 5: Apply Direct RR Interval Suppression for Scenarios === #
        if scenario == "Cognitive Load":
            rr_intervals *= 0.90  # Reduce variability
        elif health_status == "MCI" and scenario == "Cognitive Load":
            rr_intervals *= 0.85  # Stronger suppression for MCI
        elif scenario == "Stress Induction":
            rr_intervals *= 0.88  # Stress increases RR shortening
        elif health_status == "MCI" and scenario == "Stress Induction":
            rr_intervals *= 0.83  # More suppression for MCI Stress
        elif scenario == "Fatigue":
            rr_intervals *= 0.92  # Moderate RR suppression for Healthy
        elif health_status == "MCI" and scenario == "Fatigue":
            rr_intervals *= 0.88  # Stronger suppression for MCI

        # 🔹 RE-CALCULATE hrv_timestamps AFTER RR modification
        hrv_timestamps = np.cumsum(np.insert(rr_intervals, 0, 0)) / 1000  # Convert to seconds
        min_length = min(len(hrv_timestamps), len(rr_intervals))
        hrv_timestamps = hrv_timestamps[:min_length]
        rr_intervals = rr_intervals[:min_length]

        # === 📊 STEP 6: Extract HRV Features === #
        hrv_time_features = nk.hrv_time({"ECG_R_Peaks": rpeaks["ECG_R_Peaks"]}, sampling_rate=sampling_rate)
        sdnn = hrv_time_features.get("HRV_SDNN", pd.Series([np.nan])).values[0]
        rmssd = hrv_time_features.get("HRV_RMSSD", pd.Series([np.nan])).values[0]

        # === ⚡ STEP 7: Extract Frequency-Domain HRV Features === #
        if len(rr_intervals) >= 60:
            hrv_freq_features = nk.hrv_frequency({"ECG_R_Peaks": rpeaks["ECG_R_Peaks"]}, sampling_rate=sampling_rate)
            lf_hf_ratio = hrv_freq_features.get("HRV_LFHF", pd.Series([np.nan])).values[0]
            hf_power = hrv_freq_features.get("HRV_HF", pd.Series([np.nan])).values[0]
        else:
            lf_hf_ratio, hf_power = 1.0, 0.05  # Default values

        print(f"📊 HRV Features Before Adjustments: SDNN={sdnn:.2f}, RMSSD={rmssd:.2f}, LF/HF Ratio={lf_hf_ratio:.2f}, HF={hf_power:.2f}")

        # === 🔥 STEP 8: Apply Scenario-Based HRV Modifications === #
        scenario_mods = {
            "Cognitive Load": {"rmssd": 0.75, "sdnn": 0.60, "lf_hf": 1.4, "hf_power": 0.75},
            "Stress Induction": {"rmssd": 0.65, "sdnn": 0.20, "lf_hf": 1.3},
            "Fatigue": {"rmssd": 0.95, "sdnn": 0.80, "lf_hf": 1.0},
            "Motor Task": { "rmssd": 0.9, "sdnn": 1.05, "lf_hf": 1.15 },
            "Dual Task": {"rmssd": 0.72, "sdnn": 0.75, "lf_hf": 1.15, "hf_power": 0.65}
        }

        if health_status == "MCI":
            scenario_mods["Cognitive Load"] = {"rmssd": 0.65, "sdnn": 0.55, "lf_hf": 1.5, "hf_power": 0.6}
            scenario_mods["Stress Induction"] = {"rmssd": 0.55, "sdnn": 0.65, "lf_hf": 1.5}
            scenario_mods["Fatigue"] = {"rmssd": 0.85, "sdnn": 0.70, "lf_hf": 1.05}
            scenario_mods["Motor Task"] = { "rmssd": 0.75, "sdnn": 0.85, "lf_hf": 1.3 }

        if scenario in scenario_mods:
            mods = scenario_mods[scenario]
            adjusted_sdnn = sdnn * mods.get("sdnn", 1.0)
            adjusted_rmssd = rmssd * mods.get("rmssd", 1.0)

            if scenario == "Motor Task":
                adjusted_sdnn = min(sdnn * mods.get("sdnn", 1.0), 10.0)  # Prevent over-inflation
                adjusted_rmssd = min(rmssd * mods.get("rmssd", 1.0), 12.0)

            if health_status == "MCI" and scenario == "Dual Task":
                adjusted_sdnn = min(sdnn * 0.58, 11.3)  # Further suppress SDNN
                adjusted_rmssd = min(rmssd * 0.75, 13.3)  # Keep RMSSD under Healthy Dual Task

            # Apply upper limit to prevent exponential scaling
            sdnn = min(adjusted_sdnn, 20.0)  # Cap SDNN at 20ms
            rmssd = min(adjusted_rmssd, 20.0)  # Cap RMSSD at 20ms

        print(f"📉 Adjusted HRV: SDNN={sdnn:.2f}, RMSSD={rmssd:.2f}, LF/HF Ratio={lf_hf_ratio:.2f}, HF={hf_power:.2f}")

        # === 💾 STEP 9: Save HRV Data === #
        hrv_df = pd.DataFrame({"Timestamp": hrv_timestamps, "HRV": rr_intervals})

        if hrv_df.empty or hrv_df.isnull().values.any():
            raise ValueError("❌ HRV DataFrame contains NaN values or is empty!")

        hrv_filename = generate_unique_filename("hrv_data", scenario, health_status, trial_number)
        hrv_df.to_csv(os.path.join(output_dir, hrv_filename), index=False)

        # Send HRV data to queue
        data_queue.put(("hrv", hrv_df))
        print(f"✅ HRV Data Successfully Saved to {hrv_filename}")

    except Exception as e:
        print(f"❌ Error in HRV Generation: {e}")
        data_queue.put(("hrv", pd.DataFrame(columns=["Timestamp", "HRV"])))  # Send empty DataFrame instead of None

def generate_walking_pose(output_dir, scenario, health_status, duration, trial_number, data_queue):
    """
    Generates synthetic pose data simulating a walking cycle with all implemented scenarios.
    Now includes timestamp alignment with EEG-HRV via interpolation and last-value hold.
    """

    num_samples = duration * SAMPLING_RATES["Pose"]
    timestamps = np.linspace(0, duration, num_samples)

    pose_data = {"Timestamp": timestamps}

    # Base walking parameters
    step_frequency = 1.2 if health_status == "Healthy" else 0.8
    stride_amplitude = 35 if health_status == "Healthy" else 20
    arm_swing_amplitude = 10 if health_status == "Healthy" else 5
    noise_level = 1.5 if health_status == "Healthy" else 3

    # Default values
    upper_body_variation = 1.0  
    step_variation = 1.0  
    instability_factor = 1.0
    phase_offset = 0  

    # Scenario-specific adjustments
    step_delays = np.zeros(num_samples)
    step_phase_shifts = np.zeros(num_samples)
    jitter_spikes = np.zeros(num_samples)
    reaction_delays = np.zeros(num_samples)

    if scenario == "Baseline Resting":
        step_variation = 1.0
        instability_factor = 0.5

    elif scenario == "Cognitive Load":
        step_variation = 0.9
        instability_factor = 1.5 if health_status == "Healthy" else 3

        if health_status == "MCI":
            phase_shift_prob = 0.2
            apply_phase_shift = np.random.rand(num_samples) < phase_shift_prob
            step_delays = np.where(apply_phase_shift, np.pi / 8, 0)

        upper_body_variation = 1.1 if health_status == "Healthy" else 1.5

    elif scenario == "Stress Induction":
        step_variation = 1.3 if health_status == "Healthy" else 1.0
        instability_factor = 3 if health_status == "Healthy" else 5
        step_phase_shifts = np.random.uniform(-np.pi / 6, np.pi / 6, size=num_samples)
        upper_body_variation = 1.2 if health_status == "Healthy" else 1.8
        jitter_spikes = (
            np.random.normal(0, instability_factor, num_samples)
            * np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])
        )

    elif scenario == "Motor Task":
        step_variation = 1.5 if health_status == "Healthy" else 1.2
        instability_factor = 1 if health_status == "Healthy" else 2
        stride_amplitude *= 1.3
        arm_swing_amplitude *= 1.5
        upper_body_variation = 1.0 if health_status == "Healthy" else 1.3

    elif scenario == "Fatigue":
        step_variation = 1.0
        instability_factor = 2 if health_status == "Healthy" else 4
        fatigue_decay = np.exp(-timestamps / duration)

        stride_amplitude *= fatigue_decay
        arm_swing_amplitude *= fatigue_decay
        upper_body_variation = 1.1 if health_status == "Healthy" else 1.5

    elif scenario == "Dual Task":
        step_variation = 0.8 if health_status == "Healthy" else 0.6
        instability_factor = 2 if health_status == "Healthy" else 4
        reaction_delays = np.random.choice([0, np.pi / 12], size=num_samples, p=[0.85, 0.15])
        upper_body_variation = 1.2 if health_status == "Healthy" else 1.6

    else:
        raise ValueError(f"Scenario '{scenario}' not implemented")

    # Generate movement for all joints
    for joint in UPPER_BODY_JOINTS + LOWER_BODY_JOINTS:
        phase_offset = np.pi if "Right" in joint else 0  

        if joint in ["Left Hip", "Right Hip"]:
            movement = (stride_amplitude * step_variation) * np.sin(2 * np.pi * step_frequency * timestamps + phase_offset)

        elif joint in ["Left Knee", "Right Knee"]:
            movement = ((stride_amplitude + 15) * step_variation) * np.sin(2 * np.pi * step_frequency * timestamps + phase_offset + np.pi/2)

        elif joint in ["Left Ankle", "Right Ankle"]:
            movement = ((stride_amplitude + 20) * step_variation) * np.sin(2 * np.pi * step_frequency * timestamps + phase_offset + np.pi/4)

        elif joint in ["Left Shoulder", "Right Shoulder"]:
            movement = (arm_swing_amplitude * step_variation * upper_body_variation) * np.sin(2 * np.pi * step_frequency * timestamps + phase_offset + np.pi)

        elif joint in ["Left Elbow", "Right Elbow"]:
            movement = ((arm_swing_amplitude + 3) * step_variation * upper_body_variation) * np.sin(2 * np.pi * step_frequency * timestamps + phase_offset + np.pi/3)

        elif joint in ["Left Wrist", "Right Wrist"]:
            movement = ((arm_swing_amplitude + 5) * step_variation * upper_body_variation) * np.sin(2 * np.pi * step_frequency * timestamps + phase_offset + np.pi/6)

        else:
            movement = np.random.normal(0, noise_level, num_samples)  
            movement += np.random.normal(0, instability_factor, num_samples)

        # Apply scenario-specific modifications
        movement += step_delays + step_phase_shifts + jitter_spikes + reaction_delays
        pose_data[joint] = movement

    # Convert to DataFrame
    pose_df = pd.DataFrame(pose_data)
    pose_df["Timestamp"] = timestamps

    # 🔹 Synchronize Pose with EEG-HRV (256 Hz) 🔹
    eeg_hrv_timestamps = np.arange(0, duration, 1/256)  # 256 Hz timeline

    # Interpolate pose data to match EEG-HRV timestamps
    interpolated_pose = {}
    for joint in pose_df.columns:
        if joint != "Timestamp":
            interpolated_pose[joint] = np.interp(eeg_hrv_timestamps, pose_df["Timestamp"], pose_df[joint])

    # Last-value hold for missing frames
    interpolated_pose_df = pd.DataFrame(interpolated_pose)
    interpolated_pose_df["Timestamp"] = eeg_hrv_timestamps

    # Save synchronized pose data
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    pose_filename = generate_unique_filename("walking_pose", scenario, health_status, trial_number)
    interpolated_pose_df.to_csv(os.path.join(output_dir, pose_filename), index=False)
    data_queue.put(("pose", interpolated_pose_df))

    print(f"✅ {scenario} Pose Data Updated for {health_status}, Trial {trial_number} saved to {pose_filename}")


# def generate_annotation(output_dir, scenario, health_status, trial_number, data_queue):
#     """
#     Generate an annotation file including EEG and HRV anomaly detection, merging consecutive ranges.
#     """
#     annotation_filename = generate_unique_filename("annotation", scenario, health_status, trial_number)
#     annotation_path = os.path.join(output_dir, annotation_filename)

#     # Locate the EEG and HRV files
#     eeg_file = [f for f in os.listdir(output_dir) if f.startswith(f"eeg_data_{scenario}_{health_status}_trial{trial_number}")]
#     hrv_file = [f for f in os.listdir(output_dir) if f.startswith(f"hrv_data_{scenario}_{health_status}_trial{trial_number}")]

#     if not eeg_file:
#         print(f"❌ Missing EEG data for {scenario}, {health_status}, Trial {trial_number}")
#         return

#     eeg_df = pd.read_csv(os.path.join(output_dir, eeg_file[0]))
#     hrv_df = pd.read_csv(os.path.join(output_dir, hrv_file[0])) if hrv_file else pd.DataFrame()

#     # Detect anomalies
#     eeg_anomalies = detect_eeg_anomalies(eeg_df)

#     #print(f"DEBUG: EEG Anomalies to be included: {eeg_anomalies}")

#     hrv_anomalies = detect_hrv_anomalies(hrv_df)

#     # Convert EEG anomaly list into structured format
#     formatted_anomalies = ["EEG Anomalies"]
#     structured_anomalies = {}

#     for anomaly, values in eeg_anomalies.items():
#         if "too high" in anomaly or "too low" in anomaly:
#             start_time, end_time, _ = values  # Extract the first two values

#             print(f"{anomaly} - Start: {start_time}s, End: {end_time}s")

#             # Store properly formatted times in structured_anomalies
#             if anomaly in structured_anomalies:
#                 structured_anomalies[anomaly][1] = end_time  # Update end time
#             else:
#                 structured_anomalies[anomaly] = [start_time, end_time]  # Initialize

#         else:
#             if "Power" in anomaly:  # Ensure it's a valid power annotation
#                 formatted_anomalies.append(anomaly)

#     # Format EEG anomalies with start-end times
#     for description, (start, end) in structured_anomalies.items():
#         formatted_anomalies.append(f"{start:.1f}s - {end:.1f}s: {description}")

#     # 🔹 Add HRV anomalies properly formatted 🔹
#     if hrv_anomalies:
#         formatted_anomalies.append("\nHRV Anomalies")

#         for anomaly in hrv_anomalies:
#             description, time_range, start, end = anomaly
#             formatted_anomalies.append(f"{start:.1f}s - {end:.1f}s: {description}")

#     # Save annotations to a CSV file
#     with open(annotation_path, "w") as f:
#         f.write("\n".join(formatted_anomalies))

#     # Send annotation data to the queue
#     data_queue.put(("annotation", formatted_anomalies))
#     print(f"✅ Annotation file generated with EEG and HRV anomaly detection for {scenario}, {health_status}, Trial {trial_number}")

def generate_annotation(output_dir, scenario, health_status, trial_number, data_queue):
    """
    Generate an annotation file including EEG, HRV, and Pose anomaly detection.
    """
    annotation_filename = generate_unique_filename("annotation", scenario, health_status, trial_number)
    annotation_path = os.path.join(output_dir, annotation_filename)

    # Locate the EEG, HRV, and Pose files
    eeg_file = [f for f in os.listdir(output_dir) if f.startswith("eeg_data")][0]
    hrv_file = [f for f in os.listdir(output_dir) if f.startswith("hrv_data")][0]
    pose_file = [f for f in os.listdir(output_dir) if f.startswith("walking_pose")][0]

    eeg_df = pd.read_csv(os.path.join(output_dir, eeg_file))
    hrv_df = pd.read_csv(os.path.join(output_dir, hrv_file))
    pose_df = pd.read_csv(os.path.join(output_dir, pose_file))

    # Detect anomalies
    eeg_anomalies = detect_signal_anomalies(eeg_df, "EEG")
    hrv_anomalies = detect_signal_anomalies(hrv_df, "HRV")
    pose_anomalies = detect_pose_anomalies(pose_df)

    # Combine annotations
    all_anomalies = eeg_anomalies + hrv_anomalies + pose_anomalies

    print(f"DEBUG: All Anomalies Data Structure:\n{all_anomalies}")

    # Ensure all_anomalies is a flat list of strings
    flattened_anomalies = []
    for anomaly in all_anomalies:
        if isinstance(anomaly, list):  # If it's a list, extract its elements
            flattened_anomalies.extend(anomaly)
        else:
            flattened_anomalies.append(str(anomaly))  # Ensure it's string format

    # Now create the DataFrame correctly
    annotations = pd.DataFrame({"Annotation": flattened_anomalies})


    # Save annotations
    annotations.to_csv(annotation_path, index=False)
    print(f"✅ Annotations saved: {annotation_path}")

    data_queue.put(("annotations", annotations))

data_queue = queue.Queue()

def worker(task):
    session_dir, scenario, health_status, duration, trial_number = task
    generate_eeg_mne(session_dir, scenario, health_status, duration, trial_number, data_queue)
    generate_hrv(session_dir, scenario, health_status, duration, trial_number, data_queue)
    generate_walking_pose(session_dir, scenario, health_status, duration, trial_number, data_queue)
    generate_annotation(session_dir, scenario, health_status, trial_number, data_queue)

def generate_data_batch(session_dir, scenarios, health_statuses, num_trials, duration):
    tasks = [(session_dir, scenario, health_status, duration, trial) 
             for scenario in scenarios 
             for health_status in health_statuses 
             for trial in range(1, num_trials + 1)]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            future.result()  # Ensures execution order and catches errors.

def create_zip_folder(folder_path):
    zip_path = f"{folder_path}.zip"
    shutil.make_archive(folder_path, 'zip', folder_path)
    return zip_path

# Streamlit UI
st.title("🧠 EEG, HRV & Pose Data Generator")

scenario = st.selectbox("Select Scenario:", SCENARIOS, key="scenario_selector")
health_status = st.radio("Select Health Condition:", HEALTH_STATUSES, key="health_status_radio")
num_trials = st.number_input("Number of Trials:", min_value=1, max_value=100, value=5, key="num_trials_input")
generate_all = st.checkbox("Generate All Scenarios?", key="generate_all_checkbox")
duration = st.slider("Select Duration (seconds)", min_value=30, max_value=300, value=60, step=10, key="duration_slider")

if st.button("Generate Data", key="generate_data_button"):
    st.write("⏳ Generating datasets... Please wait.")
    
    session_id = f"Session_{int(time.time())}"
    session_dir = os.path.join(DATA_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    selected_scenarios = SCENARIOS if generate_all else [scenario]
    generate_data_batch(session_dir, selected_scenarios, [health_status], num_trials, duration)

    zip_path = create_zip_folder(session_dir)
    st.success(f"✅ Data successfully generated and saved in `{session_dir}`")

    with open(zip_path, "rb") as f:
        st.download_button(label="Download All Files (ZIP)", 
                           data=f.read(), 
                           file_name=f"{session_id}.zip", 
                           key="download_all")
import streamlit as st
import os
import json
import time
import numpy as np
import pandas as pd
import threading
import queue
from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import mne
import neurokit2 as nk
from scipy.signal import welch

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        .big-title {
            font-size: 36px !important;
            font-weight: bold;
            text-align: center;
            margin-bottom: 0px;
        }
        
        .small-title {
            font-size: 25px !important;
            font-weight: bold;
            text-align: center;
            margin-bottom: 40px;
            margin-top: -10px;
            color: #888;
        }
        
        .section {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        
        .section-title {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }

        .annotation-box {
            background: #fff3cd;
            padding: 10px;
            border-left: 5px solid #ffc107;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 16px;
        }
        
        .important-text {
            font-weight: bold;
            color: #e63946;
        }

        button.st-emotion-cache-ksh8e7.e1d5ycv52 {
            background: #000;
            color: #fff;
        }
        
    </style>
    """,
    unsafe_allow_html=True
)

# Ensure dataset directory exists
DATA_DIR = "generated_sessions"
os.makedirs(DATA_DIR, exist_ok=True)


st.markdown('<div class="big-title">NeuroSyncAI - Synthetic Data Generation</div>', unsafe_allow_html=True)
st.markdown('<div class="small-title">Generate synchronized EEG, HRV, and Pose data for benchmarking.</div>', unsafe_allow_html=True)

### 🔹 SESSION HANDLING
st.markdown('<div class="section-title">📌 Session Management</div>', unsafe_allow_html=True)
session_choice = st.radio("Choose an option", ["Create New Session", "Continue Previous Session"])

st.markdown('<div class="section-title">🔹 Scenario & Health Selection</div>', unsafe_allow_html=True)
### 🔹 SCENARIO & HEALTH SELECTION
scenario = st.selectbox("Select Scenario", [
    "Baseline Resting", "Cognitive Load", "Stress Induction",
    "Motor Task", "Fatigue", "Dual Task"
])

health_status = st.radio("Select Patient Health Status", ["Healthy", "MCI"])

duration = st.slider("Select Duration (seconds)", min_value=30, max_value=300, value=60, step=10)

# Generate unique timestamp for this experiment
experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

### 🔹 SESSION HANDLING (Consolidated Logic)
session_id = None
session_dir = None
metadata = None

# Load existing sessions
existing_sessions = sorted(os.listdir(DATA_DIR), reverse=True)

if session_choice == "Continue Previous Session" and existing_sessions:
    selected_session = st.selectbox("Select Session to Continue", existing_sessions, key="continue_session")
    session_dir = os.path.join(DATA_DIR, selected_session)
    metadata_path = os.path.join(session_dir, "session_metadata.json")

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        session_id = metadata.get("session_id", selected_session)
        st.write(f"🔄 **Resuming session:** `{session_id}`")
    else:
        st.warning("⚠️ Metadata file missing. Some session details might not be available.")

# Generate unique timestamp for this experiment
experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# **Only append if the experiment hasn't been added yet**
new_experiment = {
    "experiment_id": experiment_timestamp,
    "scenario": scenario,
    "health_status": health_status,
    "duration": duration
}

# Sampling rates
SAMPLING_RATES = {"EEG": 256, "HRV": 250, "Pose": 30}  # Hz

# EEG Frequencies for different scenarios
EEG_FREQS = {
    "Baseline Resting": [8, 10],  
    "Cognitive Load": [12, 18],    
    "Stress Induction": [20, 30],  
    "Fatigue State": [4, 7]        
}

def generate_annotations(output_dir, duration, eeg_df, hrv_df):
    annotations = []

    # Compute HRV baseline statistics
    hrv_mean = np.mean(hrv_df["HRV"])
    hrv_std = np.std(hrv_df["HRV"])
    
    # Define significant HRV anomaly threshold (mean ± 2 std deviations)
    upper_threshold = hrv_mean + 2 * hrv_std
    lower_threshold = hrv_mean - 2 * hrv_std

    # Identify HRV spikes that exceed this range
    hrv_anomalies = []
    start_time = None

    for i in range(len(hrv_df)):
        if hrv_df["HRV"].iloc[i] > upper_threshold or hrv_df["HRV"].iloc[i] < lower_threshold:
            if start_time is None:
                start_time = round(hrv_df["Timestamp"].iloc[i], 2)  # Mark anomaly start
        else:
            if start_time is not None:
                end_time = round(hrv_df["Timestamp"].iloc[i - 1], 2)  # Mark anomaly end
                if end_time - start_time > 5:  # Only record anomalies lasting >5 seconds
                    hrv_anomalies.append((start_time, end_time))
                start_time = None  # Reset

    # **EEG: Detect Low Theta Episodes**
    eeg_anomalies = []
    eeg_threshold = 0.15  # Threshold for low theta activity
    for col in ["F3", "F7"]:
        if col in eeg_df.columns:
            low_theta_mask = eeg_df[col] < eeg_threshold
            start_idx = None
            for i in range(len(low_theta_mask)):
                if low_theta_mask.iloc[i]:
                    if start_idx is None:
                        start_idx = i
                else:
                    if start_idx is not None:
                        start_time = round(eeg_df["Timestamp"].iloc[start_idx], 2)
                        end_time = round(eeg_df["Timestamp"].iloc[i - 1], 2)
                        if end_time - start_time > 5:
                            eeg_anomalies.append((start_time, end_time, col))
                        start_idx = None

    print("EEG Data Summary:")
    print(eeg_df.describe())  # This should show a range of values

    # **Write Annotations to File**
    annotation_file = os.path.join(output_dir, f"annotations_{experiment_timestamp}.txt")
    with open(annotation_file, "w") as f:
        for start, end in hrv_anomalies:
            f.write(f"{start} - {end}: Sudden HRV elevation detected\n")
        for start, end, col in eeg_anomalies:
            f.write(f"{start} - {end}: Low theta activity in EEG channel {col}\n")

    print("✅ Updated annotation file with realistic anomaly detection.")

EEG_REGION_MAPPING = {
    "Fp1": "Prefrontal Cortex (Left)",
    "Fp2": "Prefrontal Cortex (Right)",
    "AF3": "Frontal Cortex (Left)",
    "AF4": "Frontal Cortex (Right)",
    "F3": "Frontal Cortex (Left)",
    "F4": "Frontal Cortex (Right)",
    "F7": "Frontal Cortex (Left - Temporal Edge)",
    "F8": "Frontal Cortex (Right - Temporal Edge)",
    "C3": "Central Cortex (Left - Motor)",
    "C4": "Central Cortex (Right - Motor)",
    "P3": "Parietal Cortex (Left - Sensory Processing)",
    "P4": "Parietal Cortex (Right - Sensory Processing)",
    "O1": "Occipital Cortex (Left - Visual Processing)",
    "O2": "Occipital Cortex (Right - Visual Processing)"
}

def generate_unique_filename(prefix, scenario, health_status, extension="csv"):
    """Ensure unique filenames using timestamps with microseconds and trial number."""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{scenario}_{health_status}_trial1_{timestamp_str}.{extension}"

def save_brain_mapping(output_dir):
    mapping_file = os.path.join(output_dir, "brain_mapping.json")
    with open(mapping_file, "w") as f:
        json.dump(EEG_REGION_MAPPING, f, indent=4)
    print("✅ Brain region mapping file generated.")

def generate_insights(output_dir, scenario, health_status, eeg_df, hrv_df, pose_df):
    # Group EEG signals by brain region
    eeg_summary = {region: [] for region in set(EEG_REGION_MAPPING.values())}
    
    for channel, region in EEG_REGION_MAPPING.items():
        if channel in eeg_df.columns:
            eeg_summary[region].append(eeg_df[channel].mean())

    # Compute averages for each region
    eeg_summary = {region: np.mean(values) for region, values in eeg_summary.items() if values}

    insights = {
        "session_metadata": {
            "scenario": scenario,
            "health_status": health_status
        },
        "eeg_summary": eeg_summary,
        "hrv_summary": {
            "average_HRV": np.mean(hrv_df["HRV"]),
            "max_HRV": np.max(hrv_df["HRV"]),
            "min_HRV": np.min(hrv_df["HRV"])
        },
        "pose_summary": {
            "most_active_joint": pose_df.iloc[:, 1:].std().idxmax(),
            "least_active_joint": pose_df.iloc[:, 1:].std().idxmin()
        }
    }

    # Save as JSON
    insights_file = os.path.join(output_dir, f"insights_{experiment_timestamp}.json")
    with open(insights_file, "w") as f:
        json.dump(insights, f, indent=4)
    print("✅ Insights file generated.") 

def generate_eeg_mne(output_dir, scenario, health_status, duration, data_queue):
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
        if scenario == "Task Motor":
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
        elif scenario == "Task Motor":
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
    eeg_filename = f"eeg_data_{scenario}_{health_status}_{experiment_timestamp}.csv"

    # Save EEG data to CSV in correct format
    eeg_df = pd.DataFrame(base_eeg.T, columns=ch_names)
    eeg_df.insert(0, "Timestamp", np.linspace(0, duration, num_samples))
    eeg_df.to_csv(os.path.join(output_dir, eeg_filename), index=False)

    # Put EEG data into the queue with the expected key format
    data_queue.put(("eeg", eeg_df))

    print("✅ EEG Data (MNE) Generated Successfully with Dual Task for Healthy and MCI")
   
def generate_hrv(output_dir, scenario, health_status, duration, data_queue):
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

            if scenario == "Task Motor":
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

        hrv_filename = f"hrv_data_{scenario}_{health_status}_{experiment_timestamp}.csv"
        hrv_df.to_csv(os.path.join(output_dir, hrv_filename), index=False)

        # Send HRV data to queue
        data_queue.put(("hrv", hrv_df))
        print("✅ HRV Data Successfully Saved & Sent to Queue.")

    except Exception as e:
        print(f"❌ Error in HRV Generation: {e}")
        data_queue.put(("hrv", pd.DataFrame(columns=["Timestamp", "HRV"])))  # Send empty DataFrame instead of None

# Define the sampling rate
SAMPLING_RATES = {"Pose": 30}  # Hz

# Joint groups
UPPER_BODY_JOINTS = ["Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
                      "Left Wrist", "Right Wrist"]
LOWER_BODY_JOINTS = ["Left Hip", "Right Hip", "Left Knee", "Right Knee",
                      "Left Ankle", "Right Ankle"]

# Walking Pose Data Generator with All Scenarios (Now Includes Motor Task)
def generate_walking_pose(output_dir, scenario, health_status, duration, data_queue):
    """
    Generates synthetic pose data simulating a walking cycle with all implemented scenarios.
    """

    num_samples = duration * SAMPLING_RATES["Pose"]
    timestamps = np.linspace(0, duration, num_samples)

    pose_data = {"Timestamp": timestamps}

    # Base walking parameters
    step_frequency = 1.2 if health_status == "Healthy" else 0.8  # MCI walks slower
    stride_amplitude = 35 if health_status == "Healthy" else 20  # MCI has shorter steps
    arm_swing_amplitude = 10 if health_status == "Healthy" else 5  # MCI has reduced arm movement
    noise_level = 1.5 if health_status == "Healthy" else 3  # MCI has more instability

    # Default values
    upper_body_variation = 1.0  
    step_variation = 1.0  
    instability_factor = 1.0
    phase_offset = 0  

    # Scenario-specific adjustments
    if scenario == "Baseline Resting":
        step_variation = 1.0
        instability_factor = 0.5  # Minimal instability

    elif scenario == "Cognitive Load":
        step_variation = 0.9
        instability_factor = 1.5 if health_status == "Healthy" else 3

        # Phase shifts for MCI
        if health_status == "MCI":
            phase_shift_prob = 0.2
            apply_phase_shift = np.random.rand(num_samples) < phase_shift_prob
            step_delays = np.where(apply_phase_shift, np.pi / 8, 0)
        else:
            step_delays = np.zeros(num_samples)

        upper_body_variation = 1.1 if health_status == "Healthy" else 1.5

        movement = (
            (arm_swing_amplitude * step_variation * upper_body_variation)
            * np.sin(2 * np.pi * step_frequency * timestamps + phase_offset + step_delays)
        )

    elif scenario == "Stress Induction":
        step_variation = 1.3 if health_status == "Healthy" else 1.0
        instability_factor = 3 if health_status == "Healthy" else 5

        # Apply stress-induced phase shifts
        step_phase_shifts = np.random.uniform(-np.pi / 6, np.pi / 6, size=num_samples)
        upper_body_variation = 1.2 if health_status == "Healthy" else 1.8
        jitter_spikes = (
            np.random.normal(0, instability_factor, num_samples)
            * np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])
        )

        movement = (
            (arm_swing_amplitude * step_variation * upper_body_variation)
            * np.sin(2 * np.pi * step_frequency * timestamps + phase_offset + step_phase_shifts)
            + jitter_spikes
        )

    elif scenario == "Motor Task":
        step_variation = 1.5 if health_status == "Healthy" else 1.2  
        instability_factor = 1 if health_status == "Healthy" else 2  
        stride_amplitude *= 1.3  
        arm_swing_amplitude *= 1.5  
        upper_body_variation = 1.0 if health_status == "Healthy" else 1.3  

        movement = (
            (arm_swing_amplitude * step_variation * upper_body_variation)
            * np.sin(2 * np.pi * step_frequency * timestamps + phase_offset)
            + np.random.normal(0, instability_factor, num_samples)
        )

    elif scenario == "Fatigue":
        step_variation = 1.0  
        instability_factor = 2 if health_status == "Healthy" else 4  
        fatigue_decay = np.exp(-timestamps / duration)  

        stride_amplitude *= fatigue_decay
        arm_swing_amplitude *= fatigue_decay
        upper_body_variation = 1.1 if health_status == "Healthy" else 1.5  

        movement = (
            (arm_swing_amplitude * step_variation * upper_body_variation)
            * np.sin(2 * np.pi * step_frequency * timestamps + phase_offset)
            + np.random.normal(0, instability_factor, num_samples)
        ) * fatigue_decay

    elif scenario == "Dual Task":
        step_variation = 0.8 if health_status == "Healthy" else 0.6  
        instability_factor = 2 if health_status == "Healthy" else 4  
        reaction_delays = np.random.choice([0, np.pi / 12], size=num_samples, p=[0.85, 0.15])  
        upper_body_variation = 1.2 if health_status == "Healthy" else 1.6  

        movement = (
            (arm_swing_amplitude * step_variation * upper_body_variation)
            * np.sin(2 * np.pi * step_frequency * timestamps + phase_offset + reaction_delays)
            + np.random.normal(0, instability_factor, num_samples)
        )

    else:
        raise ValueError(f"Scenario '{scenario}' not implemented")

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

        pose_data[joint] = movement

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    pose_filename = generate_unique_filename("walking_pose", scenario, health_status)
    pose_df = pd.DataFrame(pose_data)
    pose_df.to_csv(os.path.join(output_dir, pose_filename), index=False)
    data_queue.put(("pose", pose_df))

    print(f"✅ {scenario} Pose Data Updated for {health_status} saved to {pose_filename}")

# Main function
def generate_data(output_dir, scenario, health_status, duration):
    data_queue = queue.Queue()
    results = {}

    threads = [
        threading.Thread(target=generate_eeg_mne, args=(output_dir, scenario, health_status, duration, data_queue)),
        threading.Thread(target=generate_hrv, args=(output_dir, scenario, health_status, duration, data_queue)),
        threading.Thread(target=generate_walking_pose, args=(output_dir, scenario, health_status, duration, data_queue))
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # Extract data from the queue
    while not data_queue.empty():
        key, value = data_queue.get()
        standardized_key = key.lower().split()[0]  # Convert to lowercase and standardize
        results[standardized_key] = value

    # **Ensure that "eeg" key exists before proceeding**
    if "eeg" not in results:
        st.error("⚠️ Error: EEG data was not generated correctly!")
        print("ERROR: EEG data missing from results:", results.keys())  # Debugging log
        return {}  # Return an empty dictionary to prevent crashes

    if "hrv" not in results:
        st.error("⚠️ Error: HRV data was not generated correctly!")
        print("ERROR: HRV data missing from results:", results.keys())
        return {}

    if "pose" not in results:
        st.error("⚠️ Error: Pose data was not generated correctly!")
        print("ERROR: Pose data missing from results:", results.keys())
        return {}

    # Generate insights & annotations
    generate_annotations(output_dir, duration, results["eeg"], results["hrv"])
    generate_insights(output_dir, scenario, health_status, results["eeg"], results["hrv"], results["pose"])

    return results

### 🔹 START DATA GENERATION BUTTON ###
if st.button("Start Data Generation", key="start_experiment"):
    # **Step 1: If "Create New Session", Generate a New Session Folder**
    if session_choice == "Create New Session":
        session_id = f"Session_{int(time.time())}"
        session_dir = os.path.join(DATA_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        st.write(f"🆕 **New session created:** `{session_id}`")

        # Initialize new session metadata
        metadata = {
            "session_id": session_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "experiments": []
        }

    # **Step 2: Ensure Session Metadata Exists**
    metadata_path = os.path.join(session_dir, "session_metadata.json")
    if not metadata:
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {"session_id": session_id, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "experiments": []}

    # **Step 3: Generate a Unique Experiment ID**
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_experiment = {
        "experiment_id": experiment_timestamp,
        "scenario": scenario,
        "health_status": health_status,
        "duration": duration
    }

    # **Step 4: Prevent Duplicates Before Appending**
    existing_experiment_ids = {exp["experiment_id"] for exp in metadata["experiments"]}
    if experiment_timestamp not in existing_experiment_ids:
        metadata["experiments"].append(new_experiment)
        st.write(f"📌 **New experiment added:** `{experiment_timestamp}`")

        # **Step 5: Save Updated Metadata**
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
    else:
        st.warning("⚠️ Experiment already logged. Skipping duplicate entry.")

    # **Step 6: Proceed with Data Generation**
    st.write(f"🟢 **Generating data for session `{session_id}`...**")
    results = generate_data(session_dir, scenario, health_status, duration)

    st.success(f"✅ Data saved under `{session_dir}`")

# Load Data Function
def load_generated_data(session_dir):
    """Loads EEG, HRV, Pose, and annotations from generated session."""
    data = {}
    try:
        data["eeg"] = pd.read_csv(os.path.join(session_dir, "eeg_data.csv"))
        data["hrv"] = pd.read_csv(os.path.join(session_dir, "hrv_data.csv"))
        data["pose"] = pd.read_csv(os.path.join(session_dir, "pose_keypoints.csv"))
        with open(os.path.join(session_dir, "annotations.txt"), "r") as f:
            data["annotations"] = f.readlines()
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        return None
    return data

# Function to display annotations
def display_annotations(annotations):
    st.subheader("📌 Annotations")
    for annotation in annotations:
        st.write(f"- {annotation.strip()}")

st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

### 🔹 INSPECT GENERATED DATA ###
st.subheader("🔬 Inspect EEG, HRV, and Pose Data from the Generated Session.")

if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
    sessions = sorted(os.listdir(DATA_DIR), reverse=True)
    selected_inspection_session = st.selectbox("Select Session for Inspection", sessions, key="inspect_session")

    session_dir = os.path.join(DATA_DIR, selected_inspection_session)

    # **Step 1: List Available Data Files**
    eeg_files = sorted([f for f in os.listdir(session_dir) if f.startswith("eeg_data_") and f.endswith(".csv")], reverse=True)
    hrv_files = sorted([f for f in os.listdir(session_dir) if f.startswith("hrv_data_") and f.endswith(".csv")], reverse=True)
    pose_files = sorted([f for f in os.listdir(session_dir) if f.startswith("walking_pose_") and f.endswith(".csv")], reverse=True)

    if eeg_files and hrv_files and pose_files:
        # **Step 2: Load Latest Experiment Files**
        latest_eeg_file = eeg_files[0]
        latest_hrv_file = hrv_files[0]
        latest_pose_file = pose_files[0]

        eeg_path = os.path.join(session_dir, latest_eeg_file)
        hrv_path = os.path.join(session_dir, latest_hrv_file)
        pose_path = os.path.join(session_dir, latest_pose_file)

        st.write(f"📂 **Loading latest experiment files:**")
        st.write(f"EEG: `{latest_eeg_file}`")
        st.write(f"HRV: `{latest_hrv_file}`")
        st.write(f"Pose: `{latest_pose_file}`")

        # **Step 3: Read Data**
        try:
            eeg_df = pd.read_csv(eeg_path)
            hrv_df = pd.read_csv(hrv_path)
            pose_df = pd.read_csv(pose_path)

            ### **🔹 PLOTTING FUNCTIONS UPDATED TO HANDLE NEW DATA FORMAT ###

            def plot_eeg(data):
                fig = go.Figure()
                
                channel_offset = 10  # Offset to separate channels
                base_offset = 0  # Initial offset
                time_window = 10  # Default zoom window (first 10 seconds)
                
                num_channels = len(data.columns) - 1  # Exclude "Timestamp"
                plot_height = 300 + (num_channels * 30)  # Scale height based on channel count

                for col in data.columns[1:]:  # Skip timestamp column
                    fig.add_trace(go.Scatter(
                        x=data["Timestamp"], 
                        y=data[col] + base_offset,  # Apply vertical offset
                        mode="lines", 
                        name=col
                    ))
                    base_offset += channel_offset  # Increment offset for next channel

                fig.update_layout(
                    title="EEG Data (Scrollable View)", 
                    xaxis_title="Time (s)", 
                    yaxis_title="EEG Amplitude (Offset Applied)",
                    showlegend=True,
                    height=plot_height,  # **Dynamically adjust height**
                    xaxis=dict(
                        rangeslider=dict(visible=True),  # Enable scrolling
                        range=[0, time_window]  # Show first 10 sec initially
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            def plot_hrv(data):
                fig = px.line(data, x="Timestamp", y="HRV", title="Heart Rate Variability (HRV)")
                fig.update_xaxes(title="Time (s)")
                fig.update_yaxes(title="HRV Value")
                st.plotly_chart(fig, use_container_width=True)

            def plot_pose(data):
                fig = go.Figure()
                
                joints = list(data.columns[1:])  # Skip timestamp
                joint_offset = 5  # Offset to separate joints
                base_offset = 0  # Initial offset
                time_window = 10  # Default zoom window (first 10 seconds)

                for col in joints:
                    fig.add_trace(go.Scatter(
                        x=data["Timestamp"], 
                        y=data[col] + base_offset,  # Apply vertical offset
                        mode="lines", 
                        name=col
                    ))
                    base_offset += joint_offset  # Increment offset for next joint

                fig.update_layout(
                    title="Pose Keypoints (Scrollable View)", 
                    xaxis_title="Time (s)", 
                    yaxis_title="Joint Values (Offset Applied)",
                    showlegend=True,
                    xaxis=dict(
                        rangeslider=dict(visible=True),  # Enable scrolling
                        range=[0, time_window]  # Show first 10 sec initially
                    )
                )
                st.plotly_chart(fig, use_container_width=True)



            # **Step 4: Display & Plot Data**
            st.subheader("📊 Generated EEG Data")
            st.dataframe(eeg_df.head())
            plot_eeg(eeg_df)

            st.subheader("🦾 Generated HRV Data")
            st.dataframe(hrv_df.head())
            plot_hrv(hrv_df)

            st.subheader("🏥 Generated Pose Data")
            st.dataframe(pose_df.head())
            plot_pose(pose_df)

        except Exception as e:
            st.error(f"⚠️ Error loading data: {e}")

    else:
        st.warning("⚠️ No experiment data found for this session. Generate data first!")
else:
    st.warning("No generated session found! Run the data generator first.")
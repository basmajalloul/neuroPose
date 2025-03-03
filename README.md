# **NeuroSyncAI - Synthetic Data Generator**

## **Overview**
NeuroSyncAI is a synthetic data generation tool for producing synchronized **EEG (Electroencephalography), HRV (Heart Rate Variability), and Pose** data. This dataset is designed for benchmarking and validating machine learning models in cognitive and motor function assessment.

The system generates realistic physiological and movement data under different cognitive and physical conditions, making it suitable for research in **Mild Cognitive Impairment (MCI)** detection, stress analysis, fatigue monitoring, and more.

---

## **Features**
- **Synthetic Data Generation**: EEG, HRV, and Pose data for six scenarios:
  - **Baseline Resting**
  - **Cognitive Load**
  - **Stress Induction**
  - **Motor Task**
  - **Fatigue**
  - **Dual Task**
- **Multi-Scenario & Health Status**: Data can be generated for **Healthy** or **MCI** participants.
- **Realistic EEG Frequency Modulation**: EEG signals mimic real-world characteristics based on different scenarios.
- **HRV Anomaly Detection**: HRV features such as **SDNN, RMSSD, and LF/HF ratio** are computed and adjusted for each scenario.
- **Pose Data Simulation**: Pose data includes **upper-body and lower-body joint movements**, with variations based on cognitive load and fatigue.
- **Batch Processing**: Supports **multi-threaded data generation** for large-scale dataset creation.

---

## **Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/NeuroSyncAI.git
cd NeuroSyncAI
```

### **2. Install Dependencies**
Ensure you have Python **>=3.8** and install the required packages:
```bash
pip install -r requirements.txt
```

### **3. Run the Data Generator**
To generate a **single session**, run:
```bash
streamlit run data_generator.py
```
To generate **batch sessions**, use:
```bash
streamlit run data_generator_batch.py
```

---

## **Directory Structure**
```
📂 NeuroSyncAI
 ├── 📂 generated_sessions        # Contains all generated session data
 │   ├── 📂 Session_<timestamp>/   # Each session gets a unique timestamp
 │   │   ├── eeg_data_<scenario>_<health>.csv
 │   │   ├── hrv_data_<scenario>_<health>.csv
 │   │   ├── walking_pose_<scenario>_<health>.csv
 │   │   ├── session_metadata.json
 │   │   ├── insights_<timestamp>.json
 │   │   ├── annotations_<timestamp>.txt
 ├── data_generator.py            # Single-session data generation (GUI)
 ├── data_generator_batch.py       # Batch data generation (multi-threaded)
 ├── README.md                     # This file
 ├── requirements.txt               # Required Python libraries
```

---

## **Generated Data Format**
Each generated session consists of:
- **EEG Data (`eeg_data_*.csv`)**
  - Timestamp, Fp1, Fp2, AF3, AF4, F3, F4, F7, F8, C3, C4, P3, P4, O1, O2
  - Frequencies modulated based on task difficulty and cognitive load.
  
- **HRV Data (`hrv_data_*.csv`)**
  - Timestamp, HRV values with scenario-based variability.

- **Pose Data (`walking_pose_*.csv`)**
  - Timestamp, Joint movement values (Shoulders, Elbows, Wrists, Hips, Knees, Ankles).

- **Annotations (`annotations_*.txt`)**
  - Detected anomalies in EEG, HRV, or Pose data.

- **Session Metadata (`session_metadata.json`)**
  - Experiment ID, scenario, health status, and duration.

- **Insights (`insights_*.json`)**
  - Statistical summary of EEG, HRV, and Pose data.

---

## **Usage**
### **Generating New Data**
1. Launch the **Streamlit** app:
   ```bash
   streamlit run data_generator.py
   ```
2. Select **Create New Session**.
3. Choose a scenario and participant health status.
4. Click **Start Data Generation**.

### **Batch Processing**
For multiple experiments:
```bash
streamlit run data_generator_batch.py
```
This will generate multiple trials across different scenarios.

---

## **Future Enhancements**
- Add **real-time EEG signal visualization**.
- Integrate **deep learning models** for anomaly detection.
- Expand dataset **to include real-world recorded data for comparison**.

---

## **License**
This project is licensed under the **MIT License**.

---

## **Contributors**
- **Basma Jalloul**  
  PhD Researcher in Computer Vision & Signal Processing  
  [GitHub Profile](https://github.com/your-username)

---

This **README** provides a structured and professional overview of the **NeuroSyncAI** repository, making it easy for others to use and contribute. 🚀

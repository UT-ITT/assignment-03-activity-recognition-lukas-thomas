# preprocessing.py
# this programm handles preprocessing (loading and feature extraction)

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import skew


DATA_DIR = Path("data")



# ---------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------


def extract_features(df):
    """
    Extract features from one recording.

    Parameters
    ----------
    df : pandas.DataFrame
        One CSV recording.

    Returns
    -------
    dict
        Dictionary containing extracted features.
    """
    features = {}
    fs = 100  # Required 100Hz sampling rate [cite: 16]
    df = df.fillna(0)

    
    # 1. Magnitudes (The most robust signals)
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['gyro_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)

    # 2. Strategic Loop
    raw_axes = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    magnitudes = ["acc_mag", "gyro_mag"]


    # Simple stats for raw axes (captures general orientation/pose)
    for col in raw_axes:
        values = df[col]
        features[f"{col}_mean"] = np.mean(values)
        features[f"{col}_std"] = np.std(values)
        features[f"{col}_max"] = np.max(values)
        features[f"{col}_iqr"] = np.percentile(values, 75) - np.percentile(values, 25)
        features[f"{col}_skew"] = skew(values)

    
        

    # Richer features for magnitudes 
    for col in magnitudes:
        values = df[col]
        features[f"{col}_mean"] = np.mean(values)
        features[f"{col}_max"] = np.max(values)
        features[f"{col}_std"] = np.std(values)
    

        rms = np.sqrt(np.mean(values**2))

        features[f"{col}_crest_factor"] = (
            np.max(np.abs(values)) / (rms + 1e-9)
        )

         # autocorr
        for lag in [20, 50, 100]:
            corr_value = df[col].corr(df[col].shift(lag))
            if not np.isnan(corr_value):
                features[f"{col}_autocorr_{lag}"] = corr_value
            else:
                features[f"{col}_autocorr_{lag}"] = 0.0
        
        
        # --- FFT only on Magnitudes ---
        
        fft_values = np.abs(np.fft.rfft(values))
        freqs = np.fft.rfftfreq(len(values), d=1/fs)
        
        # We only need the dominant frequency and the total energy
        power = fft_values[1:]
        top_indices = np.argsort(power)[-3:][::-1] + 1

        features[f"{col}_dom_freq"] = freqs[top_indices[0]]
        features[f"{col}_2dom_freq"] = freqs[top_indices[1]] if len(top_indices) > 1 else 0.0
        features[f"{col}_3dom_freq"] = freqs[top_indices[2]] if len(top_indices) > 2 else 0.0
        features[f"{col}_energy"] = np.sum(fft_values**2) / len(values)
        
    features['gyro_acc_energy_ratio'] = features['gyro_mag_energy'] / (features['acc_mag_energy'] + 1e-9)


    # FINAL SAFETY SWEEP: If ANY feature is still NaN, force it to 0.0
    for key in features:
        if np.isnan(features[key]) or np.isinf(features[key]):
            features[key] = 0.0
    

    return features
    


# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------


def get_activity_from_filename(filename):
    """
    Example filename:
    susi-running-1.csv

    Returns:
    running
    """

    stem = Path(filename).stem
    parts = stem.split("-")

    # assumes structure: name-activity-number.csv
    return parts[1]

def get_person_from_filename(filename):
    """
    Example filename:
    susi-running-1.csv

    Returns:
    susi
    """

    stem = Path(filename).stem
    parts = stem.split("-")

    # assumes structure: name-activity-number.csv
    return parts[0]

def segment_data(df, window_size=100, step_size=50):
    """
    Splits a DataFrame into segments of a fixed size.
    
    Parameters
    ----------
    df : pd.DataFrame
        The full recording.
    window_size : int
        Number of samples per window (e.g., 100 for 1 second at 100Hz).
    step_size : int
        Number of samples to slide the window. 50 = 50% overlap.
        
    Yields
    ------
    pd.DataFrame
        A segment of the original data.
    """
    for start in range(0, len(df) - window_size + 1, step_size):
        yield df.iloc[start : start + window_size]


# ---------------------------------------------------
# MAIN DATASET CREATION
# ---------------------------------------------------

def create_feature_dataset(data_dir, windowing = False, window_size=100, step_size=50):
    rows = []
    csv_files = list(Path(data_dir).glob("*.csv"))

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        
        if windowing:
            # Segment the data into windows
            for segment in segment_data(df, window_size, step_size):
                # Extract features from the segment instead of the whole df
                features = extract_features(segment)

                # Add labels and metadata
                features["activity"] = get_activity_from_filename(file_path.name)
                features["person"] = get_person_from_filename(file_path.name)
                
                rows.append(features)
        else:
            # Extract features from the whole recording
            features = extract_features(df)

            # Add labels and metadata
            features["activity"] = get_activity_from_filename(file_path.name)
            features["person"] = get_person_from_filename(file_path.name)
            
            rows.append(features)

    feature_df = pd.DataFrame(rows)
    return feature_df

# ---------------------------------------------------
# RUN SCRIPT
# ---------------------------------------------------

if __name__ == "__main__":

    feature_df = create_feature_dataset(DATA_DIR)

    print("\nFeature DataFrame:")
    print(feature_df.head())

    # optional: save processed dataset
    feature_df.to_csv("features.csv", index=False)

    print("\nSaved features.csv")
# preprocessing.py
# this programm handles preprocessing (loading and feature extraction)

from pathlib import Path
import pandas as pd
import numpy as np


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

    sensor_columns = [
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z"
    ]

    # ---------------------------------------------
    # Basic statistical features
    # ---------------------------------------------
    for col in sensor_columns:

        values = df[col].fillna(0) # Replace NaNs with 0

        features[f"{col}_mean"] = values.mean()
        features[f"{col}_std"] = values.std()

        features[f"{col}_min"] = values.min()
        features[f"{col}_max"] = values.max()

        features[f"{col}_median"] = values.median()   



        # # 3. FFT (Check if signal is not just constant zeros)
        # if values.std() == 0:
        #     features[f"{col}_dom_freq"] = 0
        #     features[f"{col}_dom_mag"] = 0
        #     features[f"{col}_spectral_energy"] = 0
        # else:
        #     fft_vals = np.fft.rfft(values)
        #     fft_freq = np.fft.rfftfreq(len(values), d=1/100)
        #     magnitudes = np.abs(fft_vals) / len(values)
            
        #     # Find dominant freq (ignore DC)
        #     dominant_idx = np.argmax(magnitudes[1:]) + 1
        #     features[f"{col}_dom_freq"] = fft_freq[dominant_idx]
        #     features[f"{col}_dom_mag"] = magnitudes[dominant_idx]
        #     features[f"{col}_spectral_energy"] = np.sum(magnitudes**2)


    # ---------------------------------------------
    # Example combined acceleration magnitude
    # ---------------------------------------------
    # acc_magnitude = np.sqrt(
    #     df["acc_x"]**2 +
    #     df["acc_y"]**2 +
    #     df["acc_z"]**2
    # )

    # features["acc_mag_mean"] = acc_magnitude.mean()
    # features["acc_mag_max"] = acc_magnitude.max()
    # features["acc_mag_std"] = acc_magnitude.std()
    
    # 2. Combined Magnitude (Orientation Invariant)
    # This prevents the model from failing if a person holds the sensor slightly differently
    acc_magnitude = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2).fillna(0)
    gyro_magnitude = np.sqrt(df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2).fillna(0)

    features["acc_mag_mean"] = acc_magnitude.mean()
    features["acc_mag_max"] = acc_magnitude.max()
    features["acc_mag_std"] = acc_magnitude.std()

    features["gyro_mag_mean"] = gyro_magnitude.mean()
    features["gyro_mag_max"] = gyro_magnitude.max()
    features["gyro_mag_std"] = gyro_magnitude.std()

    # 3. Stable Frequency Features: Energy Bands (Instead of Dominant Frequency)
    # We only apply FFT to the magnitudes to keep the feature count low and relevant
    # for name, mag_data in [("acc_mag", acc_magnitude), ("gyro_mag", gyro_magnitude)]:
    #     if mag_data.std() > 0:
    #         fft_vals = np.fft.rfft(mag_data)
    #         magnitudes = np.abs(fft_vals) / len(mag_data)
    #         fft_freq = np.fft.rfftfreq(len(mag_data), d=1/100) # Assuming 100Hz
            
    #         # Ignore DC component (0 Hz)
    #         freqs = fft_freq[1:]
    #         mags = magnitudes[1:]
            
    #         # Create masks for frequency bands
    #         # Adjust these ranges based on your specific activities
    #         low_band = (freqs >= 0.1) & (freqs <= 3.0)   # Slow movements, posture changes
    #         med_band = (freqs > 3.0) & (freqs <= 10.0)   # Active human movement (running/walking)
    #         high_band = (freqs > 10.0)                   # Jitters, impacts, fast transients
            
    #         # Calculate sum of energy in each band
    #         features[f"{name}_energy_low"] = np.sum(mags[low_band]**2) if np.any(low_band) else 0
    #         features[f"{name}_energy_med"] = np.sum(mags[med_band]**2) if np.any(med_band) else 0
    #         features[f"{name}_energy_high"] = np.sum(mags[high_band]**2) if np.any(high_band) else 0
    #     else:
    #         features[f"{name}_energy_low"] = 0
    #         features[f"{name}_energy_med"] = 0
    #         features[f"{name}_energy_high"] = 0
    

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


# ---------------------------------------------------
# MAIN DATASET CREATION
# ---------------------------------------------------

def create_feature_dataset(data_dir):
    """
    Creates one dataframe where:
    - each row = one recording
    - columns = extracted features + activity label
    """

    rows = []

    csv_files = list(Path(data_dir).glob("*.csv"))

    print(f"Found {len(csv_files)} CSV files")

    for file_path in csv_files:

        print(f"Processing: {file_path.name}")

        # load csv
        df = pd.read_csv(file_path)

        # extract features
        features = extract_features(df)

        # add label
        features["activity"] = get_activity_from_filename(file_path.name)

        # add person
        features["person"] = get_person_from_filename(file_path.name)

        rows.append(features)

    # create dataframe
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
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

        values = df[col]

        features[f"{col}_mean"] = values.mean()
        features[f"{col}_std"] = values.std()

        features[f"{col}_min"] = values.min()
        features[f"{col}_max"] = values.max()

        features[f"{col}_median"] = values.median()

    # ---------------------------------------------
    # Example combined acceleration magnitude
    # ---------------------------------------------
    acc_magnitude = np.sqrt(
        df["acc_x"]**2 +
        df["acc_y"]**2 +
        df["acc_z"]**2
    )

    features["acc_mag_mean"] = acc_magnitude.mean()
    features["acc_mag_max"] = acc_magnitude.max()
    features["acc_mag_std"] = acc_magnitude.std()

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
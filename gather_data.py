# this program gathers sensor data
import time
import pandas as pd
import os
from DIPPID import SensorUDP

# ----------------------------
# CONFIGURATION
# ----------------------------
PORT = 5700
RECORD_SECONDS = 10

# ----------------------------
# USER INPUT
# ----------------------------
NAME = input("Your name: ")
ACTION = input("Activity (running/rowing/lifting/jumpingjacks): ")
NUMBER = input("Recording number: ")

filename = f"data/{NAME}-{ACTION}-{NUMBER}.csv"

print(f"Output file: {filename}")

# ----------------------------
# CONNECT TO DIPPID
# ----------------------------
sensor = SensorUDP(PORT)

print("Waiting for DIPPID device...")

# wait until accelerometer data exists
while sensor.get_value("accelerometer") is None:
    time.sleep(0.1)

print("Connected!")
print("Press button_1 on the DIPPID device to start recording.")

# ----------------------------
# WAIT FOR BUTTON PRESS
# ----------------------------
while True:
    button = sensor.get_value("button_1")

    if button == 1:
        break

    time.sleep(0.01)

print("Recording started...")

# ----------------------------
# RECORD SENSOR DATA
# ----------------------------
data = []

start_time = time.time()

while time.time() - start_time < RECORD_SECONDS:

    timestamp = time.time()

    acc = sensor.get_value("accelerometer")
    gyro = sensor.get_value("gyroscope")

    # only save if both sensors exist
    if acc is not None and gyro is not None:
        print(acc)
        print(gyro)
        row = {
            "timestamp": timestamp,
            "acc_x": acc["x"],
            "acc_y": acc["y"],
            "acc_z": acc["z"],
            "gyro_x": gyro["x"],
            "gyro_y": gyro["y"],
            "gyro_z": gyro["z"]
        }

        data.append(row)

    # tiny delay to reduce CPU load
    time.sleep(0.001)

print("Recording finished.")

# ----------------------------
# CREATE DATAFRAME
# ----------------------------
df = pd.DataFrame(data)

# ----------------------------
# RESAMPLE TO 100 HZ
# ----------------------------

# convert timestamps to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# use timestamp as dataframe index
df.set_index('timestamp', inplace=True)

# 100 Hz = every 10 milliseconds
df_resampled = df.resample('10ms').mean()

# reset index
df_resampled.reset_index(inplace=True)

# convert timestamps back to milliseconds
df_resampled['timestamp'] = (
    (df_resampled['timestamp'] - pd.Timestamp("1970-01-01"))
    // pd.Timedelta('1ms')
)

# ----------------------------
# SAVE CSV
# ----------------------------

# use dataframe index as id column
df_resampled.index.name = 'id'

df_resampled.to_csv(filename, index=True)

print(f"Saved {len(df_resampled)} samples to {filename}")
print("Program finished.")
os._exit(0)
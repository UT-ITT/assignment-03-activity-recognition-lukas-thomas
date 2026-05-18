# this program visualizes activities with pyglet

import time
import activity_recognizer as activity
from preprocessing import extract_features
import pyglet
from DIPPID import SensorUDP
import pandas as pd
import os
import numpy as np
from collections import deque, Counter

# Windows settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BACKGROUND_COLOR = (0, 31, 63)

PORT = 5700

# Animation settings
ANIMATION_FRAME_RATE = 0.5  # seconds per frame
ANIMATION_POSITION = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 5)  # x, y position of the animation on the window

# Text settings
TEXT_COLOR = (255, 255, 255, 255)  # white
TEXT_FONT_SIZE = 24
TEXT_FONT_NAME = 'Arial'
ACTIVITY_TEXT_POSITION = (WINDOW_WIDTH // 2, WINDOW_HEIGHT - 100)
COUNTER_TEXT_POSITION = (WINDOW_WIDTH // 2, WINDOW_HEIGHT - 500)
FALSE_ACTIVITY_TEXT_POSITION = (WINDOW_WIDTH // 2, WINDOW_HEIGHT - 560)

# create sensor
sensor = SensorUDP(PORT)
data_buffer = deque(maxlen=100) # current data buffer
DATA_PATH = "data"

# train the classifier
clf = activity.train_classifier()
print("Classifier trained")

# game parameters
activities = ["jumpingjacks", "lifting", "rowing", "running"]
activity_name = "lifting"  # current activity
counter = 0 # counts how many times the same activity was predicted(done) in a row

# create window
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

# create text label
activity_text = pyglet.text.Label('Current Activity: ', font_name=TEXT_FONT_NAME, font_size=TEXT_FONT_SIZE, x=ACTIVITY_TEXT_POSITION[0], y=ACTIVITY_TEXT_POSITION[1], anchor_x='center', color=TEXT_COLOR)
counter_text = pyglet.text.Label('Counter: 0', font_name=TEXT_FONT_NAME, font_size=TEXT_FONT_SIZE, x=COUNTER_TEXT_POSITION[0], y=COUNTER_TEXT_POSITION[1], anchor_x='center', color=TEXT_COLOR)
false_activity_text = pyglet.text.Label('', font_name=TEXT_FONT_NAME, font_size=TEXT_FONT_SIZE, x=FALSE_ACTIVITY_TEXT_POSITION[0], y=FALSE_ACTIVITY_TEXT_POSITION[1], anchor_x='center', color=TEXT_COLOR)
# load images for activities
jumping_jacks_images = [pyglet.resource.image('img/jumpingjack_1.png'),
                         pyglet.resource.image('img/jumpingjack_2.png')]

lifting_images = [pyglet.resource.image('img/lifting_1.png'),
                   pyglet.resource.image('img/lifting_2.png')]

rowing_images = [pyglet.resource.image('img/rowing_1.png'),
                 pyglet.resource.image('img/rowing_2.png')]

running_images = [pyglet.resource.image('img/running_1.png'),
                  pyglet.resource.image('img/running_2.png')]

# create animations for activities
jumping_jacks_animation = pyglet.image.Animation.from_image_sequence(jumping_jacks_images, ANIMATION_FRAME_RATE, loop=True)
lifting_animation = pyglet.image.Animation.from_image_sequence(lifting_images, ANIMATION_FRAME_RATE, loop=True)
rowing_animation = pyglet.image.Animation.from_image_sequence(rowing_images, ANIMATION_FRAME_RATE, loop=True)
running_animation = pyglet.image.Animation.from_image_sequence(running_images, ANIMATION_FRAME_RATE, loop=True)

# create sprites for each animation
jumping_jacks_sprite = pyglet.sprite.Sprite(jumping_jacks_animation, ANIMATION_POSITION[0], ANIMATION_POSITION[1])
lifting_sprite = pyglet.sprite.Sprite(lifting_animation, ANIMATION_POSITION[0], ANIMATION_POSITION[1])
rowing_sprite = pyglet.sprite.Sprite(rowing_animation, ANIMATION_POSITION[0], ANIMATION_POSITION[1])
running_sprite = pyglet.sprite.Sprite(running_animation, ANIMATION_POSITION[0], ANIMATION_POSITION[1])

# Scale sprites to fit the window
TARGET_WIDTH = int(WINDOW_WIDTH * 0.5)
TARGET_HEIGHT = int(WINDOW_HEIGHT * 0.5)

def fit_sprite(sprite):
    scale = min(TARGET_WIDTH / sprite.width, TARGET_HEIGHT / sprite.height)
    sprite.update(scale=scale)
    sprite.update(
        x=(WINDOW_WIDTH - sprite.width) // 2,
        y=(WINDOW_HEIGHT - sprite.height) // 2,
    )

fit_sprite(jumping_jacks_sprite)
fit_sprite(lifting_sprite)
fit_sprite(rowing_sprite)
fit_sprite(running_sprite)

# choose activity to be done
def choose_activity(dt):
    global activity_name
    activity_name = np.random.choice([a for a in activities if a != activity_name])
 
# extract features 
def transform_data_for_model(df):
    rows = []
    features = extract_features(df)
    rows.append(features)
    feature_df = pd.DataFrame(rows)
    return feature_df

# collect data from dippid device
def collect(dt):
    acc = sensor.get_value("accelerometer"); gyro = sensor.get_value("gyroscope")
    if acc and gyro:
        row = {
            "timestamp": time.perf_counter(),
            "acc_x": acc["x"],
            "acc_y": acc["y"],
            "acc_z": acc["z"],
            "gyro_x": gyro["x"],
            "gyro_y": gyro["y"],
            "gyro_z": gyro["z"]
        }

        data_buffer.append(row)

# predict activity based on collected data
def predict(dt):
    global activity_name, counter
    # Only predict if we have enough data for one window
    if len(data_buffer) < 100:
        return
    # Create a DataFrame from the current buffer and extract features
    window_df = pd.DataFrame(list(data_buffer))[-100:].reset_index(drop=True)
    feats = transform_data_for_model(window_df)
    feats = feats.reindex(columns=list(clf.feature_columns)).fillna(0).astype(float) # Ensure correct feature order and handle missing features
    # predict activity
    pred = clf.predict(feats)[0]
    activity_prediction = clf.label_encoder.inverse_transform([pred])[0]

    # control the game logic based on the predicted activity
    if activity_name == activity_prediction:
        counter += 1
        false_activity_text.text = ""
    else:
        false_activity_text.text = f"Keep going!"
    if counter >= 10: # if the same activity was predicted 10 times in a row, choose a new one
        counter = 0
        choose_activity(0)
        counter_text.text = f"Counter: {counter}"

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.Q:   # Press Q to quit the game
        os._exit(0)

@window.event
def on_draw():
    global clf
    pyglet.gl.glClearColor(BACKGROUND_COLOR[0]/255, BACKGROUND_COLOR[1]/255, BACKGROUND_COLOR[2]/255, 1)
    window.clear()
    
    if activity_name == "jumpingjacks":
        jumping_jacks_sprite.draw()
    elif activity_name == "lifting":
        lifting_sprite.draw()
    elif activity_name == "rowing":
        rowing_sprite.draw()
    elif activity_name == "running":
        running_sprite.draw()
    
    activity_text.text = f"Give me 10 seconds of {activity_name.capitalize()}!"
    activity_text.draw()
    if counter > 0  and counter < 5:
        counter_text.text = f"You already did {counter} seconds of {activity_name.capitalize()}!"
    elif counter >= 5 and counter < 8:
        counter_text.text = f"You already did {counter} seconds of {activity_name.capitalize()}! Great!"
    elif counter >= 8:
        counter_text.text = f"You already did {counter} seconds of {activity_name.capitalize()}! Almost there! "
    counter_text.draw()
    false_activity_text.draw()

pyglet.clock.schedule_interval(collect, 0.01)
pyglet.clock.schedule_interval(predict, 1.0)
pyglet.app.run()

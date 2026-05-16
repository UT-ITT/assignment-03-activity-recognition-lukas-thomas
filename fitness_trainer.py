# this program visualizes activities with pyglet

import time
import activity_recognizer as activity
from preprocessing import extract_features
import pyglet
from DIPPID import SensorUDP
import pandas as pd
import os

# Todos
# Read in data from dippid device
# Transform data with functions for the model
# Run model to get activity prediction
# Scale all images to same size

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
TEXT_POSITION = (WINDOW_WIDTH // 2, WINDOW_HEIGHT - 100)
TEXT_FONT_SIZE = 24
TEXT_FONT_NAME = 'Arial'

# create sensor
sensor = SensorUDP(PORT)
data = [] # current data buffer
DATA_PATH = "data"

# game parameters
activity_name = "lifting"  # current activity

# create window
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

# create text label
text = pyglet.text.Label('Current Activity: ', font_name=TEXT_FONT_NAME, font_size=TEXT_FONT_SIZE, x=TEXT_POSITION[0], y=TEXT_POSITION[1], anchor_x='center', color=TEXT_COLOR)

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
jumping_jacks_sprite.update(scale=0.3)
lifting_sprite.update(scale=0.3)
rowing_sprite.update(scale=0.3)
running_sprite.update(scale=0.3)

jumping_jacks_sprite.update(x=ANIMATION_POSITION[0] - jumping_jacks_sprite.width // 2)
lifting_sprite.update(x=ANIMATION_POSITION[0] - lifting_sprite.width // 2)
rowing_sprite.update(x=ANIMATION_POSITION[0] - rowing_sprite.width // 2)
running_sprite.update(x=ANIMATION_POSITION[0] - running_sprite.width // 2)

def train_classifier():
    X, y, persons, encoder = activity.load_and_preprocess_data(DATA_PATH)
    clf = activity.get_pipeline()
    activity.run_single_evaluation(X, y, persons, 42, encoder, clf)

    return clf

def read_sensor_data(dt):
    acc = sensor.get_value("accelerometer")
    gyro = sensor.get_value("gyroscope")
    timestamp = time.time()

    if len(data) > 1002:  # limit buffer size to last 1000 readings
        data.pop(0)
    # only save if both sensors exist
    if acc is not None and gyro is not None:
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
    df = pd.DataFrame(data)
    return df
 
# extract features 
def transform_data_for_model(df):
    return extract_features(df)

def predict_activity(classifier, transformed_data):
    return classifier.predict(transformed_data)

def update(dt, classifier):
    global activity_name
    df = read_sensor_data(dt)
    transformed_data = transform_data_for_model(df)
    activity_name = predict_activity(classifier, transformed_data)
    text.text = f'Current Activity: {activity_name}'

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.Q:   # Press Q to quit the game
        os._exit(0)

@window.event
def on_draw():
    pyglet.gl.glClearColor(BACKGROUND_COLOR[0]/255, BACKGROUND_COLOR[1]/255, BACKGROUND_COLOR[2]/255, 1)
    window.clear()
    
    if activity_name == "jumping_jacks":
        jumping_jacks_sprite.draw()
    elif activity_name == "lifting":
        lifting_sprite.draw()
    elif activity_name == "rowing":
        rowing_sprite.draw()
    elif activity_name == "running":
        running_sprite.draw()
        
    text.draw()

    print(data)
    print(read_sensor_data(0))  # Print sensor data to console for debugging
    print(clf)
clf = train_classifier()
pyglet.app.run()

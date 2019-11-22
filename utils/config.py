"""
config.py

Configuration file, maintaining the following sections:
 * game_environment (folder)
 
# TODO: Revamp the config-file
# TODO: Create a .cfg file in environment instead of using this file...
"""
import numpy as np

# ----------------------------------------------------- CONTROLS ----------------------------------------------------- #

TIME_ALL = False  # Time every component in the full pipeline (only used to find bottlenecks) [def=False]

# ------------------------------------------------- GAME ENVIRONMENT ------------------------------------------------- #

# --> MAIN CONTROL <-- #
FPS = 24  # Number of frames each second  [def=24]
HUMAN_INPUT = False  # Control the player by hand instead of by an algorithm [def=True]

# --> WINDOW SPECIFIC <-- #
PTM = 50  # Pixel-to-meters: number of pixels that represent one meter  [def=50]
AXIS_X = 14  # Number of meters the x-axis represents [def=14]
AXIS_Y = 14  # Number of meters the y-axis represents [def=14]

# --> IDs <-- #
ID_TARGET = -1
ID_PLAYER = 0
ID_WALL = 1

# --> BOT SPECIFIC <-- #
BOT_DRIVING_SPEED = 0.3  # Speed of bot when driving straight expressed in m/s [def=0.3]
BOT_MASS = 1.8  # Mass of the bot expressed in kg [def=1.8]
BOT_RADIUS = 0.065  # Radius of the bot expressed in meters [def=0.085]
BOT_TURNING_SPEED = 3 * np.pi / 4  # Speed of bot when turning expressed in radians per second [def=3*pi/4]

# --> TARGET SPECIFIC <-- #
TARGET_RADIUS = 0.065  # Radius of the target expressed in meters [def=0.05]
TARGET_REACHED = 0.1  # Target is reached when within this range, expressed in meters [def=0.1]

# --> WALL SPECIFIC <-- #
WALL_THICKNESS = 0.1  # Thickness of the wall expressed in meters [def=0.1]

# --> SENSOR SPECIFIC <-- #
SENSOR_RAY_DISTANCE = 1.5  # Distance a ray-sensor reaches, expressed in meters [def=1.5]

# --> GAME CREATION SPECIFIC <-- #
MIN_ROOM_WIDTH = int((2 * AXIS_X + 1) / 5)  # Minimal width for one room
ROOM_ATTEMPTS = 8  # Number of times a room is tried to be added to the room [def=8]
GAMES_AMOUNT = 1  # 000  # Number of games created  [def=1000]  TODO
FILLED_ROOM_RATIO = 0.2  # The maximum percentage of tiles one room may contain

# --> NOISE <-- #
NOISE_TIME = 0.02  # Alpha for Gaussian distribution concerning the time noise during game-progress [def=0.02]
NOISE_SENSOR_ANGLE = 0.002  # Alpha for the Gaussian distribution concerning the distance sensor [def=0.002]
NOISE_SENSOR_DIST = 0.2  # Alpha for the Gaussian distribution concerning the distance sensor [def=0.2]
NOISE_SENSOR_PROXY = 0.2  # Alpha for the Gaussian distribution concerning the proximity sensor [def=0.2]

import numpy as np
cimport numpy as np

cdef class GameConfigCy:
    __slots__ = ("bot_driving_speed", "bot_radius", "bot_turning_speed",
                 "batch", "duration", "max_game_id", "max_eval_game_id", "time_all", "fps",
                 "p2m", "x_axis", "y_axis",
                 "noise_time", "noise_angle", "noise_distance", "noise_proximity",
                 "sensor_ray_distance",
                 "target_reached")
    
    def __init__(self):
        # [BOT]
        # Speed of bot when driving straight expressed in m/s [def=0.5]
        self.bot_driving_speed = 0.5
        # Radius of the bot expressed in meters [def=0.1]
        self.bot_radius = 0.1
        # Speed of bot when turning expressed in radians per second [def=13*np.pi/16]
        self.bot_turning_speed = 13 * np.pi / 16
        
        # [CONTROL]
        # Number of games on which a single genome is evaluated [def=8]
        self.batch = 8
        # Number of seconds it takes for one game to complete [def=100]
        self.duration = 100
        # Max ID of game (starting from 1) [def=1000]
        self.max_game_id = 1000
        # Max ID of evaluation game (starting from max_id) [def=1200]
        self.max_eval_game_id = 1200
        # Time every component in the full pipeline (only used to find bottlenecks) [def=False]
        self.time_all = False
        # Number of frames each second  [def=20]
        self.fps = 20
        
        # [CREATION]
        # Pixel-to-meters: number of pixels that represent one meter  [def=50]
        self.p2m = 50
        # Number of meters the x-axis represents [def=14]
        self.x_axis = 14
        # Number of meters the y-axis represents [def=14]
        self.y_axis = 14
        
        # [NOISE]
        # Alpha in Gaussian distribution for time noise, max 0.02s deviation [def=0.005]
        self.noise_time = 0.005
        # Alpha in Gaussian distribution for angular sensor, max 0.7° deviation [def=0.001]
        self.noise_angle = 0.001
        # Alpha in Gaussian distribution for distance sensor, max ~1.5cm deviation [def=0.005]
        self.noise_distance = 0.005
        # Alpha in Gaussian distribution for proximity sensor, max ~1.5cm deviation [def=0.005]
        self.noise_proximity = 0.005
        
        # [SENSOR]
        # Distance a ray-sensor reaches, expressed in meters [def=1.5]
        self.sensor_ray_distance = 1.5
        
        # [TARGET]
        # Target is reached when within this range, expressed in meters [def=0.5]
        self.target_reached = 0.5

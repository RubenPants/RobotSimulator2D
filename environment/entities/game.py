"""
game.py

Game class which contains the player, target, and all the walls.
"""
import pickle
import random

import numpy as np
import pylab as pl
from matplotlib import collections as mc

from configs.config import GameConfig
from environment.entities.robots import FootBot
from utils.dictionary import *
from utils.intersection import circle_line_intersection
from utils.line2d import Line2d
from utils.vec2d import Vec2d


class Game:
    """
    A game environment is built up from the following segments:
        * walls: Set of Line2d objects representing the walls in the maze
        * robot: The player manoeuvring in the environment
        * target: Robot that must be reached by the robot
    """
    
    __slots__ = ("bot_driving_speed", "bot_radius", "bot_turning_speed",
                 "batch", "duration", "max_game_id", "max_eval_game_id", "fps",
                 "p2m", "x_axis", "y_axis",
                 "noise_time", "noise_angle", "noise_distance", "noise_proximity",
                 "sensor_ray_distance",
                 "target_reached",
                 "silent", "noise",
                 "done", "id", "path", "player", "steps_taken", "target", "walls")
    
    def __init__(self,
                 config: GameConfig = None,
                 game_id: int = 0,
                 noise: bool = False,
                 overwrite: bool = False,
                 silent: bool = False,
                 ):
        """
        Define a new game.

        :param config: Configuration file related to the game (only needed to pass during creation)
        :param game_id: Game id
        :param noise: Add noise when progressing the game
        :param overwrite: Overwrite pre-existing games
        :param silent: Do not print anything
        """
        # Set config (or placeholder if config not defined)
        self.bot_driving_speed: float = config.bot_driving_speed if config else 0
        self.bot_radius: float = config.bot_radius if config else 0
        self.bot_turning_speed: float = config.bot_turning_speed if config else 0
        self.batch: int = config.batch if config else 0
        self.duration: int = config.duration if config else 0
        self.max_game_id: int = config.max_game_id if config else 0
        self.max_eval_game_id: int = config.max_eval_game_id if config else 0
        self.fps: int = config.fps if config else 0
        self.p2m: int = config.p2m if config else 0
        self.x_axis: int = config.x_axis if config else 0
        self.y_axis: int = config.y_axis if config else 0
        self.noise_time: float = config.noise_time if config else 0
        self.noise_angle: float = config.noise_angle if config else 0
        self.noise_distance: float = config.noise_distance if config else 0
        self.noise_proximity: float = config.noise_proximity if config else 0
        self.sensor_ray_distance: float = config.sensor_ray_distance if config else 0
        self.target_reached: float = config.target_reached if config else 0
        
        # Environment specific parameters
        self.silent: bool = silent  # True: Do not print out statistics
        self.noise: bool = noise  # Add noise to the game-environment
        
        # Placeholders for parameters
        self.done: bool = False  # Game has finished
        self.id: int = game_id  # Game's ID-number
        self.path: dict = None  # Coordinates together with distance to target
        self.player: FootBot = None  # Candidate-robot
        self.steps_taken: int = 0  # Number of steps taken by the agent
        self.target: Vec2d = None  # Target-robot
        self.walls: list = None  # List of all walls in the game
        
        # Check if game already exists, if not create new game
        if overwrite or not self.load(): self.create_empty_game()
    
    def __str__(self):
        return f"game_{self.id:05d}"
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def close(self):
        """
        :return: Final state and useful statistics
        """
        return {
            D_A_STAR:         self.path[self.player.init_pos[0], self.player.init_pos[1]],
            D_DIST_TO_TARGET: self.player.get_sensor_reading_distance(),
            D_DONE:           self.done,
            D_GAME_ID:        self.id,
            D_PATH:           self.path,
            D_POS:            self.player.pos,
            D_STEPS:          self.steps_taken,
        }
    
    def reset(self):
        """
        Reset the game.

        :return: Observation
        """
        self.steps_taken = 0
        self.player.reset()
        return self.get_observation()
    
    def step(self, l: float, r: float):
        """
        Progress one step in the game.

        :param l: Left wheel speed [-1..1]
        :param r: Right wheel speed [-1..1]
        :return: Observation (Dictionary), target_reached (Boolean)
        """
        # Progress the game
        dt = 1.0 / self.fps + (abs(random.gauss(0, self.noise_time)) if self.noise else 0)
        return self.step_dt(dt=dt, l=l, r=r)
    
    def step_dt(self, dt: float, l: float, r: float):
        """
        Progress one step in the game based on a predefined delta-time. This method should only be used for debugging or
        visualization purposes.

        :param dt: Delta time
        :param l: Left wheel speed [-1..1]
        :param r: Right wheel speed [-1..1]
        :return: Observation (Dictionary), target_reached (Boolean)
        """
        # Progress the game
        self.steps_taken += 1
        self.player.drive(dt, lw=l, rw=r)
        
        # Check if intersected with a wall, if so then set player back to old position
        for wall in self.walls:
            inter, _ = circle_line_intersection(c=self.player.pos, r=self.player.radius, l=wall)
            if inter:
                self.player.pos.x = self.player.prev_pos.x
                self.player.pos.y = self.player.prev_pos.y
                self.player.angle = self.player.prev_angle
                break
        
        # Check if target reached
        if self.player.get_sensor_reading_distance() <= self.target_reached: self.done = True
        
        # Return the current observations
        return self.get_observation()
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    def create_empty_game(self):
        """
        Create an empty game that only contains the boundary walls.
        """
        # Create random set of walls
        self.walls = get_boundary_walls(x_axis=self.x_axis, y_axis=self.y_axis)
        self.target = Vec2d(0.5, self.y_axis - 0.5)
        self.player = FootBot(game=self,
                              init_pos=Vec2d(self.x_axis - 0.5, 0.5),
                              init_orient=np.pi / 2)
        
        # Save the new game
        self.save()
        if not self.silent: print("New game created under id: {}".format(self.id))
    
    def get_observation(self):
        """
        Get the current observation of the game. The following gets returned as a dictionary:
         * D_ANGLE: The angle the player is currently heading
         * D_DIST_TO_TARGET: Distance from player's current position to target in crows flight
         * D_GAME_ID: The game's ID
         * D_POS: The current position of the player in the maze (expressed in pixels)
         * D_SENSOR_LIST: List of all the sensors (proximity, angular, distance)
        """
        return {
            D_ANGLE:          self.player.angle,
            D_DIST_TO_TARGET: self.player.get_sensor_reading_distance(),
            D_DONE:           self.done,
            D_GAME_ID:        self.id,
            D_POS:            self.player.pos,
            D_SENSOR_LIST:    self.get_sensor_list(),
            D_STEPS:          self.steps_taken,
        }
    
    def get_sensor_list(self):
        """
        Return a list of sensory-readings, with first the proximity sensors, then the angular sensors and at the end the
        distance-sensor.
        
        :return: [proximity_sensors, angular_sensors, distance_sensor]
        """
        # Read the sensors
        sensor_readings = self.player.get_sensor_readings()
        proximity = sensor_readings[D_SENSOR_PROXIMITY]
        angular = sensor_readings[D_SENSOR_ANGLE]
        distance = sensor_readings[D_SENSOR_DISTANCE]
        
        result = []  # Add sensory-readings in one list
        for i in range(len(proximity)): result.append(proximity[i])  # Proximity IDs go from 0 to proximity_length
        for i in range(len(angular)): result.append(angular[i])  # Angular IDs go from 0 to angular_length
        result.append(distance)
        return result
    
    def set_player_angle(self, a: float):
        """
        Set a new initial angle for the player.
        """
        self.player.init_angle = a
        self.player.angle = a
    
    def set_player_pos(self, p: Vec2d):
        """
        Set a new initial position for the player.
        """
        self.player.init_pos.x = p.x
        self.player.init_pos.y = p.y
        self.player.pos.x = p.x
        self.player.pos.y = p.y
        self.player.prev_pos.x = p.x
        self.player.prev_pos.y = p.y
    
    # ---------------------------------------------> FUNCTIONAL METHODS <--------------------------------------------- #
    
    def save(self):
        persist_dict = dict()
        persist_dict[D_BOT_DRIVING_SPEED] = self.bot_driving_speed
        persist_dict[D_BOT_RADIUS] = self.bot_radius
        persist_dict[D_BOT_TURNING_SPEED] = self.bot_turning_speed
        persist_dict[D_BATCH] = self.batch
        persist_dict[D_DURATION] = self.duration
        persist_dict[D_MAX_GAME_ID] = self.max_game_id
        persist_dict[D_MAX_EVAL_GAME_ID] = self.max_eval_game_id
        persist_dict[D_FPS] = self.fps
        persist_dict[D_PTM] = self.p2m
        persist_dict[D_X_AXIS] = self.x_axis
        persist_dict[D_Y_AXIS] = self.y_axis
        persist_dict[D_NOISE_TIME] = self.noise_time
        persist_dict[D_NOISE_ANGLE] = self.noise_angle
        persist_dict[D_NOISE_DISTANCE] = self.noise_distance
        persist_dict[D_NOISE_PROXIMITY] = self.noise_proximity
        persist_dict[D_SENSOR_RAY_DISTANCE] = self.sensor_ray_distance
        persist_dict[D_TARGET_REACHED] = self.target_reached
        persist_dict[D_ANGLE] = self.player.init_angle  # Initial angle of player
        if self.path: persist_dict[D_PATH] = [(p[0], p[1]) for p in self.path.items()]
        persist_dict[D_POS] = (self.player.init_pos.x, self.player.init_pos.y)  # Initial position of player
        persist_dict[D_TARGET] = (self.target.x, self.target.y)
        persist_dict[D_WALLS] = [((w.x.x, w.x.y), (w.y.x, w.y.y)) for w in self.walls]
        with open(f'environment/games_db/{self}', 'wb') as f: pickle.dump(persist_dict, f)
    
    def load(self):
        """
        Load in a game, specified by its current id.

        :return: True: game successfully loaded | False: otherwise
        """
        try:
            with open(f'environment/games_db/{self}', 'rb') as f:
                game = pickle.load(f)
            self.bot_driving_speed = game[D_BOT_DRIVING_SPEED]
            self.bot_radius = game[D_BOT_RADIUS]
            self.bot_turning_speed = game[D_BOT_TURNING_SPEED]
            self.batch = game[D_BATCH]
            self.duration = game[D_DURATION]
            self.max_game_id = game[D_MAX_GAME_ID]
            self.max_eval_game_id = game[D_MAX_EVAL_GAME_ID]
            self.fps = game[D_FPS]
            self.p2m = game[D_PTM]
            self.x_axis = game[D_X_AXIS]
            self.y_axis = game[D_Y_AXIS]
            self.noise_time = game[D_NOISE_TIME]
            self.noise_angle = game[D_NOISE_ANGLE]
            self.noise_distance = game[D_NOISE_DISTANCE]
            self.noise_proximity = game[D_NOISE_PROXIMITY]
            self.sensor_ray_distance = game[D_SENSOR_RAY_DISTANCE]
            self.target_reached = game[D_TARGET_REACHED]
            self.player = FootBot(game=self)  # Create a dummy-player to set values on
            self.set_player_angle(game[D_ANGLE])
            self.set_player_pos(Vec2d(game[D_POS][0], game[D_POS][1]))
            self.path = {p[0]: p[1] for p in game[D_PATH]}
            self.target = Vec2d(game[D_TARGET][0], game[D_TARGET][1])
            self.walls = [Line2d(Vec2d(w[0][0], w[0][1]), Vec2d(w[1][0], w[1][1])) for w in game[D_WALLS]]
            if not self.silent: print(f"Existing game loaded with id: {self.id}")
            return True
        except FileNotFoundError:
            return False
    
    def get_blueprint(self, ax=None):
        """
        :return: The blue-print map of the board (matplotlib Figure)
        """
        if not ax: fig, ax = pl.subplots()
        
        # Draw all the walls
        walls = []
        for w in self.walls: walls.append([(w.x.x, w.x.y), (w.y.x, w.y.y)])
        lc = mc.LineCollection(walls, linewidths=2, colors='k')
        ax.add_collection(lc)
        
        # Add target to map
        pl.plot(0.5, self.y_axis - 0.5, 'go')
        
        # Adjust the boundaries
        pl.xlim(0, self.x_axis)
        pl.ylim(0, self.y_axis)
        
        # Return the figure in its current state
        return ax


def get_boundary_walls(x_axis, y_axis):
    """ :return: Set of the four boundary walls """
    a = Vec2d(0, 0)
    b = Vec2d(x_axis, 0)
    c = Vec2d(x_axis, y_axis)
    d = Vec2d(0, y_axis)
    return [Line2d(a, b), Line2d(b, c), Line2d(c, d), Line2d(d, a)]

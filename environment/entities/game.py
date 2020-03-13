"""
game.py

Game class which contains the player, target, and all the walls.
"""
import random

import numpy as np
import pylab as pl
from matplotlib import collections as mc

from configs.config import GameConfig
from environment.entities.robots import FootBot
from utils.dictionary import *
from utils.intersection import circle_line_intersection
from utils.line2d import Line2d
from utils.myutils import load_pickle, store_pickle
from utils.vec2d import Vec2d


class Game:
    """
    A game environment is built up from the following segments:
        * walls: Set of Line2d objects representing the walls in the maze
        * robot: The player manoeuvring in the environment
        * target: Robot that must be reached by the robot
    """
    
    __slots__ = (
        "bot_driving_speed", "bot_radius", "bot_turning_speed",
        "fps", "p2m", "x_axis", "y_axis",
        "noise_time", "noise_angle", "noise_distance", "noise_proximity",
        "sensor_ray_distance",
        "target_reached",
        "silent", "noise", "save_path",
        "done", "id", "path", "player", "steps_taken", "target", "walls"
    )
    
    def __init__(self,
                 game_id: int = 0,
                 config: GameConfig = None,
                 noise: bool = False,
                 overwrite: bool = False,
                 save_path: str = '',
                 silent: bool = False,
                 ):
        """
        Define a new game.

        :param config: Configuration file related to the game (only needed to pass during creation)
        :param game_id: Game id
        :param noise: Add noise when progressing the game
        :param overwrite: Overwrite pre-existing games
        :param save_path: Save and load the game from different directories
        :param silent: Do not print anything
        """
        # Set config (or placeholder if config not defined)
        self.bot_driving_speed: float = 0
        self.bot_radius: float = 0
        self.bot_turning_speed: float = 0
        self.fps: int = 0
        self.p2m: int = 0
        self.x_axis: int = 0
        self.y_axis: int = 0
        self.noise_time: float = 0
        self.noise_angle: float = 0
        self.noise_distance: float = 0
        self.noise_proximity: float = 0
        self.sensor_ray_distance: float = 0
        self.target_reached: float = 0
        
        # Environment specific parameters
        self.silent: bool = silent  # True: Do not print out statistics
        self.noise: bool = noise  # Add noise to the game-environment
        self.save_path: str = save_path if save_path else 'environment/games_db/'
        
        # Placeholders for parameters
        self.done: bool = False  # Game has finished
        self.id: int = game_id  # Game's ID-number
        self.path: dict = None  # Coordinates together with distance to target
        self.player: FootBot = None  # Candidate-robot
        self.steps_taken: int = 0  # Number of steps taken by the agent
        self.target: Vec2d = None  # Target-robot
        self.walls: set = None  # Set of all walls in the game
        
        # Check if game already exists, if not create new game
        if overwrite or not self.load():
            if not config: config = GameConfig()
            self.set_config_params(config)
            self.create_empty_game()
    
    def __str__(self):
        return f"game_{self.id:05d}"
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def close(self):
        """Final state of the agent's statistics."""
        return {
            D_DIST_TO_TARGET: self.player.get_sensor_readings_distance(),
            D_DONE:           self.done,
            D_GAME_ID:        self.id,
            D_POS:            self.player.pos,
            D_STEPS:          self.steps_taken
        }
    
    def game_params(self):
        """Get all the game-related parameters."""
        return {
            D_A_STAR:  self.path[self.player.init_pos[0], self.player.init_pos[1]],
            D_PATH:    self.path,
            D_FPS:     self.fps,
            D_GAME_ID: self.id,
        }
    
    def get_observation(self):
        """Get the current observation of the game in the form of a dictionary."""
        return {
            D_DONE:        self.done,
            D_GAME_ID:     self.id,
            D_SENSOR_LIST: self.player.get_sensor_readings(),
        }
    
    def reset(self):
        """Reset the game and return initial observations."""
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
        if self.player.get_sensor_readings_distance() <= self.target_reached: self.done = True
        
        # Return the current observations
        return self.get_observation()
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    def create_empty_game(self):
        """ Create an empty game that only contains the boundary walls. """
        # Create random set of walls
        self.walls = get_boundary_walls(x_axis=self.x_axis, y_axis=self.y_axis)
        self.target = Vec2d(0.5, self.y_axis - 0.5)
        self.player = FootBot(game=self,
                              init_pos=Vec2d(self.x_axis - 0.5, 0.5),
                              init_orient=np.pi / 2)
        
        # Save the new game
        self.save()
        if not self.silent: print("New game created under id: {}".format(self.id))
    
    def set_config_params(self, config):
        """ Store all the configured parameters locally. """
        self.bot_driving_speed: float = config.bot_driving_speed
        self.bot_radius: float = config.bot_radius
        self.bot_turning_speed: float = config.bot_turning_speed
        self.fps: int = config.fps
        self.p2m: int = config.p2m
        self.x_axis: int = config.x_axis
        self.y_axis: int = config.y_axis
        self.noise_time: float = config.noise_time
        self.noise_angle: float = config.noise_angle
        self.noise_distance: float = config.noise_distance
        self.noise_proximity: float = config.noise_proximity
        self.sensor_ray_distance: float = config.sensor_ray_distance
        self.target_reached: float = config.target_reached
    
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
        store_pickle(persist_dict, f'{self.save_path}{self}')
    
    def load(self):
        """
        Load in a game, specified by its current id.

        :return: True: game successfully loaded | False: otherwise
        """
        try:
            game = load_pickle(f'{self.save_path}{self}')
            self.bot_driving_speed = game[D_BOT_DRIVING_SPEED]
            self.bot_radius = game[D_BOT_RADIUS]
            self.bot_turning_speed = game[D_BOT_TURNING_SPEED]
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
            self.walls = {Line2d(Vec2d(w[0][0], w[0][1]), Vec2d(w[1][0], w[1][1])) for w in game[D_WALLS]}
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
    return {Line2d(a, b), Line2d(b, c), Line2d(c, d), Line2d(d, a)}


def initial_sensor_readings():
    """Return a list of the sensors their maximum value."""
    game = Game(game_id=0, silent=True)
    return game.player.get_sensor_readings()

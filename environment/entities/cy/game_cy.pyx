"""
game_cy.pyx

Cython version of the game.py file. Note that this file co-exists with a .pxd file (needed to import the game methods
in other files).
"""
import pickle
import random

import numpy as np
cimport numpy as np
import pylab as pl
from matplotlib import collections as mc

from configs.config import GameConfig
from environment.entities.cy.robots_cy cimport FootBotCy
from utils.dictionary import *
from utils.cy.intersection_cy cimport circle_line_intersection_cy
from utils.cy.line2d_cy cimport Line2dCy
from utils.cy.vec2d_cy cimport Vec2dCy

cdef class GameCy:
    """
    A game environment is built up from the following segments:
        * walls: Set of Line2d objects representing the walls in the maze
        * robot: The player manoeuvring in the environment
        * target: Robot that must be reached by the robot
    """

    __slots__ = ("bot_driving_speed", "bot_radius", "bot_turning_speed",
                 "fps", "p2m", "x_axis", "y_axis",
                 "noise_time", "noise_angle", "noise_distance", "noise_proximity",
                 "sensor_ray_distance",
                 "target_reached",
                 "silent", "noise", "save_path",
                 "done", "id", "path", "player", "steps_taken", "target", "walls")
    
    def __init__(self,
                 int game_id=0,
                 config=None,
                 bint noise=True,
                 bint overwrite=False,
                 str save_path = '',
                 bint silent=False,
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
        # Set config
        self.bot_driving_speed = 0
        self.bot_radius = 0
        self.bot_turning_speed = 0
        self.fps = 0
        self.p2m = 0
        self.x_axis = 0
        self.y_axis = 0
        self.noise_time = 0
        self.noise_angle = 0
        self.noise_distance = 0
        self.noise_proximity = 0
        self.sensor_ray_distance = 0
        self.target_reached = 0
        
        # Environment specific parameters
        self.silent = silent  # True: Do not print out statistics
        self.noise = noise  # Add noise to the game-environment
        self.save_path = save_path if save_path else 'environment/games_db/'
        
        # Placeholders for parameters
        self.done = False  # Game has finished
        self.id = game_id  # Game's ID-number
        self.path = dict()  # Coordinates together with distance to target
        self.player = None  # Candidate-robot
        self.steps_taken = 0  # Number of steps taken by the agent
        self.target = None  # Target-robot
        self.walls = None  # List of all walls in the game
        
        # Check if game already exists, if not create new game
        if overwrite or not self.load():
            if not config: config = GameConfig()
            self.set_config_params(config)
            self.create_empty_game()
    
    def __str__(self):
        return f"game_{self.id:05d}"
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    cpdef dict close(self):
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
    
    cpdef dict reset(self):
        """
        Reset the game.

        :return: Observation
        """
        self.steps_taken = 0
        self.player.reset()
        return self.get_observation()
    
    cpdef step(self, float l, float r):
        """
        Progress one step in the game.

        :param l: Left wheel speed [-1..1]
        :param r: Right wheel speed [-1..1]
        :return: Observation (Dictionary), target_reached (Boolean)
        """
        # Define used parameters
        cdef float dt
        
        # Progress the game
        dt = 1.0 / self.fps + (abs(random.gauss(0, self.noise_time)) if self.noise else 0)
        return self.step_dt(dt=dt, l=l, r=r)
    
    cpdef step_dt(self, float dt, float l, float r):
        """
        Progress one step in the game.

        :param dt: Delta time
        :param l: Left wheel speed [-1..1]
        :param r: Right wheel speed [-1..1]
        :return: Observation (Dictionary), target_reached (Boolean)
        """
        # Define used parameters
        cdef Line2dCy wall
        cdef bint inter
        cdef dict obs
        
        # Progress the game
        self.steps_taken += 1
        self.player.drive(dt, lw=l, rw=r)
        
        # Check if intersected with a wall, if so then set player back to old position
        for wall in self.walls:
            inter, _ = circle_line_intersection_cy(c=self.player.pos, r=self.player.radius, l=wall)
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
    
    cpdef void create_empty_game(self):
        """
        Create an empty game that only contains the boundary walls.
        """
        # Create random set of walls
        self.walls = get_boundary_walls(x_axis=self.x_axis, y_axis=self.y_axis)
        self.target = Vec2dCy(0.5, self.y_axis - 0.5)
        self.player = FootBotCy(game=self,
                                init_pos=Vec2dCy(self.x_axis - 0.5, 0.5),
                                init_orient=np.pi / 2)
        
        # Save the new game
        self.save()
        if not self.silent: print("New game created under id: {}".format(self.id))
    
    cpdef dict get_observation(self):
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
    
    cpdef list get_sensor_list(self):
        """
        Return a list of sensory-readings, with first the proximity sensors, then the angular sensors and at the end the
        distance-sensor.
        """
        cdef dict sensor_readings
        cdef dict proximity
        cdef dict angular
        cdef float distance
        cdef list result
        cdef int i
        
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
    
    cpdef void set_config_params(self, config):
        """ Store all the configured parameters locally. """
        self.bot_driving_speed = config.bot_driving_speed
        self.bot_radius = config.bot_radius
        self.bot_turning_speed = config.bot_turning_speed
        self.fps = config.fps
        self.p2m = config.p2m
        self.x_axis = config.x_axis
        self.y_axis = config.y_axis
        self.noise_time = config.noise_time
        self.noise_angle = config.noise_angle
        self.noise_distance = config.noise_distance
        self.noise_proximity = config.noise_proximity
        self.sensor_ray_distance = config.sensor_ray_distance
        self.target_reached = config.target_reached
    
    cpdef void set_player_angle(self, float a):
        """
        Set a new initial angle for the player.
        """
        self.player.init_angle = a
        self.player.angle = a
    
    cpdef void set_player_pos(self, Vec2dCy p):
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
    
    cpdef void save(self):
        cdef dict persist_dict = dict()
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
        with open(f'environment/games_db/{self}', 'wb') as f: pickle.dump(persist_dict, f)
    
    cpdef bint load(self):
        """
        Load in a game, specified by its current id.

        :return: True: game successfully loaded | False: otherwise
        """
        # Define used parameter
        cdef dict game
        
        try:
            with open(f'environment/games_db/{self}', 'rb') as f:
                game = pickle.load(f)
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
            self.player = FootBotCy(game=self)  # Create a dummy-player to set values on
            self.set_player_angle(game[D_ANGLE])
            self.set_player_pos(Vec2dCy(game[D_POS][0], game[D_POS][1]))
            self.path = {p[0]: p[1] for p in game[D_PATH]}
            self.target = Vec2dCy(game[D_TARGET][0], game[D_TARGET][1])
            self.walls = [Line2dCy(Vec2dCy(w[0][0], w[0][1]), Vec2dCy(w[1][0], w[1][1])) for w in game[D_WALLS]]
            if not self.silent: print(f"Existing game loaded with id: {self.id}")
            return True
        except FileNotFoundError:
            return False
    
    cpdef get_blueprint(self, ax=None):
        """
        :return: The blue-print map of the board (matplotlib Figure)
        """
        # Define used parameters
        cdef list walls
        cdef Line2dCy w
        
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


cpdef list get_boundary_walls(int x_axis, int y_axis):
    """ :return: Set of the four boundary walls """
    a = Vec2dCy(0, 0)
    b = Vec2dCy(x_axis, 0)
    c = Vec2dCy(x_axis, y_axis)
    d = Vec2dCy(0, y_axis)
    return [Line2dCy(a, b), Line2dCy(b, c), Line2dCy(c, d), Line2dCy(d, a)]

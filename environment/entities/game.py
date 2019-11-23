"""
game.py

Game class which contains the player, target, and all the walls.
"""
import pickle
import random

import pylab as pl
from matplotlib import collections as mc

from environment.entities.robots import FootBot
from utils.config import *
from utils.dictionary import *
from utils.intersection import circle_line_intersection
from utils.line2d import Line2d
from utils.myutils import drop, prep
from utils.vec2d import Vec2d


class Game:
    """
    A game environment is built up from the following segments:
        * walls: Set of Line2d objects representing the walls in the maze
        * robot: The player manoeuvring in the environment
        * target: Robot that must be reached by the robot
    """
    
    def __init__(self,
                 game_id: int = 0,
                 noise: bool = True,
                 overwrite: bool = False,
                 rel_path: str = 'environment/',
                 silent: bool = False):
        """
        Define a new game.

        :param game_id: Game id
        :param noise: Add noise when progressing the game
        :param overwrite: Overwrite pre-existing games
        :param rel_path: Relative path where Game object is stored or will be stored
        :param silent: Do not print anything
        """
        # Set path correct
        self.rel_path: str = rel_path  # Relative path to the 'environment' folder
        self.silent: bool = silent  # True: Do not print out statistics
        
        # Environment specific parameters
        self.noise: bool = noise  # Add noise to the game-environment
        
        # Placeholders for parameters
        self.id: int = game_id  # Game's ID-number
        self.player: FootBot = None  # Candidate-robot
        self.target: Vec2d = None  # Target-robot
        self.walls: list = None  # List of all walls in the game
        self.done: bool = False  # Game has finished
        
        # Check if game already exists, if not create new game
        if overwrite or not self.load():
            self.create_empty_game()
    
    def __str__(self):
        return "game_{id:05d}".format(id=self.id)
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    def close(self):
        """
        :return: Final state and useful statistics
        """
        return {
            D_GAME_ID:        self.id,
            D_POS:            self.player.pos,
            D_DIST_TO_TARGET: self.player.get_sensor_reading_distance(),
        }
    
    def reset(self):
        """
        Reset the game.

        :return: Observation
        """
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
        dt = 1.0 / FPS + abs(random.gauss(0, NOISE_TIME)) if self.noise else 1.0 / FPS
        self.player.drive(dt, lw=l, rw=r)
        
        # Check if intersected with a wall, if so then set player back to old position
        for wall in self.walls:
            inter, _ = circle_line_intersection(c=self.player.pos, r=self.player.radius, l=wall)
            if inter:
                self.player.pos.x = self.player.prev_pos.x
                self.player.pos.y = self.player.prev_pos.y
                break
        
        # Get the current observations
        obs = self.get_observation()
        
        # Check if target reached
        if obs[D_DIST_TO_TARGET] <= TARGET_REACHED:
            self.done = True
        
        return obs, self.done
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    def create_empty_game(self):
        """
        Create an empty game that only contains the boundary walls.
        """
        # Create random set of walls
        self.walls = get_boundary_walls()
        self.target = Vec2d(0.5, AXIS_Y - 0.5)
        self.player = FootBot(game=self,
                              init_pos=Vec2d(AXIS_X - 0.5, 0.5),
                              init_orient=np.pi / 2)
        
        # Save the new game
        self.save()
        
        if not self.silent:
            print("New game created under id: {}".format(self.id))
    
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
            D_GAME_ID:        self.id,
            D_POS:            self.player.pos,
            D_SENSOR_LIST:    self.get_sensor_list(),
        }
    
    def get_sensor_list(self):
        """
        Return a list of sensory-readings, with first the proximity sensors, then the angular sensors and at the end the
        distance-sensor.
        """
        # Read the sensors
        sensor_readings = self.player.get_sensor_readings()
        proximity = sensor_readings[D_SENSOR_PROXIMITY]
        angular = sensor_readings[D_SENSOR_ANGLE]
        distance = sensor_readings[D_SENSOR_DISTANCE]
        
        # Add sensory-readings in one list
        result = []
        for i in range(len(proximity)):  # Proximity IDs go from 0 to proximity_length
            result.append(proximity[i])
        for i in range(len(angular)):  # Angular IDs go from 0 to angular_length
            result.append(angular[i])
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
        if TIME_ALL:
            prep(key='load_save', silent=True)
        try:
            persist_dict = dict()
            persist_dict[D_ANGLE] = self.player.init_angle  # Initial angle of player
            persist_dict[D_POS] = (self.player.init_pos.x, self.player.init_pos.y)  # Initial position of player
            persist_dict[D_TARGET] = (self.target.x, self.target.y)
            persist_dict[D_WALLS] = [((w.x.x, w.x.y), (w.y.x, w.y.y)) for w in self.walls]
            with open('{p}games_db/{g}'.format(p=self.rel_path, g=self), 'wb') as f:
                pickle.dump(persist_dict, f)
        finally:
            if TIME_ALL:
                drop(key='load_save', silent=True)
    
    def load(self):
        """
        Load in a game, specified by its current id.

        :return: True: game successfully loaded | False: otherwise
        """
        if TIME_ALL:
            prep(key='load_save', silent=True)
        try:
            with open('{p}games_db/{g}'.format(p=self.rel_path, g=self), 'rb') as f:
                game = pickle.load(f)
            self.player = FootBot(game=self)  # Create a dummy-player to set values on
            self.set_player_angle(game[D_ANGLE])
            self.set_player_pos(Vec2d(game[D_POS][0], game[D_POS][1]))
            self.target = Vec2d(game[D_TARGET][0], game[D_TARGET][1])
            self.walls = [Line2d(Vec2d(w[0][0], w[0][1]), Vec2d(w[1][0], w[1][1])) for w in game[D_WALLS]]
            if not self.silent:
                print("Existing game loaded with id: {}".format(self.id))
            return True
        except FileNotFoundError:
            return False
        finally:
            if TIME_ALL:
                drop(key='load_save', silent=True)
    
    def get_blueprint(self):
        """
        :return: The blue-print map of the board (matplotlib Figure)
        """
        fig, ax = pl.subplots()
        
        # Draw all the walls
        walls = []
        for w in self.walls:
            walls.append([(w.x.x, w.x.y), (w.y.x, w.y.y)])
        lc = mc.LineCollection(walls, linewidths=2)
        ax.add_collection(lc)
        
        # Add target to map
        pl.plot(0.5, AXIS_Y - 0.5, 'go')
        
        # Adjust the boundaries
        pl.xlim(0, AXIS_X)
        pl.ylim(0, AXIS_Y)
        
        # Return the figure in its current state
        return ax


def get_boundary_walls():
    """
    :return: Set of the four boundary walls
    """
    a = Vec2d(0, 0)
    b = Vec2d(AXIS_X, 0)
    c = Vec2d(AXIS_X, AXIS_Y)
    d = Vec2d(0, AXIS_Y)
    return [Line2d(a, b), Line2d(b, c), Line2d(c, d), Line2d(d, a)]

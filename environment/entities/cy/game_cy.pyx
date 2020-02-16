"""
game_cy.pyx

Cython version of the game.py file. Note that this file co-exists with a .pxd file (needed to import the game methods
in other files).
"""
import pickle
import random

import pylab as pl
from matplotlib import collections as mc


from utils.dictionary import *

from utils.cy.intersection_cy cimport circle_line_intersection_cy
from utils.cy.line2d_cy cimport Line2dCy

cdef class GameCy:
    """
    A game environment is built up from the following segments:
        * walls: Set of Line2d objects representing the walls in the maze
        * robot: The player manoeuvring in the environment
        * target: Robot that must be reached by the robot
    """
    
    __slots__ = ("rel_path", "silent", "noise", "done", "id", "path", "player", "steps_taken", "target", "walls")
    
    def __init__(self,
                 int game_id=0,
                 bint noise=True,
                 bint overwrite=False,
                 str rel_path='environment/',
                 bint silent=False):
        """
        Define a new game.

        :param game_id: Game id
        :param noise: Add noise when progressing the game
        :param overwrite: Overwrite pre-existing games
        :param rel_path: Relative path where Game object is stored or will be stored
        :param silent: Do not print anything
        """
        # Set path correct
        self.rel_path = rel_path  # Relative path to the 'environment' folder
        self.silent = silent  # True: Do not print out statistics
        
        # Environment specific parameters
        self.noise = noise  # Add noise to the game-environment
        
        # Placeholders for parameters
        self.done = False  # Game has finished
        self.id = game_id  # Game's ID-number
        self.path = None  # Coordinates together with distance to target
        self.player = None  # Candidate-robot
        self.steps_taken = 0  # Number of steps taken by the agent
        self.target = None  # Target-robot
        self.walls = None  # List of all walls in the game
        
        # Check if game already exists, if not create new game
        if overwrite or not self.load():
            self.create_empty_game()
    
    def __str__(self):
        return "game_{id:05d}".format(id=self.id)
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    cpdef dict close(self):
        """
        :return: Final state and useful statistics
        """
        return {
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
        cdef Line2dCy wall
        cdef bint inter
        cdef dict obs
        
        # Progress the game
        dt = 1.0 / FPS + abs(random.gauss(0, NOISE_TIME)) if self.noise else 1.0 / FPS
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
        if self.player.get_sensor_reading_distance() <= TARGET_REACHED:
            self.done = True
        
        # Return the current observations
        return self.get_observation()
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    cpdef void create_empty_game(self):
        """
        Create an empty game that only contains the boundary walls.
        """
        # Create random set of walls
        self.walls = get_boundary_walls()
        self.target = Vec2dCy(0.5, AXIS_Y - 0.5)
        self.player = FootBotCy(game=self,
                                init_pos=Vec2dCy(AXIS_X - 0.5, 0.5),
                                init_orient=np.pi / 2)
        
        # Save the new game
        self.save()
        
        if not self.silent:
            print("New game created under id: {}".format(self.id))
    
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
        
        # Add sensory-readings in one list
        result = []
        for i in range(len(proximity)):  # Proximity IDs go from 0 to proximity_length
            result.append(proximity[i])
        for i in range(len(angular)):  # Angular IDs go from 0 to angular_length
            result.append(angular[i])
        result.append(distance)
        return result
    
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
        persist_dict[D_ANGLE] = self.player.init_angle  # Initial angle of player
        if self.path: persist_dict[D_PATH] = [(p[0], p[1]) for p in self.path.items()]
        persist_dict[D_POS] = (self.player.init_pos.x, self.player.init_pos.y)  # Initial position of player
        persist_dict[D_TARGET] = (self.target.x, self.target.y)
        persist_dict[D_WALLS] = [((w.x.x, w.x.y), (w.y.x, w.y.y)) for w in self.walls]
        with open('{p}games_db/{g}'.format(p=self.rel_path, g=self), 'wb') as f:
            pickle.dump(persist_dict, f)
    
    cpdef bint load(self):
        """
        Load in a game, specified by its current id.

        :return: True: game successfully loaded | False: otherwise
        """
        # Define used parameter
        cdef dict game
        
        try:
            with open('{p}games_db/{g}'.format(p=self.rel_path, g=self), 'rb') as f:
                game = pickle.load(f)
            self.player = FootBotCy(game=self)  # Create a dummy-player to set values on
            self.set_player_angle(game[D_ANGLE])
            self.set_player_pos(Vec2dCy(game[D_POS][0], game[D_POS][1]))
            self.path = {p[0]: p[1] for p in game[D_PATH]}
            self.target = Vec2dCy(game[D_TARGET][0], game[D_TARGET][1])
            self.walls = [Line2dCy(Vec2dCy(w[0][0], w[0][1]), Vec2dCy(w[1][0], w[1][1])) for w in game[D_WALLS]]
            if not self.silent:
                print("Existing game loaded with id: {}".format(self.id))
            return True
        except FileNotFoundError:
            return False
    
    cpdef get_blueprint(self):
        """
        :return: The blue-print map of the board (matplotlib Figure)
        """
        # Define used parameters
        cdef list walls
        cdef Line2dCy w
        
        fig, ax = pl.subplots()
        
        # Draw all the walls
        walls = []
        for w in self.walls:
            walls.append([(w.x.x, w.x.y), (w.y.x, w.y.y)])
        lc = mc.LineCollection(walls, linewidths=2, colors='k')
        ax.add_collection(lc)
        
        # Add target to map
        pl.plot(0.5, AXIS_Y - 0.5, 'go')
        
        # Adjust the boundaries
        pl.xlim(0, AXIS_X)
        pl.ylim(0, AXIS_Y)
        
        # Return the figure in its current state
        return ax

cpdef list get_boundary_walls():
    """
    :return: Set of the four boundary walls
    """
    cdef Vec2dCy a = Vec2dCy(0, 0)
    cdef Vec2dCy b = Vec2dCy(AXIS_X, 0)
    cdef Vec2dCy c = Vec2dCy(AXIS_X, AXIS_Y)
    cdef Vec2dCy d = Vec2dCy(0, AXIS_Y)
    return [Line2dCy(a, b), Line2dCy(b, c), Line2dCy(c, d), Line2dCy(d, a)]

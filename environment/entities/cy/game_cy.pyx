"""
game_cy.pyx

Game class which contains the player, target, and all the walls.
"""
import random

import numpy as np
cimport numpy as np
import pylab as plt
from matplotlib import collections as mc

from environment.entities.cy.robots_cy cimport MarXBotCy
from utils.dictionary import *
from utils.cy.intersection_cy cimport circle_line_intersection_cy
from utils.cy.line2d_cy cimport Line2dCy
from utils.cy.vec2d_cy cimport Vec2dCy
from utils.myutils import store_pickle, load_pickle


cdef class GameCy:
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
        "ray_distance", "ray_distance_cum",
        "target_reached",
        "silent", "noise", "save_path",
        "done", "id", "path", "player", "steps_taken", "target", "walls"
    )
    
    def __init__(self,
                 int game_id,
                 config,
                 bint noise=False,
                 bint overwrite=False,
                 str save_path = '',
                 bint silent=True,
                 ):
        """
        Define a new game.

        :param game_id: Game id
        :param config: Configuration file related to the game (only needed to pass during creation)
        :param noise: Add noise when progressing the game
        :param overwrite: Overwrite pre-existing games
        :param save_path: Save and load the game from different directories
        :param silent: Do not print anything
        """
        # Set the game's configuration
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
        self.ray_distance = config.sensor_ray_distance
        self.ray_distance_cum = config.bot_radius + config.sensor_ray_distance
        self.target_reached = config.target_reached
        
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
        self.walls = None  # Set of all walls in the game
        
        # Check if game already exists, if not create new game
        if overwrite or not self.load(): self.create_empty_game()
    
    def __str__(self):
        return f"game_{self.id:05d}"
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    cpdef dict close(self):
        """Final state of the agent's statistics."""
        return {
            D_DIST_TO_TARGET: self.player.get_sensor_readings_distance(),
            D_DONE:           self.done,
            D_GAME_ID:        self.id,
            D_POS:            self.player.pos,
            D_STEPS:          self.steps_taken
        }
    
    cpdef dict game_params(self):
        """Get all the game-related parameters."""
        return {
            D_A_STAR:  self.path[self.player.init_pos[0], self.player.init_pos[1]],
            D_PATH:    self.path,
            D_FPS:     self.fps,
            D_GAME_ID: self.id,
        }
    
    cpdef dict get_observation(self, set close_walls=None):
        """Get the current observation of the game in the form of a dictionary."""
        return {
            D_DONE:        self.done,
            D_GAME_ID:     self.id,
            D_SENSOR_LIST: self.player.get_sensor_readings(close_walls),
        }
    
    cpdef dict reset(self):
        """Reset the game and return initial observations."""
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
        cdef float dt
        
        # Progress the game
        dt = 1.0 / self.fps + (abs(random.gauss(0, self.noise_time)) if self.noise else 0)
        return self.step_dt(dt=dt, l=l, r=r)
    
    cpdef step_dt(self, float dt, float l, float r):
        """
        Progress one step in the game based on a predefined delta-time. This method should only be used for debugging or
        visualization purposes.

        :param dt: Delta time
        :param l: Left wheel speed [-1..1]
        :param r: Right wheel speed [-1..1]
        :return: Observation (Dictionary), target_reached (Boolean)
        """
        cdef Line2dCy wall
        cdef set close_walls
        cdef bint inter
        
        # Progress the game
        self.steps_taken += 1
        self.player.drive(dt, lw=l, rw=r)
        
        # Check if intersected with a wall, if so then set player back to old position
        close_walls = {w for w in self.walls if w.close_by(pos=self.player.pos, r=self.ray_distance_cum)}
        for wall in close_walls:
            inter, _ = circle_line_intersection_cy(c=self.player.pos, r=self.player.radius, l=wall)
            if inter:
                self.player.pos.x = self.player.prev_pos.x
                self.player.pos.y = self.player.prev_pos.y
                self.player.angle = self.player.prev_angle
                break
        
        # Check if target reached
        if self.player.get_sensor_readings_distance() <= self.target_reached: self.done = True
        
        # Return the current observations
        return self.get_observation(close_walls)
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    cpdef void create_empty_game(self):
        """Create an empty game that only contains the boundary walls."""
        # Create random set of walls
        self.walls = get_boundary_walls(x_axis=self.x_axis, y_axis=self.y_axis)
        self.target = Vec2dCy(0.5, self.y_axis - 0.5)
        self.player = MarXBotCy(game=self,
                                init_pos=Vec2dCy(self.x_axis - 0.5, 0.5),
                                init_orient=np.pi / 2)
        
        # Save the new game
        self.save()
        if not self.silent: print(f"New game created under id: {self.id}")
    
    cpdef void set_player_angle(self, float a):
        """Set a new initial angle for the player."""
        self.player.init_angle = a
        self.player.angle = a
    
    cpdef void set_player_pos(self, Vec2dCy p):
        """Set a new initial position for the player."""
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
        store_pickle(persist_dict, f'{self.save_path}{self}')
    
    cpdef bint load(self):
        """
        Load in a game, specified by its current id.

        :return: True: game successfully loaded | False: otherwise
        """
        cdef dict game
        try:
            game = load_pickle(f'{self.save_path}{self}')
            self.player = MarXBotCy(game=self)  # Create a dummy-player to set values on
            self.set_player_angle(game[D_ANGLE])
            self.set_player_pos(Vec2dCy(game[D_POS][0], game[D_POS][1]))
            self.path = {p[0]: p[1] for p in game[D_PATH]}
            self.target = Vec2dCy(game[D_TARGET][0], game[D_TARGET][1])
            self.walls = {Line2dCy(Vec2dCy(w[0][0], w[0][1]), Vec2dCy(w[1][0], w[1][1])) for w in game[D_WALLS]}
            if not self.silent: print(f"Existing game loaded with id: {self.id}")
            return True
        except FileNotFoundError:
            return False
    
    cpdef get_blueprint(self, ax=None):
        """
        :return: The blue-print map of the board (matplotlib Figure)
        """
        cdef list walls
        cdef Line2dCy w
        
        if not ax: fig, ax = plt.subplots()
        
        # Draw all the walls
        walls = []
        for w in self.walls: walls.append([(w.x.x, w.x.y), (w.y.x, w.y.y)])
        lc = mc.LineCollection(walls, linewidths=2, colors='k')
        ax.add_collection(lc)
        
        # Add target to map
        plt.plot(self.target.x, self.target.y, 'go')
        
        # Adjust the boundaries
        plt.xlim(0, self.x_axis)
        plt.ylim(0, self.y_axis)
        
        # Return the figure in its current state
        return ax


cpdef set get_boundary_walls(int x_axis, int y_axis):
    """Get a set of the boundary walls."""
    a = Vec2dCy(0, 0)
    b = Vec2dCy(x_axis, 0)
    c = Vec2dCy(x_axis, y_axis)
    d = Vec2dCy(0, y_axis)
    return {Line2dCy(a, b), Line2dCy(b, c), Line2dCy(c, d), Line2dCy(d, a)}


cpdef GameCy get_game_cy(int i, cfg):
    """
    Create a game-object.
    
    :param i: Game-ID
    :param cfg: GameConfig object
    :return: Game or GameCy object
    """
    return GameCy(game_id=i,
                  config=cfg,
                  silent=True)

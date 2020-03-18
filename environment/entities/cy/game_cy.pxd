"""
game_cy.pxd

Used to declare all the game_cy class and method that must be callable from outside of other objects.
"""
from environment.entities.cy.robots_cy cimport MarXBotCy
from utils.cy.vec2d_cy cimport Vec2dCy

cdef class GameCy:
    """
    A game environment is built up from the following segments:
        * walls: Set of Line2d objects representing the walls in the maze
        * robot: The player manoeuvring in the environment
        * target: Robot that must be reached by the robot
    """
    cdef public bint done, noise, silent
    cdef public int id, steps_taken
    cdef public dict path
    cdef public set walls
    cdef public MarXBotCy player
    cdef public Vec2dCy target
    cdef public float bot_driving_speed, bot_radius, bot_turning_speed
    cdef public int batch, duration, max_game_id, max_eval_game_id, fps, p2m, x_axis, y_axis
    cdef public float noise_time, noise_angle, noise_distance, noise_proximity, ray_distance, ray_distance_cum, target_reached
    cdef public str save_path
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    cpdef dict close(self)
    
    cpdef dict game_params(self)
    
    cpdef dict get_observation(self, set close_walls=?)
    
    cpdef dict reset(self)
    
    cpdef step(self, float l, float r)
    
    cpdef step_dt(self, float dt, float l, float r)
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    cpdef void create_empty_game(self)
    
    cpdef void set_player_angle(self, float a)
    
    cpdef void set_player_pos(self, Vec2dCy p)
    
    # ---------------------------------------------> FUNCTIONAL METHODS <--------------------------------------------- #
    
    cpdef void save(self)
    
    cpdef bint load(self)
    
    cpdef get_blueprint(self, ax=?)

cpdef set get_boundary_walls(int x_axis, int y_axis)

cpdef GameCy get_game_cy(int i, cfg)

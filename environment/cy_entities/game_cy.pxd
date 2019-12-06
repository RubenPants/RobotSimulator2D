"""
game_cy.pxd

Used to declare all the game_cy class and method that must be callable from outside of other objects.
"""
from robots_cy cimport FootBotCy
from vec2d_cy cimport Vec2dCy

cdef class GameCy:
    """
    A game environment is built up from the following segments:
        * walls: Set of Line2d objects representing the walls in the maze
        * robot: The player manoeuvring in the environment
        * target: Robot that must be reached by the robot
    """
    cdef public bint done
    cdef public int id
    cdef public bint noise
    cdef public dict path
    cdef public FootBotCy player
    cdef public str rel_path
    cdef public bint silent
    cdef public int steps_taken
    cdef public Vec2dCy target
    cdef public list walls
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    cpdef dict close(self)
    
    cpdef dict reset(self)
    
    cpdef step(self, float l, float r)
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    cpdef void create_empty_game(self)
    
    cpdef dict get_observation(self)
    
    cpdef list get_sensor_list(self)
    
    cpdef void set_player_angle(self, float a)
    
    cpdef void set_player_pos(self, Vec2dCy p)
    
    # ---------------------------------------------> FUNCTIONAL METHODS <--------------------------------------------- #
    
    cpdef void save(self)
    
    cpdef bint load(self)
    
    cpdef get_blueprint(self)

cpdef list get_boundary_walls()

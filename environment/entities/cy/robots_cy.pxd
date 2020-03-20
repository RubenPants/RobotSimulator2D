"""
robots_cy.pxd

Used to declare all the methods and classes inside of robots_cy that must be callable from outside of other objects.
"""
from environment.entities.cy.game_cy cimport GameCy
from utils.cy.vec2d_cy cimport Vec2dCy

cdef class MarXBotCy:
    """
    The FootBot is the main bot used in this project. It is a simple circular robot with two wheels on its sides.
    """
    cdef public GameCy game
    cdef public Vec2dCy pos, prev_pos, init_pos
    cdef public float angle, prev_angle, init_angle, radius
    cdef public dict sensors
    cdef public set active_sensors
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    cpdef void drive(self, float dt, float lw, float rw)
    
    cpdef list get_sensor_readings(self, set close_walls=?)
    
    cpdef float get_sensor_readings_distance(self)
    
    # -----------------------------------------------> SENSOR METHODS <----------------------------------------------- #
    
    cpdef void add_angular_sensors(self, bint clockwise=?)
    
    cpdef void add_distance_sensor(self)
    
    cpdef void add_proximity_sensor(self, float angle)
    
    cpdef void create_angular_sensors(self)
    
    cpdef void create_proximity_sensors(self)
    
    cpdef list get_proximity_sensors(self)
    
    cpdef void set_active_sensors(self, set connections)


cpdef set get_active_sensors(set connections, int total_input_size)

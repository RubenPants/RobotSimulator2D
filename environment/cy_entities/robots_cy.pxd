"""
robots_cy.pxd

Used to declare all the methods and classes inside of robots_cy that must be callable from outside of other objects.
"""
from game_cy cimport GameCy
from sensors_cy cimport DistanceSensorCy
from vec2d_cy cimport Vec2dCy

cdef class FootBotCy:
    """
    The FootBot is the main bot used in this project. It is a simple circular robot with two wheels on its sides.
    """
    cdef public GameCy game
    cdef public Vec2dCy pos
    cdef public Vec2dCy prev_pos
    cdef public Vec2dCy init_pos
    cdef public float angle
    cdef public float prev_angle
    cdef public float init_angle
    cdef public float radius
    cdef public set angular_sensors
    cdef public DistanceSensorCy distance_sensor
    cdef public set proximity_sensors
    
    # ------------------------------------------------> MAIN METHODS <------------------------------------------------ #
    
    cdef void drive(self, float dt, float lw, float rw)
    
    cdef dict get_sensor_readings(self)
    
    # -----------------------------------------------> HELPER METHODS <----------------------------------------------- #
    
    cpdef dict get_sensor_reading_angle(self)
    
    cpdef float get_sensor_reading_distance(self)
    
    cpdef dict get_sensor_reading_proximity(self)
    
    # -----------------------------------------------> SENSOR METHODS <----------------------------------------------- #
    
    cpdef void add_angular_sensors(self, bint clockwise=?)
    
    cpdef void add_proximity_sensor(self, float angle)
    
    cdef void create_angular_sensors(self)
    
    cdef void create_distance_sensor(self)
    
    cdef void create_proximity_sensors(self)

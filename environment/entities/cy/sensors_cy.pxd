"""
sensors_cy.pxd

Used to declare all the methods and classes inside of sensors_cy that must be callable from outside of other objects.
"""
from environment.entities.cy.game_cy cimport GameCy
from utils.cy.vec2d_cy cimport Vec2dCy


cdef class SensorCy:
    cdef public GameCy game
    cdef public int id
    cdef public float angle, pos_offset, max_dist, value
    
    cpdef void measure(self, set close_walls=?)


cdef class AngularSensorCy(SensorCy):
    """
    Angle deviation between bot and wanted direction in 'crow flight'.
    """
    cdef public bint clockwise
    
    cpdef void measure(self, set close_walls=?)


cdef class DistanceSensorCy(SensorCy):
    """
    Distance from bot to the target in 'crows flight'.
    """
    
    cpdef void measure(self, set close_walls=?)


cdef class ProximitySensorCy(SensorCy):
    """
    The proximity sensor is attached to a bot's edge and measures the distance min(max_distance, object_distance). In
    other words, it returns the distance to an object in its path (only a straight line) if this distance is within a
    certain threshold, otherwise the maximum value will be returned.
    """
    cdef public Vec2dCy start_pos
    cdef public Vec2dCy end_pos
    
    cpdef void measure(self, set close_walls=?)
